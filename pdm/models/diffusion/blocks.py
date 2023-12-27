from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

from diffusers.models import DualTransformer2DModel, Transformer2DModel
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU
from diffusers.models.lora import LoRACompatibleLinear
from diffusers.models.resnet import ResnetBlock2D, Upsample2D, Downsample2D
from torch import nn
from diffusers.configuration_utils import register_to_config

from pdm.models.hypernet.gates import BlockVirtualGate
from diffusers.models.attention import BasicTransformerBlock, FeedForward
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UpBlock2D
from diffusers.utils import logging, USE_PEFT_BACKEND
from diffusers.models.attention_processor import AttnProcessor2_0, Attention

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class GEGLUGated(GEGLU):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__(dim_in, dim_out)
        self.gate = BlockVirtualGate(dim_out * 2)

    def forward(self, hidden_states, scale: float = 1.0):
        args = () if USE_PEFT_BACKEND else (scale,)
        hidden_states = self.proj(hidden_states, *args)
        hidden_states, gate = (self.gate(hidden_states)).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class FeedForwardGated(FeedForward):
    r"""
     A Gated feed-forward layer.

     Parameters:
         dim (`int`): The number of channels in the input.
         dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
         mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
         dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
         activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
         final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
     """

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            mult: int = 4,
            dropout: float = 0.0,
            activation_fn: str = "geglu",
            final_dropout: bool = False,
    ):
        super().__init__(dim, dim_out, mult, dropout, activation_fn, final_dropout)
        inner_dim = int(dim * mult)

        if activation_fn == "geglu":
            act_fn = GEGLUGated(dim, inner_dim)
            self.net[0] = act_fn

    def set_virtual_gate(self, gate_val):
        self.net[0].gate.set_structure_value(gate_val)

    def get_gate_structure(self):
        return self.net[0].gate.width


class GatedAttnProcessor2(AttnProcessor2_0):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = (
            attn.to_k(encoder_hidden_states, scale=scale) if not USE_PEFT_BACKEND else attn.to_k(encoder_hidden_states)
        )
        value = (
            attn.to_v(encoder_hidden_states, scale=scale) if not USE_PEFT_BACKEND else attn.to_v(encoder_hidden_states)
        )

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # apply gate
        query = attn.gate(query)
        key = attn.gate(key)
        value = attn.gate(value)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = (
            attn.to_out[0](hidden_states, scale=scale) if not USE_PEFT_BACKEND else attn.to_out[0](hidden_states)
        )
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class ResnetBlock2DGated(ResnetBlock2D):
    def __init__(self, *args, **kwargs):
        # extract gate_flag from kwargs
        super().__init__(*args, **kwargs)
        self.gate = BlockVirtualGate(self.norm1.num_groups)

    def forward(self, input_tensor, temb, scale: float = 1.0):
        hidden_states = input_tensor

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = (
                self.upsample(input_tensor, scale=scale)
                if isinstance(self.upsample, Upsample2D)
                else self.upsample(input_tensor)
            )
            hidden_states = (
                self.upsample(hidden_states, scale=scale)
                if isinstance(self.upsample, Upsample2D)
                else self.upsample(hidden_states)
            )
        elif self.downsample is not None:
            input_tensor = (
                self.downsample(input_tensor, scale=scale)
                if isinstance(self.downsample, Downsample2D)
                else self.downsample(input_tensor)
            )
            hidden_states = (
                self.downsample(hidden_states, scale=scale)
                if isinstance(self.downsample, Downsample2D)
                else self.downsample(hidden_states)
            )

        hidden_states = self.conv1(hidden_states, scale) if not USE_PEFT_BACKEND else self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = (
                self.time_emb_proj(temb, scale)[:, :, None, None]
                if not USE_PEFT_BACKEND
                else self.time_emb_proj(temb)[:, :, None, None]
            )

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        # DO it here or after norm2?
        hidden_states = self.gate(hidden_states)

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, scale) if not USE_PEFT_BACKEND else self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = (
                self.conv_shortcut(input_tensor, scale) if not USE_PEFT_BACKEND else self.conv_shortcut(input_tensor)
            )

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor

    def set_virtual_gate(self, gate_val):
        self.gate.set_structure_value(gate_val)

    def get_gate_structure(self):
        return self.gate.width


class ResnetBlock2DWidthDepthGated(ResnetBlock2D):
    def __init__(self, *args, **kwargs):
        # extract gate_flag from kwargs
        super().__init__(*args, **kwargs)
        self.gate = BlockVirtualGate(self.norm1.num_groups)
        self.depth_gate = BlockVirtualGate(1)

    def forward(self, input_tensor, temb, scale: float = 1.0):
        hidden_states = input_tensor

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = (
                self.upsample(input_tensor, scale=scale)
                if isinstance(self.upsample, Upsample2D)
                else self.upsample(input_tensor)
            )
            hidden_states = (
                self.upsample(hidden_states, scale=scale)
                if isinstance(self.upsample, Upsample2D)
                else self.upsample(hidden_states)
            )
        elif self.downsample is not None:
            input_tensor = (
                self.downsample(input_tensor, scale=scale)
                if isinstance(self.downsample, Downsample2D)
                else self.downsample(input_tensor)
            )
            hidden_states = (
                self.downsample(hidden_states, scale=scale)
                if isinstance(self.downsample, Downsample2D)
                else self.downsample(hidden_states)
            )

        hidden_states = self.conv1(hidden_states, scale) if not USE_PEFT_BACKEND else self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = (
                self.time_emb_proj(temb, scale)[:, :, None, None]
                if not USE_PEFT_BACKEND
                else self.time_emb_proj(temb)[:, :, None, None]
            )

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        # DO it here or after norm2?
        hidden_states = self.gate(hidden_states)

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, scale) if not USE_PEFT_BACKEND else self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = (
                self.conv_shortcut(input_tensor, scale) if not USE_PEFT_BACKEND else self.conv_shortcut(input_tensor)
            )

        hidden_states = self.depth_gate(hidden_states)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor

    def set_virtual_gate(self, gate_val):
        self.gate.set_structure_value(gate_val[:, :-1])
        self.depth_gate.set_structure_value(gate_val[:, -1:])

    def get_gate_structure(self):
        return self.gate.width + self.depth_gate.width


class BasicTransformerBlockGated(BasicTransformerBlock):
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_type: str = "layer_norm",
            final_dropout: bool = False,
            attention_type: str = "default",
    ):
        super().__init__(dim, num_attention_heads, attention_head_dim, dropout, cross_attention_dim, activation_fn,
                         num_embeds_ada_norm, attention_bias, only_cross_attention, double_self_attention,
                         upcast_attention, norm_elementwise_affine, norm_type, final_dropout, attention_type)
        self.num_attention_heads = b=num_attention_heads
        self.attention_head_dim = attention_head_dim
        gate1 = BlockVirtualGate(self.num_attention_heads)
        if self.attn1 is not None:
            self.attn1.set_processor(GatedAttnProcessor2())
            self.attn1.gate = gate1

        gate2 = BlockVirtualGate(self.num_attention_heads)
        if self.attn2 is not None:
            self.attn2.set_processor(GatedAttnProcessor2())
            self.attn2.gate = gate2

        self.ff = FeedForwardGated(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])
        # 2.5 ends

        # 3. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [
                    self.ff(hid_slice, scale=lora_scale)
                    for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)
                ],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states

    def set_virtual_gate(self, gate_val):
        gate_1_val = gate_val[:, :self.attn1.gate.width]
        gate_2_val = gate_val[:, self.attn1.gate.width:self.attn1.gate.width + self.attn2.gate.width]
        gate_3_val = gate_val[:, self.attn1.gate.width + self.attn2.gate.width:]
        self.attn1.gate.set_structure_value(gate_1_val)
        self.attn2.gate.set_structure_value(gate_2_val)
        self.ff.set_virtual_gate(gate_3_val)

    def get_gate_structure(self):
        return self.attn1.gate.width + self.attn2.gate.width + self.ff.get_gate_structure()


class BasicTransformerBlockWidthDepthGated(BasicTransformerBlock):
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_type: str = "layer_norm",
            final_dropout: bool = False,
            attention_type: str = "default",
    ):
        super().__init__(dim, num_attention_heads, attention_head_dim, dropout, cross_attention_dim, activation_fn,
                         num_embeds_ada_norm, attention_bias, only_cross_attention, double_self_attention,
                         upcast_attention, norm_elementwise_affine, norm_type, final_dropout, attention_type)
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        gate1 = BlockVirtualGate(self.num_attention_heads)
        if self.attn1 is not None:
            self.attn1.set_processor(GatedAttnProcessor2())
            self.attn1.gate = gate1
        gate2 = BlockVirtualGate(self.num_attention_heads)
        if self.attn2 is not None:
            self.attn2.set_processor(GatedAttnProcessor2())
            self.attn2.gate = gate2

        self.ff = FeedForwardGated(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        self.depth_gate = BlockVirtualGate(1)

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])
        # 2.5 ends

        # 3. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [
                    self.ff(hid_slice, scale=lora_scale)
                    for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)
                ],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        hidden_states = self.depth_gate(hidden_states)

        return hidden_states

    def set_virtual_gate(self, gate_val):
        gate_1_val = gate_val[:, :self.attn1.gate.width]
        gate_2_val = gate_val[:, self.attn1.gate.width:self.attn1.gate.width + self.attn2.gate.width]
        gate_3_val = gate_val[:, self.attn1.gate.width + self.attn2.gate.width:-1]
        self.attn1.gate.set_structure_value(gate_1_val)
        self.attn2.gate.set_structure_value(gate_2_val)
        self.ff.set_virtual_gate(gate_3_val)
        self.depth_gate.set_structure_value(gate_val[:, -1:])

    def get_gate_structure(self):
        return self.attn1.gate.width + self.attn2.gate.width + self.ff.get_gate_structure() + self.depth_gate.width


class Transformer2DModelGated(Transformer2DModel):
    @register_to_config
    def __init__(
            self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 88,
            in_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            num_layers: int = 1,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            cross_attention_dim: Optional[int] = None,
            attention_bias: bool = False,
            sample_size: Optional[int] = None,
            num_vector_embeds: Optional[int] = None,
            patch_size: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            use_linear_projection: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_type: str = "layer_norm",
            norm_elementwise_affine: bool = True,
            attention_type: str = "default",
    ):
        super().__init__(num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim,
                         in_channels=in_channels, out_channels=out_channels, num_layers=num_layers, dropout=dropout,
                         norm_num_groups=norm_num_groups, cross_attention_dim=cross_attention_dim,
                         attention_bias=attention_bias, sample_size=sample_size, num_vector_embeds=num_vector_embeds,
                         patch_size=patch_size, activation_fn=activation_fn,
                         num_embeds_ada_norm=num_embeds_ada_norm, use_linear_projection=use_linear_projection,
                         only_cross_attention=only_cross_attention, double_self_attention=double_self_attention,
                         upcast_attention=upcast_attention, norm_type=norm_type,
                         norm_elementwise_affine=norm_elementwise_affine, attention_type=attention_type)

        inner_dim = num_attention_heads * attention_head_dim

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlockGated(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    attention_type=attention_type,
                )
                for _ in range(num_layers)
            ]
        )


class Transformer2DModelWidthDepthGated(Transformer2DModel):
    @register_to_config
    def __init__(
            self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 88,
            in_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            num_layers: int = 1,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            cross_attention_dim: Optional[int] = None,
            attention_bias: bool = False,
            sample_size: Optional[int] = None,
            num_vector_embeds: Optional[int] = None,
            patch_size: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            use_linear_projection: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_type: str = "layer_norm",
            norm_elementwise_affine: bool = True,
            attention_type: str = "default",
    ):
        super().__init__(num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim,
                         in_channels=in_channels, out_channels=out_channels, num_layers=num_layers, dropout=dropout,
                         norm_num_groups=norm_num_groups, cross_attention_dim=cross_attention_dim,
                         attention_bias=attention_bias, sample_size=sample_size, num_vector_embeds=num_vector_embeds,
                         patch_size=patch_size, activation_fn=activation_fn,
                         num_embeds_ada_norm=num_embeds_ada_norm, use_linear_projection=use_linear_projection,
                         only_cross_attention=only_cross_attention, double_self_attention=double_self_attention,
                         upcast_attention=upcast_attention, norm_type=norm_type,
                         norm_elementwise_affine=norm_elementwise_affine, attention_type=attention_type)

        inner_dim = num_attention_heads * attention_head_dim

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlockWidthDepthGated(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    attention_type=attention_type,
                )
                for _ in range(num_layers)
            ]
        )


class DualTransformer2DModelGated(DualTransformer2DModel):
    def __init__(
            self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 88,
            in_channels: Optional[int] = None,
            num_layers: int = 1,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            cross_attention_dim: Optional[int] = None,
            attention_bias: bool = False,
            sample_size: Optional[int] = None,
            num_vector_embeds: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
    ):
        super().__init__(num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim,
                         in_channels=in_channels, num_layers=num_layers, dropout=dropout,
                         norm_num_groups=norm_num_groups, cross_attention_dim=cross_attention_dim,
                         attention_bias=attention_bias, sample_size=sample_size, num_vector_embeds=num_vector_embeds,
                         activation_fn=activation_fn, num_embeds_ada_norm=num_embeds_ada_norm)

        self.transformers = nn.ModuleList(
            [
                Transformer2DModelGated(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    in_channels=in_channels,
                    num_layers=num_layers,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    attention_bias=attention_bias,
                    sample_size=sample_size,
                    num_vector_embeds=num_vector_embeds,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                )
                for _ in range(2)
            ]
        )


class DualTransformer2DModelWidthDepthGated(DualTransformer2DModel):
    def __init__(
            self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 88,
            in_channels: Optional[int] = None,
            num_layers: int = 1,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            cross_attention_dim: Optional[int] = None,
            attention_bias: bool = False,
            sample_size: Optional[int] = None,
            num_vector_embeds: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
    ):
        super().__init__(num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim,
                         in_channels=in_channels, num_layers=num_layers, dropout=dropout,
                         norm_num_groups=norm_num_groups, cross_attention_dim=cross_attention_dim,
                         attention_bias=attention_bias, sample_size=sample_size, num_vector_embeds=num_vector_embeds,
                         activation_fn=activation_fn, num_embeds_ada_norm=num_embeds_ada_norm)

        self.transformers = nn.ModuleList(
            [
                Transformer2DModelWidthDepthGated(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    in_channels=in_channels,
                    num_layers=num_layers,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    attention_bias=attention_bias,
                    sample_size=sample_size,
                    num_vector_embeds=num_vector_embeds,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                )
                for _ in range(2)
            ]
        )


class CrossAttnDownBlock2DGated(CrossAttnDownBlock2D):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            transformer_layers_per_block: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            num_attention_heads=1,
            cross_attention_dim=1280,
            output_scale_factor=1.0,
            downsample_padding=1,
            add_downsample=True,
            dual_cross_attention=False,
            use_linear_projection=False,
            only_cross_attention=False,
            upcast_attention=False,
            attention_type="default",
    ):
        super().__init__(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels,
                         dropout=dropout, num_layers=num_layers,
                         transformer_layers_per_block=transformer_layers_per_block, resnet_eps=resnet_eps,
                         resnet_time_scale_shift=resnet_time_scale_shift, resnet_act_fn=resnet_act_fn,
                         resnet_groups=resnet_groups, resnet_pre_norm=resnet_pre_norm,
                         num_attention_heads=num_attention_heads, cross_attention_dim=cross_attention_dim,
                         output_scale_factor=output_scale_factor, downsample_padding=downsample_padding,
                         add_downsample=add_downsample, dual_cross_attention=dual_cross_attention,
                         use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention,
                         upcast_attention=upcast_attention, attention_type=attention_type)
        resnets = []
        attentions = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2DWidthDepthGated(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModelWidthDepthGated(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModelWidthDepthGated(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)


class CrossAttnDownBlock2DHalfGated(CrossAttnDownBlock2D):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            transformer_layers_per_block: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            num_attention_heads=1,
            cross_attention_dim=1280,
            output_scale_factor=1.0,
            downsample_padding=1,
            add_downsample=True,
            dual_cross_attention=False,
            use_linear_projection=False,
            only_cross_attention=False,
            upcast_attention=False,
            attention_type="default",
    ):
        super().__init__(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels,
                         dropout=dropout, num_layers=num_layers,
                         transformer_layers_per_block=transformer_layers_per_block, resnet_eps=resnet_eps,
                         resnet_time_scale_shift=resnet_time_scale_shift, resnet_act_fn=resnet_act_fn,
                         resnet_groups=resnet_groups, resnet_pre_norm=resnet_pre_norm,
                         num_attention_heads=num_attention_heads, cross_attention_dim=cross_attention_dim,
                         output_scale_factor=output_scale_factor, downsample_padding=downsample_padding,
                         add_downsample=add_downsample, dual_cross_attention=dual_cross_attention,
                         use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention,
                         upcast_attention=upcast_attention, attention_type=attention_type)
        resnets = []
        attentions = []

        for i in range(num_layers // 2):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2DGated(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModelGated(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModelGated(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )

        for i in range(num_layers // 2, num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2DWidthDepthGated(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModelWidthDepthGated(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModelWidthDepthGated(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)


class CrossAttnUpBlock2DGated(CrossAttnUpBlock2D):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            prev_output_channel: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            transformer_layers_per_block: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            num_attention_heads=1,
            cross_attention_dim=1280,
            output_scale_factor=1.0,
            add_upsample=True,
            dual_cross_attention=False,
            use_linear_projection=False,
            only_cross_attention=False,
            upcast_attention=False,
            attention_type="default",
    ):
        super().__init__(in_channels=in_channels, out_channels=out_channels, prev_output_channel=prev_output_channel,
                         temb_channels=temb_channels, dropout=dropout, num_layers=num_layers,
                         transformer_layers_per_block=transformer_layers_per_block, resnet_eps=resnet_eps,
                         resnet_time_scale_shift=resnet_time_scale_shift, resnet_act_fn=resnet_act_fn,
                         resnet_groups=resnet_groups, resnet_pre_norm=resnet_pre_norm,
                         num_attention_heads=num_attention_heads, cross_attention_dim=cross_attention_dim,
                         output_scale_factor=output_scale_factor, add_upsample=add_upsample,
                         dual_cross_attention=dual_cross_attention, use_linear_projection=use_linear_projection,
                         only_cross_attention=only_cross_attention, upcast_attention=upcast_attention,
                         attention_type=attention_type)

        resnets = []
        attentions = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2DWidthDepthGated(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModelWidthDepthGated(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModelWidthDepthGated(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)


class CrossAttnUpBlock2DHalfGated(CrossAttnUpBlock2D):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            prev_output_channel: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            transformer_layers_per_block: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            num_attention_heads=1,
            cross_attention_dim=1280,
            output_scale_factor=1.0,
            add_upsample=True,
            dual_cross_attention=False,
            use_linear_projection=False,
            only_cross_attention=False,
            upcast_attention=False,
            attention_type="default",
    ):
        super().__init__(in_channels=in_channels, out_channels=out_channels, prev_output_channel=prev_output_channel,
                         temb_channels=temb_channels, dropout=dropout, num_layers=num_layers,
                         transformer_layers_per_block=transformer_layers_per_block, resnet_eps=resnet_eps,
                         resnet_time_scale_shift=resnet_time_scale_shift, resnet_act_fn=resnet_act_fn,
                         resnet_groups=resnet_groups, resnet_pre_norm=resnet_pre_norm,
                         num_attention_heads=num_attention_heads, cross_attention_dim=cross_attention_dim,
                         output_scale_factor=output_scale_factor, add_upsample=add_upsample,
                         dual_cross_attention=dual_cross_attention, use_linear_projection=use_linear_projection,
                         only_cross_attention=only_cross_attention, upcast_attention=upcast_attention,
                         attention_type=attention_type)

        resnets = []
        attentions = []

        for i in range(num_layers // 2):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2DGated(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModelGated(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModelGated(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )

        for i in range(num_layers // 2, num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2DWidthDepthGated(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModelWidthDepthGated(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModelWidthDepthGated(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)


class DownBlock2DGated(DownBlock2D):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            output_scale_factor=1.0,
            add_downsample=True,
            downsample_padding=1,
    ):
        super().__init__(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels,
                         dropout=dropout, num_layers=num_layers, resnet_eps=resnet_eps,
                         resnet_time_scale_shift=resnet_time_scale_shift, resnet_act_fn=resnet_act_fn,
                         resnet_groups=resnet_groups, resnet_pre_norm=resnet_pre_norm,
                         output_scale_factor=output_scale_factor, add_downsample=add_downsample,
                         downsample_padding=downsample_padding)
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2DWidthDepthGated(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)


class DownBlock2DHalfGated(DownBlock2D):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            output_scale_factor=1.0,
            add_downsample=True,
            downsample_padding=1,
    ):
        super().__init__(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels,
                         dropout=dropout, num_layers=num_layers, resnet_eps=resnet_eps,
                         resnet_time_scale_shift=resnet_time_scale_shift, resnet_act_fn=resnet_act_fn,
                         resnet_groups=resnet_groups, resnet_pre_norm=resnet_pre_norm,
                         output_scale_factor=output_scale_factor, add_downsample=add_downsample,
                         downsample_padding=downsample_padding)
        resnets = []

        for i in range(num_layers // 2):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2DGated(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        for i in range(num_layers // 2, num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2DWidthDepthGated(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)


class UpBlock2DGated(UpBlock2D):
    def __init__(
            self,
            in_channels: int,
            prev_output_channel: int,
            out_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            output_scale_factor=1.0,
            add_upsample=True,
    ):
        super().__init__(in_channels=in_channels, prev_output_channel=prev_output_channel, out_channels=out_channels,
                         temb_channels=temb_channels, dropout=dropout, num_layers=num_layers, resnet_eps=resnet_eps,
                         resnet_time_scale_shift=resnet_time_scale_shift, resnet_act_fn=resnet_act_fn,
                         resnet_groups=resnet_groups, resnet_pre_norm=resnet_pre_norm,
                         output_scale_factor=output_scale_factor, add_upsample=add_upsample)
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2DWidthDepthGated(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)


class UpBlock2DHalfGated(UpBlock2D):
    def __init__(
            self,
            in_channels: int,
            prev_output_channel: int,
            out_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            output_scale_factor=1.0,
            add_upsample=True,
    ):
        super().__init__(in_channels=in_channels, prev_output_channel=prev_output_channel, out_channels=out_channels,
                         temb_channels=temb_channels, dropout=dropout, num_layers=num_layers, resnet_eps=resnet_eps,
                         resnet_time_scale_shift=resnet_time_scale_shift, resnet_act_fn=resnet_act_fn,
                         resnet_groups=resnet_groups, resnet_pre_norm=resnet_pre_norm,
                         output_scale_factor=output_scale_factor, add_upsample=add_upsample)
        resnets = []

        for i in range(num_layers // 2):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2DGated(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        for i in range(num_layers // 2, num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2DWidthDepthGated(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)
