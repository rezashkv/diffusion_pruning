from typing import Optional, Dict, Any, Union, Tuple

import torch
import torch.nn.functional as F

from diffusers.models import DualTransformer2DModel, Transformer2DModel
from diffusers.models.activations import GEGLU
from diffusers.models.resnet import ResnetBlock2D, Upsample2D, Downsample2D
from diffusers.models.transformer_2d import Transformer2DModelOutput
from torch import nn
from diffusers.configuration_utils import register_to_config

from pdm.models.hypernet.gates import BlockVirtualGate, LinearVirtualGate
from pdm.models.hypernet.gates import DepthGate, WidthGate
from diffusers.models.attention import BasicTransformerBlock, FeedForward
from diffusers.models.unet_2d_blocks import (CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UpBlock2D,
                                             UNetMidBlock2DCrossAttn)
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
        # self.gate = LinearVirtualGate(dim_out)
        self.gate = WidthGate(dim_out)

    def forward(self, hidden_states, scale: float = 1.0):
        args = () if USE_PEFT_BACKEND else (scale,)
        hidden_states, gate = self.proj(hidden_states, *args).chunk(2, dim=-1)
        hidden_states, gate = (self.gate(hidden_states), self.gate(gate))
        return hidden_states * self.gelu(gate)


class FeedForwardWidthGated(FeedForward):
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
        return {"width": [self.net[0].gate.width]}


class HeadGatedAttnProcessor2(AttnProcessor2_0):
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

        # ########## Apply Width Gate
        assert key.shape[1] == attn.gate.gate_f.shape[1]
        mask = attn.gate.gate_f.unsqueeze(-1).unsqueeze(-1)
        query = query * mask
        key = key * mask
        value = value * mask
        # query = attn.gate(query)
        # key = attn.gate(key)
        # value = attn.gate(value)

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


class ResnetBlock2DWidthGated(ResnetBlock2D):
    def __init__(self, is_input_concatenated=False, *args, **kwargs):
        # extract gate_flag from kwargs
        super().__init__(*args, **kwargs)
        # self.gate = BlockVirtualGate(self.norm1.num_groups)
        self.gate = WidthGate(self.norm1.num_groups)
        self.is_input_concatenated = is_input_concatenated
        self.structure = {'width': [], 'depth': []}

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
        assert self.norm2.num_groups == self.gate.gate_f.shape[1]
        num_repeat = int(hidden_states.shape[1] / self.norm2.num_groups)
        mask = torch.repeat_interleave(self.gate.gate_f, repeats=num_repeat, dim=1).unsqueeze(-1).unsqueeze(-1)
        hidden_states = hidden_states * mask
        # hidden_states = self.gate(hidden_states)

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
        if self.structure["width"] == []:
            self.structure = {"width": [self.gate.width], "depth": [0]}
        return self.structure

    def set_gate_structure(self, arch_vectors):
        assert len(arch_vectors['depth']) == 0
        assert len(arch_vectors['width']) == 1
        assert arch_vectors['width'][0].shape[1] == self.gate.width
        self.gate.set_structure_value(arch_vectors['width'][0])


class ResnetBlock2DWidthDepthGated(ResnetBlock2D):
    def __init__(self, is_input_concatenated=False, *args, **kwargs):
        # extract gate_flag from kwargs
        super().__init__(*args, **kwargs)
        # self.gate = BlockVirtualGate(self.norm1.num_groups)
        # self.depth_gate = BlockVirtualGate(1)
        self.gate = WidthGate(self.norm1.num_groups)
        self.depth_gate = DepthGate(1)
        self.is_input_concatenated = is_input_concatenated
        self.structure = {'width': [], 'depth': []}

    def forward(self, input_tensor, temb, scale: float = 1.0):
        assert (self.upsample is None) and (self.downsample is None) # Depth gate cannot be in the up/down sample blocks.
        if self.is_input_concatenated:  # We are in the upsample blocks, input is concatenated.
            input_hidden_states = input_tensor.chunk(2, dim=1)[0]  # [0] because the forward pass is hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1) 
            # in here: https://github.com/huggingface/diffusers/blob/acd926f4f208e4cf12be69315787c450da48913b/src/diffusers/models/unet_2d_blocks.py#L2324
             
        else:  # We are in the downsample blocks
            input_hidden_states = input_tensor

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
        assert self.norm2.num_groups == self.gate.gate_f.shape[1]
        num_repeat = int(hidden_states.shape[1] / self.norm2.num_groups)
        mask = torch.repeat_interleave(self.gate.gate_f, repeats=num_repeat, dim=1).unsqueeze(-1).unsqueeze(-1)
        hidden_states = hidden_states * mask
        # hidden_states = self.gate(hidden_states)

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

        # ########### TODO: Depth gate
        # hidden_states = self.depth_gate(hidden_states)
        # output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        # return output_tensor
        
        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        assert torch.equal(output_tensor.shape, input_hidden_states.shape)
        output = (1 - self.depth_gate.gate_f.expand_as(input_hidden_states)) * input_hidden_states + (self.depth_gate.gate_f.expand_as(output_tensor)) * output_tensor 
        # output = self.depth_gate(output)
        return output

    def set_virtual_gate(self, gate_val):
        self.gate.set_structure_value(gate_val[:, :-1])
        self.depth_gate.set_structure_value(gate_val[:, -1:])

    def get_gate_structure(self):
        if self.structure["width"] == []:
            self.structure = {"depth": [self.depth_gate.width], "width": [self.gate.width]}
        return self.structure

    def set_gate_structure(self, arch_vectors):
        assert len(arch_vectors['depth']) == 1
        assert len(arch_vectors['width']) == 1
        assert arch_vectors['width'][0].shape[1] == self.gate.width
        self.gate.set_structure_value(arch_vectors['width'][0])
        self.depth_gate.set_structure_value(arch_vectors['depth'][0])


class BasicTransformerBlockWidthGated(BasicTransformerBlock):
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
            gated_ff: bool = False,
    ):
        super().__init__(dim, num_attention_heads, attention_head_dim, dropout, cross_attention_dim, activation_fn,
                         num_embeds_ada_norm, attention_bias, only_cross_attention, double_self_attention,
                         upcast_attention, norm_elementwise_affine, norm_type, final_dropout, attention_type)

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        # gate1 = BlockVirtualGate(self.num_attention_heads)
        gate1 = WidthGate(self.num_attention_heads)

        if self.attn1 is not None:
            self.attn1.set_processor(HeadGatedAttnProcessor2())
            self.attn1.gate = gate1

        # gate2 = BlockVirtualGate(self.num_attention_heads)
        gate2 = WidthGate(self.num_attention_heads)
        if self.attn2 is not None:
            self.attn2.set_processor(HeadGatedAttnProcessor2())
            self.attn2.gate = gate2

        self.gated_ff = gated_ff
        if gated_ff:
            self.ff = FeedForwardWidthGated(dim, dropout=dropout, activation_fn=activation_fn,
                                            final_dropout=final_dropout)
        else:
            self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        self.structure = {'width': [], 'depth': []}

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

    def get_gate_structure(self):
        if len(self.structure['width']) == 0:
            self.structure['width'] = [self.attn1.gate.width, self.attn2.gate.width]
            if isinstance(self.ff, FeedForwardWidthGated):
                self.structure['width'].append(self.ff.get_gate_structure()['width'])
            self.structure['depth'] = [0]
        return self.structure

    def set_virtual_gate(self, gate_val):
        gate_1_val = gate_val[:, :self.attn1.gate.width]
        gate_2_val = gate_val[:, self.attn1.gate.width:self.attn1.gate.width + self.attn2.gate.width]
        self.attn1.gate.set_structure_value(gate_1_val)
        self.attn2.gate.set_structure_value(gate_2_val)

        if isinstance(self.ff, FeedForwardWidthGated):
            gate_3_val = gate_val[:, self.attn1.gate.width + self.attn2.gate.width:-1]
            self.ff.set_virtual_gate(gate_3_val)

    def set_gate_structure(self, arch_vectors):
        assert len(arch_vectors['depth']) == 0
        assert len(arch_vectors['width']) >= 2

        # attn1
        assert arch_vectors['width'][0].shape[1] == self.attn1.gate.width
        self.attn1.gate.set_structure_value(arch_vectors['width'][0])

        # attn2
        assert arch_vectors['width'][1].shape[1] == self.attn2.gate.width
        self.attn2.gate.set_structure_value(arch_vectors['width'][1])

        # ff
        if self.gated_ff:
            assert len(arch_vectors['width']) == 3
            assert arch_vectors['width'][2].shape[1] == self.ff.gate.width
            self.ff.gate.set_structure_value(arch_vectors['width'][2])


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
            gated_ff: bool = False,
    ):
        super().__init__(dim, num_attention_heads, attention_head_dim, dropout, cross_attention_dim, activation_fn,
                         num_embeds_ada_norm, attention_bias, only_cross_attention, double_self_attention,
                         upcast_attention, norm_elementwise_affine, norm_type, final_dropout, attention_type)
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # gate1 = BlockVirtualGate(self.num_attention_heads)
        gate1 = WidthGate(self.num_attention_heads)
        if self.attn1 is not None:
            self.attn1.set_processor(HeadGatedAttnProcessor2())
            self.attn1.gate = gate1

        # gate2 = BlockVirtualGate(self.num_attention_heads)
        gate2 = WidthGate(self.num_attention_heads)
        if self.attn2 is not None:
            self.attn2.set_processor(HeadGatedAttnProcessor2())
            self.attn2.gate = gate2
        if gated_ff:
            self.ff = FeedForwardWidthGated(dim, dropout=dropout, activation_fn=activation_fn,
                                            final_dropout=final_dropout)
        else:
            self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        # self.depth_gate = DepthGate(1)

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

        # attn_output = self.depth_gate(attn_output)

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

            # attn_output = self.depth_gate(attn_output)
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

        # ff_output = self.depth_gate(ff_output)
        hidden_states = ff_output + hidden_states

        # hidden_states = self.depth_gate(hidden_states)
        return hidden_states

    def get_gate_structure(self):
        width = [self.attn1.gate.width, self.attn2.gate.width]
        if isinstance(self.ff, FeedForwardWidthGated):
            width.append(self.ff.get_gate_structure())
        depth = self.depth_gate.width
        return {"width": width, "depth": [depth]}

    def set_virtual_gate(self, gate_val):
        gate_1_val = gate_val[:, :self.attn1.gate.width]
        gate_2_val = gate_val[:, self.attn1.gate.width:self.attn1.gate.width + self.attn2.gate.width]
        self.attn1.gate.set_structure_value(gate_1_val)
        self.attn2.gate.set_structure_value(gate_2_val)

        if isinstance(self.ff, FeedForwardWidthGated):
            gate_3_val = gate_val[:, self.attn1.gate.width + self.attn2.gate.width:-1]
            self.ff.set_virtual_gate(gate_3_val)

        self.depth_gate.set_structure_value(gate_val[:, -1:])


class Transformer2DModelWidthGated(Transformer2DModel):
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
            gated_ff: bool = False
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
                BasicTransformerBlockWidthGated(
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
                    gated_ff=gated_ff
                )
                for _ in range(num_layers)
            ]
        )
        self.structure = {'width': [], 'depth': []}

    def get_gate_structure(self):
        if len(self.structure['width']) == 0:
            for tb in self.transformer_blocks:
                tb_structure = tb.get_gate_structure()
                # assert len(tb_structure) == 1
                self.structure['width'] = self.structure['width'] + tb_structure['width']
                # self.structure['width'].append(tb_structure['width'])
            self.structure['depth'] = [0]
        return self.structure

    def set_gate_structure(self, arch_vectors):
        if len(self.transformer_blocks) > 1:
            raise NotImplementedError

        assert len(arch_vectors['depth']) == 0
        self.transformer_blocks[0].set_gate_structure(arch_vectors)


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
            gated_ff: bool = False
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
                BasicTransformerBlockWidthGated(
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
                    gated_ff=gated_ff
                )
                for _ in range(num_layers)
            ]
        )

        self.depth_gate = DepthGate(1)
        self.structure = {'width': [], 'depth': []}

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Dict[str, torch.Tensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # ########### 1. Input
        input_hidden_states = hidden_states
        if self.is_input_continuous:
            batch, _, height, width = hidden_states.shape
            residual = hidden_states

            hidden_states = self.norm(hidden_states)
            if not self.use_linear_projection:
                hidden_states = (
                    self.proj_in(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_in(hidden_states)
                )
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            else:
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                hidden_states = (
                    self.proj_in(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_in(hidden_states)
                )

        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
            hidden_states = self.pos_embed(hidden_states)

            if self.adaln_single is not None:
                if self.use_additional_conditions and added_cond_kwargs is None:
                    raise ValueError(
                        "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                    )
                batch_size = hidden_states.shape[0]
                timestep, embedded_timestep = self.adaln_single(
                    timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
                )

        # ########### 2. Blocks
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

        # ########### 3. Output
        if self.is_input_continuous:
            if not self.use_linear_projection:
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                hidden_states = (
                    self.proj_out(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_out(hidden_states)
                )
            else:
                hidden_states = (
                    self.proj_out(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_out(hidden_states)
                )
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

            output = hidden_states + residual
        elif self.is_input_vectorized:
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
            logits = logits.permute(0, 2, 1)

            # log(p(x_0))
            output = F.log_softmax(logits.double(), dim=1).float()

        if self.is_input_patches:
            if self.config.norm_type != "ada_norm_single":
                conditioning = self.transformer_blocks[0].norm1.emb(
                    timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
                shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
                hidden_states = self.proj_out_2(hidden_states)
            elif self.config.norm_type == "ada_norm_single":
                shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states)
                # Modulation
                hidden_states = hidden_states * (1 + scale) + shift
                hidden_states = self.proj_out(hidden_states)
                hidden_states = hidden_states.squeeze(1)

            # unpatchify
            if self.adaln_single is None:
                height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )

        # ########### TODO: Depth gate
        output = (1 - self.depth_gate.gate_f.expand_as(output)) * input_hidden_states + (self.depth_gate.gate_f.expand_as(output)) * output 
        # output = self.depth_gate(output)

        if not return_dict:
            return (output_tensor,)

        return Transformer2DModelOutput(sample=output_tensor)

        # output = (1 - self.depth_gate.gate_f.expand_as(output)) * input_hidden_states + (self.depth_gate.gate_f.expand_as(output)) * output 
        # output = self.depth_gate(output)

        # if not return_dict:
        #     return (output,)

        # return Transformer2DModelOutput(sample=output)

    def get_gate_structure(self):
        if len(self.structure['width']) == 0:
            for tb in self.transformer_blocks:
                tb_structure = tb.get_gate_structure()
                # assert len(tb_structure) == 1
                self.structure['width'] = self.structure['width'] + tb_structure['width']
            self.structure['depth'].append(1)
        return self.structure

    def set_gate_structure(self, arch_vectors):
        if len(self.transformer_blocks) > 1:
            raise NotImplementedError

        assert len(arch_vectors['depth']) == 1
        self.depth_gate.set_structure_value(arch_vectors['depth'][0])
        self.transformer_blocks[0].set_gate_structure({'width': arch_vectors['width'], 'depth': []})


class DualTransformer2DModelWidthGated(DualTransformer2DModel):
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
                Transformer2DModelWidthGated(
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

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            timestep=None,
            attention_mask=None,
            cross_attention_kwargs=None,
            return_dict: bool = True,
    ):
        """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continuous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            attention_mask (`torch.FloatTensor`, *optional*):
                Optional attention mask to be applied in Attention.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `tuple`:
            [`~models.transformer_2d.Transformer2DModelOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        input_states = hidden_states

        encoded_states = []
        tokens_start = 0
        # attention_mask is not used yet
        for i in range(2):
            # for each of the two transformers, pass the corresponding condition tokens
            condition_state = encoder_hidden_states[:, tokens_start: tokens_start + self.condition_lengths[i]]
            transformer_index = self.transformer_index_for_condition[i]
            encoded_state = self.transformers[transformer_index](
                input_states,
                encoder_hidden_states=condition_state,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]
            encoded_state = self.transformers[transformer_index].depth_gate(encoded_state)
            encoded_states.append(encoded_state - input_states)
            tokens_start += self.condition_lengths[i]

        output_states = encoded_states[0] * self.mix_ratio + encoded_states[1] * (1 - self.mix_ratio)
        output_states = output_states + input_states

        if not return_dict:
            return (output_states,)

        return Transformer2DModelOutput(sample=output_states)


class CrossAttnDownBlock2DWidthDepthGated(CrossAttnDownBlock2D):
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


class CrossAttnDownBlock2DWidthHalfDepthGated(CrossAttnDownBlock2D):
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
            gated_ff: bool = False
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

        for i in range(num_layers - 1):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2DWidthGated(
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
                    is_input_concatenated=False
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModelWidthGated(
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
                        gated_ff=gated_ff
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModelWidthGated(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )

        for i in range(num_layers - 1, num_layers):
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
                    is_input_concatenated=False
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
                        gated_ff=gated_ff
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
        self.structure = {'width': [], 'depth': []}

    def get_gate_structure(self):
        if len(self.structure['width']) == 0:
            structure = {'width': [], 'depth': []}
            for b in self.resnets:
                assert hasattr(b, "get_gate_structure")
                b_structure = b.get_gate_structure()
                structure['width'].append(b_structure['width'])
                structure['depth'].append(b_structure['depth'])
                # if len(b_structure) == 1:
                #     structure['width'] = structure['width'] + b_structure['width']

                # elif len(b_structure) == 2:
                #     structure['width'] = structure['width'] + b_structure['width']
                #     structure['depth'] = structure['depth'] + b_structure['depth']

            for b in self.attentions:
                assert hasattr(b, "get_gate_structure")
                b_structure = b.get_gate_structure()
                structure['width'].append(b_structure['width'])
                structure['depth'].append(b_structure['depth'])
                # if len(b_structure) == 1:
                #     structure['width'] = structure['width'] + b_structure['width']

                # elif len(b_structure) == 2:
                #     structure['width'] = structure['width'] + b_structure['width']
                #     structure['depth'] = structure['depth'] + b_structure['depth']

            self.structure = structure

        return self.structure

    def set_gate_structure(self, arch_vectors):
        # We first read the resnets and then attentions in the "get_gate_structure" method. Thus, we do a similar approach here.
        width_vectors, depth_vectors = arch_vectors['width'], arch_vectors['depth']

        for b in self.resnets:

            assert hasattr(b, "get_gate_structure")
            b_structure = b.get_gate_structure()
            assert len(b_structure) == 2
            block_vectors = {'width': [], 'depth': []}
            for i in range(len(b_structure['width'])):
                b_structure['width'][i] == width_vectors[0].shape[1]
                block_vectors['width'].append(width_vectors.pop(0))
            for i in range(len(b_structure['depth'])):
                if b_structure['depth'][i] == 1:
                    block_vectors['depth'].append(depth_vectors.pop(0))
            b.set_gate_structure(block_vectors)

        for b in self.attentions:

            assert hasattr(b, "get_gate_structure")
            b_structure = b.get_gate_structure()
            assert len(b_structure) == 2
            block_vectors = {'width': [], 'depth': []}
            for i in range(len(b_structure['width'])):
                assert b_structure['width'][i] == width_vectors[0].shape[1]
                block_vectors['width'].append(width_vectors.pop(0))
            for i in range(len(b_structure['depth'])):
                if b_structure['depth'][i] == 1:
                    block_vectors['depth'].append(depth_vectors.pop(0))
            b.set_gate_structure(block_vectors)


class CrossAttnUpBlock2DWidthDepthGated(CrossAttnUpBlock2D):
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


class CrossAttnUpBlock2DWidthHalfDepthGated(CrossAttnUpBlock2D):
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
            gated_ff: bool = False
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

        for i in range(num_layers - 1):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2DWidthGated(
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
                    is_input_concatenated=True
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModelWidthGated(
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
                        gated_ff=gated_ff
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModelWidthGated(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )

        for i in range(num_layers - 1, num_layers):
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
                    is_input_concatenated=True
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
                        gated_ff=gated_ff
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
        self.structure = {'width': [], 'depth': []}

    def get_gate_structure(self):
        if len(self.structure['width']) == 0:
            structure = {'width': [], 'depth': []}
            for b in self.resnets:
                assert hasattr(b, "get_gate_structure")
                b_structure = b.get_gate_structure()
                structure['width'].append(b_structure['width'])
                structure['depth'].append(b_structure['depth'])
                # if len(b_structure) == 1:
                #     structure['width'] = structure['width'] + b_structure['width']

                # elif len(b_structure) == 2:
                #     structure['width'] = structure['width'] + b_structure['width']
                #     structure['depth'] = structure['depth'] + b_structure['depth']

            for b in self.attentions:
                assert hasattr(b, "get_gate_structure")
                b_structure = b.get_gate_structure()
                structure['width'].append(b_structure['width'])
                structure['depth'].append(b_structure['depth'])
                # if len(b_structure) == 1:
                #     structure['width'] = structure['width'] + b_structure['width']

                # elif len(b_structure) == 2:
                #     structure['width'] = structure['width'] + b_structure['width']
                #     structure['depth'] = structure['depth'] + b_structure['depth']

            self.structure = structure

        return self.structure

    def set_gate_structure(self, arch_vectors):
        # We first read the resnets and then attentions in the "get_gate_structure" method. Thus, we do a similar approach here.
        width_vectors, depth_vectors = arch_vectors['width'], arch_vectors['depth']

        for b in self.resnets:

            assert hasattr(b, "get_gate_structure")
            b_structure = b.get_gate_structure()
            assert len(b_structure) == 2
            block_vectors = {'width': [], 'depth': []}
            for i in range(len(b_structure['width'])):
                b_structure['width'][i] == width_vectors[0].shape[1]
                block_vectors['width'].append(width_vectors.pop(0))
            for i in range(len(b_structure['depth'])):
                if b_structure['depth'][i] == 1:
                    block_vectors['depth'].append(depth_vectors.pop(0))
            b.set_gate_structure(block_vectors)

        for b in self.attentions:

            assert hasattr(b, "get_gate_structure")
            b_structure = b.get_gate_structure()
            assert len(b_structure) == 2
            block_vectors = {'width': [], 'depth': []}
            for i in range(len(b_structure['width'])):
                assert b_structure['width'][i] == width_vectors[0].shape[1]
                block_vectors['width'].append(width_vectors.pop(0))
            for i in range(len(b_structure['depth'])):
                if b_structure['depth'][i] == 1:
                    block_vectors['depth'].append(depth_vectors.pop(0))
            b.set_gate_structure(block_vectors)


class DownBlock2DWidthDepthGated(DownBlock2D):
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


class DownBlock2DWidthHalfDepthGated(DownBlock2D):
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

        for i in range(num_layers - 1):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2DWidthGated(
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
                    is_input_concatenated=False
                )
            )
        for i in range(num_layers - 1, num_layers):
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
                    is_input_concatenated=False
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.structure = {'width': [], 'depth': []}

    def get_gate_structure(self):
        if len(self.structure['width']) == 0:
            structure = {'width': [], 'depth': []}
            for b in self.resnets:
                assert hasattr(b, "get_gate_structure")
                b_structure = b.get_gate_structure()
                structure['width'].append(b_structure['width'])
                structure['depth'].append(b_structure['depth'])
                # if len(b_structure) == 1:
                #     structure['width'] = structure['width'] + b_structure['width']

                # elif len(b_structure) == 2:
                #     structure['width'] = structure['width'] + b_structure['width']
                #     structure['depth'] = structure['depth'] + b_structure['depth']

            self.structure = structure
        return self.structure

    def set_gate_structure(self, arch_vectors):

        width_vectors, depth_vectors = arch_vectors['width'], arch_vectors['depth']
        for b in self.resnets:
            assert hasattr(b, "get_gate_structure")
            b_structure = b.get_gate_structure()
            assert len(b_structure) == 2
            block_vectors = {'width': [], 'depth': []}
            for i in range(len(b_structure['width'])):
                b_structure['width'][i] == width_vectors[0].shape[1]
                block_vectors['width'].append(width_vectors.pop(0))
            for i in range(len(b_structure['depth'])):
                if b_structure['depth'][i] == 1:
                    block_vectors['depth'].append(depth_vectors.pop(0))
            b.set_gate_structure(block_vectors)


class UpBlock2DWidthDepthGated(UpBlock2D):
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


class UpBlock2DWidthHalfDepthGated(UpBlock2D):
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

        for i in range(num_layers - 1):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2DWidthGated(
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
                    is_input_concatenated=True
                )
            )

        for i in range(num_layers - 1, num_layers):
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
                    is_input_concatenated=True
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.structure = {'width': [], 'depth': []}

    def get_gate_structure(self):
        if len(self.structure['width']) == 0:
            structure = {'width': [], 'depth': []}
            for b in self.resnets:
                assert hasattr(b, "get_gate_structure")
                b_structure = b.get_gate_structure()
                structure['width'].append(b_structure['width'])
                structure['depth'].append(b_structure['depth'])
                # if len(b_structure) == 1:
                #     structure['width'] = structure['width'] + b_structure['width']

                # elif len(b_structure) == 2:
                #     structure['width'] = structure['width'] + b_structure['width']
                #     structure['depth'] = structure['depth'] + b_structure['depth']

            self.structure = structure
        return self.structure

    def set_gate_structure(self, arch_vectors):

        width_vectors, depth_vectors = arch_vectors['width'], arch_vectors['depth']
        for b in self.resnets:
            assert hasattr(b, "get_gate_structure")
            b_structure = b.get_gate_structure()
            assert len(b_structure) == 2
            block_vectors = {'width': [], 'depth': []}
            for i in range(len(b_structure['width'])):
                b_structure['width'][i] == width_vectors[0].shape[1]
                block_vectors['width'].append(width_vectors.pop(0))
            for i in range(len(b_structure['depth'])):
                if b_structure['depth'][i] == 1:
                    block_vectors['depth'].append(depth_vectors.pop(0))
            b.set_gate_structure(block_vectors)


class UNetMidBlock2DCrossAttnWidthGated(UNetMidBlock2DCrossAttn):
    def __init__(
            self,
            in_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            transformer_layers_per_block: Union[int, Tuple[int]] = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            num_attention_heads: int = 1,
            output_scale_factor: float = 1.0,
            cross_attention_dim: int = 1280,
            dual_cross_attention: bool = False,
            use_linear_projection: bool = False,
            upcast_attention: bool = False,
            attention_type: str = "default",
            gated_ff: bool = False
    ):
        super().__init__(in_channels=in_channels, temb_channels=temb_channels, dropout=dropout, num_layers=num_layers,
                         transformer_layers_per_block=transformer_layers_per_block, resnet_eps=resnet_eps,
                         resnet_time_scale_shift=resnet_time_scale_shift, resnet_act_fn=resnet_act_fn,
                         resnet_groups=resnet_groups, resnet_pre_norm=resnet_pre_norm,
                         num_attention_heads=num_attention_heads, output_scale_factor=output_scale_factor,
                         cross_attention_dim=cross_attention_dim, dual_cross_attention=dual_cross_attention,
                         use_linear_projection=use_linear_projection, upcast_attention=upcast_attention,
                         attention_type=attention_type)

        # support for variable transformer layers per block
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        resnets = [ResnetBlock2DWidthGated(
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            eps=resnet_eps,
            groups=resnet_groups,
            dropout=dropout,
            time_embedding_norm=resnet_time_scale_shift,
            non_linearity=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            pre_norm=resnet_pre_norm,
        )]
        attentions = []

        for i in range(num_layers):
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModelWidthGated(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                        gated_ff=gated_ff
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModelWidthGated(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
            resnets.append(
                ResnetBlock2DWidthGated(
                    in_channels=in_channels,
                    out_channels=in_channels,
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

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.structure = {'width': [], 'depth': []}

    def get_gate_structure(self):
        if len(self.structure['width']) == 0:
            structure = {'width': [], 'depth': []}
            for b in self.resnets:
                assert hasattr(b, "get_gate_structure")
                b_structure = b.get_gate_structure()
                structure['width'].append(b_structure['width'])
                structure['depth'].append(b_structure['depth'])
                # assert len(b_structure) == 1
                # structure['width'] = structure['width'] + b_structure['width']

            for b in self.attentions:
                assert hasattr(b, "get_gate_structure")
                b_structure = b.get_gate_structure()
                structure['width'].append(b_structure['width'])
                structure['depth'].append(b_structure['depth'])
                # assert len(b_structure) == 1
                # structure['width'] = structure['width'] + b_structure['width']

            self.structure = structure

        return self.structure

    def set_gate_structure(self, arch_vectors):
        # We first read the resnets and then attentions in the "get_gate_structure" method. Thus, we do a similar approach here.
        width_vectors, depth_vectors = arch_vectors['width'], arch_vectors['depth']

        for b in self.resnets:

            assert hasattr(b, "get_gate_structure")
            b_structure = b.get_gate_structure()
            assert len(b_structure) == 2
            block_vectors = {'width': [], 'depth': []}
            for i in range(len(b_structure['width'])):
                b_structure['width'][i] == width_vectors[0].shape[1]
                block_vectors['width'].append(width_vectors.pop(0))
            for i in range(len(b_structure['depth'])):
                if b_structure['depth'][i] == 1:
                    block_vectors['depth'].append(depth_vectors.pop(0))
            b.set_gate_structure(block_vectors)

        for b in self.attentions:

            assert hasattr(b, "get_gate_structure")
            b_structure = b.get_gate_structure()
            assert len(b_structure) == 2
            block_vectors = {'width': [], 'depth': []}
            for i in range(len(b_structure['width'])):
                assert b_structure['width'][i] == width_vectors[0].shape[1]
                block_vectors['width'].append(width_vectors.pop(0))
            for i in range(len(b_structure['depth'])):
                if b_structure['depth'][i] == 1:
                    block_vectors['depth'].append(depth_vectors.pop(0))
            b.set_gate_structure(block_vectors)
