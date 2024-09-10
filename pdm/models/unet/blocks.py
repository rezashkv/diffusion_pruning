from typing import Optional, Dict, Any, Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from diffusers.configuration_utils import register_to_config
from diffusers.models import DualTransformer2DModel, Transformer2DModel
from diffusers.models.activations import GEGLU
from diffusers.models.resnet import ResnetBlock2D, Upsample2D, Downsample2D
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.models.attention import BasicTransformerBlock, FeedForward
from diffusers.models.unets.unet_2d_blocks import (CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UpBlock2D,
                                                   UNetMidBlock2DCrossAttn)
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.utils import logging, deprecate, is_torch_npu_available, is_torch_version

from ..gates import DepthGate, WidthGate, LinearWidthGate
from ...utils.estimation_utils import hard_concrete

if is_torch_npu_available():
    import torch_npu

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class GEGLUGated(GEGLU):
    r"""
    A pruning-gated [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        gate_width (`int`, *optional*, defaults to 32): The width of the pruning gate.
    """

    def __init__(self, dim_in: int, dim_out: int, gate_width: int = 32):
        super().__init__(dim_in, dim_out)
        self.dim_out = dim_out
        self.gate = LinearWidthGate(gate_width)
        self.total_macs, self.prunable_macs = 0., 0.
        self.pruned = False

    def forward(self, hidden_states, *args, **kwargs):
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        hidden_states = self.proj(hidden_states)
        if is_torch_npu_available():
            if not self.pruned:
                hidden_states = self.gate(hidden_states)
            # using torch_npu.npu_geglu can run faster and save memory on NPU.
            return torch_npu.npu_geglu(hidden_states, dim=-1, approximate=1)[0]
        else:
            hidden_states, gate = hidden_states.chunk(2, dim=-1)
            if not self.pruned:
                hidden_states = self.gate(hidden_states)
                gate = self.gate(gate)
            return hidden_states * self.gelu(gate)

    @torch.no_grad()
    def prune_gate(self):
        assert self.gate.gate_f.shape[0] == 1, "Pruning is only supported for single batch size"
        gate_hard = hard_concrete(self.gate.gate_f.repeat_interleave(self.dim_out // self.gate.gate_f.shape[1], dim=1))
        gate_hard = torch.cat([gate_hard, gate_hard], dim=1)[0]

        # remove elements with zero gate from self.proj
        linear_cls = self.proj.__class__
        proj = linear_cls(self.proj.in_features, gate_hard.sum().int().item(), bias=self.proj.bias is not None)
        proj.weight.data = self.proj.weight.data[gate_hard.bool(), :]
        if self.proj.bias is not None:
            proj.bias.data = self.proj.bias.data[gate_hard.bool()]
        self.proj = proj
        self.pruned = True

        return gate_hard[:self.dim_out]


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
         gate_width (`int`, *optional*, defaults to 32): The width of the pruning gate.
     """

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            mult: int = 4,
            dropout: float = 0.0,
            activation_fn: str = "geglu",
            final_dropout: bool = False,
            gate_width: int = 32,
    ):
        super().__init__(dim, dim_out, mult, dropout, activation_fn, final_dropout)
        inner_dim = int(dim * mult)

        assert activation_fn == "geglu", f"Only GEGLU is supported in {self.__class__.__name__}"
        act_fn = GEGLUGated(dim, inner_dim, gate_width=gate_width)
        self.net[0] = act_fn
        self.structure = {'width': [], 'depth': []}
        self.prunable_macs, self.total_macs = 0., 0.

    def calc_macs(self):
        if self.total_macs == 0. or self.prunable_macs == 0.:
            self.total_macs, self.prunable_macs = 0., 0.
            # GEGLU
            self.total_macs += self.net[0].proj.__macs__
            self.prunable_macs += self.net[0].proj.__macs__

            # Linear
            self.total_macs += self.net[2].__macs__
            self.prunable_macs += self.net[2].__macs__

        hard_width_gate = hard_concrete(self.net[0].gate.gate_f)
        ratio = hard_width_gate.sum(dim=1, keepdim=True) / hard_width_gate.shape[1]
        return {"prunable_macs": self.prunable_macs,
                "total_macs": self.total_macs,
                "cur_prunable_macs": ratio * self.prunable_macs,
                "cur_total_macs": ratio.detach() * self.prunable_macs + (self.total_macs - self.prunable_macs)}

    @torch.no_grad()
    def prune(self):
        gate_hard = self.net[0].prune_gate()
        linear_cls = self.net[2].__class__
        linear = linear_cls(self.net[0].proj.out_features, self.net[2].out_features, bias=self.net[2].bias is not None)
        linear.weight.data = self.net[2].weight.data[:, gate_hard.bool()]
        if self.net[2].bias is not None:
            linear.bias.data = self.net[2].bias.data
        self.net[2] = linear


class GatedAttention(Attention):
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gate = WidthGate(self.heads)
        self.set_processor(HeadGatedAttnProcessor2())
        self.prunable_macs, self.total_macs = 0., 0.
        self.pruned = False

    def calc_macs(self):
        assert ((self.total_macs != 0.) and (self.prunable_macs != 0.))
        hard_width_gate = hard_concrete(self.gate.gate_f)
        ratio = hard_width_gate.sum(dim=1, keepdim=True) / hard_width_gate.shape[1]
        return {"prunable_macs": self.prunable_macs,
                "total_macs": self.total_macs,
                "cur_prunable_macs": ratio * self.prunable_macs,
                "cur_total_macs": ratio.detach() * self.prunable_macs + (self.total_macs - self.prunable_macs)}

    @torch.no_grad()
    def prune(self):
        def prune_linear(layer, gate_hard, out=False):
            num_new_heads = gate_hard.sum().int().item()
            assert num_new_heads > 0
            linear_cls = layer.__class__
            if out:
                head_dim = layer.in_features // self.heads
                linear = linear_cls(num_new_heads * head_dim, layer.out_features, bias=layer.bias is not None)
                orig_linear_weight = layer.weight.data.view(layer.out_features, self.heads, head_dim)
                new_linear_weight = orig_linear_weight[:, gate_hard.bool(), :].view(layer.out_features, -1)
            else:
                head_dim = layer.out_features // self.heads
                linear = linear_cls(layer.in_features, num_new_heads * head_dim, bias=layer.bias is not None)
                orig_linear_weight = layer.weight.data.view(self.heads, head_dim, layer.in_features)
                new_linear_weight = orig_linear_weight[gate_hard.bool(), :, :].view(-1, layer.in_features)

            linear.weight.data = new_linear_weight
            if layer.bias is not None:
                if out:
                    linear.bias.data = layer.bias.data
                else:
                    orig_linear_bias = layer.bias.data.view(self.heads, head_dim)
                    new_linear_bias = orig_linear_bias[gate_hard.bool(), :].view(-1)
                    linear.bias.data = new_linear_bias
            return linear

        assert self.gate.gate_f.shape[0] == 1, "Pruning is only supported for single batch size"
        gate_h = hard_concrete(self.gate.gate_f)[0]
        self.to_q = prune_linear(self.to_q, gate_h)
        self.to_k = prune_linear(self.to_k, gate_h)
        self.to_v = prune_linear(self.to_v, gate_h)
        self.to_out[0] = prune_linear(self.to_out[0], gate_h, out=True)
        self.heads = gate_h.sum().int().item()
        self.pruned = True


class HeadGatedAttnProcessor2(AttnProcessor2_0):
    def __init__(self):
        super().__init__()

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            *args,
            **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = ("The `scale` argument is deprecated and will be ignored. Please remove it,"
                                   " as passing it will raise an error in the future."
                                   " `scale` should directly be passed while calling the underlying pipeline component"
                                   " i.e., via `cross_attention_kwargs`.")
            deprecate("scale", "1.0.0", deprecation_message)

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

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if not attn.pruned:
            # ########## Apply Width Gate ##########
            assert key.shape[1] == attn.gate.gate_f.shape[1]
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
        hidden_states = attn.to_out[0](hidden_states)
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
        self.gate = WidthGate(self.norm2.num_groups)
        self.is_input_concatenated = is_input_concatenated
        self.structure = {'width': [], 'depth': []}
        self.prunable_macs, self.total_macs = 0., 0.
        self.pruned = False

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = ("The `scale` argument is deprecated and will be ignored."
                                   " Please remove it, as passing it will raise an error in the future."
                                   " `scale` should directly be passed while calling the underlying pipeline component"
                                   " i.e., via `cross_attention_kwargs`.")
            deprecate("scale", "1.0.0", deprecation_message)

        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb

            if not self.pruned:
                # ########## Apply Width Gate ##########
                assert self.norm2.num_groups == self.gate.gate_f.shape[1]
                hidden_states = self.gate(hidden_states)

            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)

            if not self.pruned:
                # ########## Apply Width Gate ##########
                assert self.norm2.num_groups == self.gate.gate_f.shape[1]
                hidden_states = self.gate(hidden_states)

            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            if not self.pruned:
                # ########## Apply Width Gate ##########
                assert self.norm2.num_groups == self.gate.gate_f.shape[1]
                hidden_states = self.gate(hidden_states)

            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor

    def get_gate_structure(self):
        if not self.structure["width"]:
            self.structure = {"width": [self.gate.width], "depth": [0]}
        return self.structure

    def set_gate_structure(self, arch_vectors):
        assert len(arch_vectors['depth']) == 0
        assert len(arch_vectors['width']) == 1
        assert arch_vectors['width'][0].shape[1] == self.gate.width
        self.gate.set_structure_value(arch_vectors['width'][0])

    def calc_macs(self):
        if self.total_macs == 0. or self.prunable_macs == 0.:
            self.total_macs, self.prunable_macs = 0., 0.
            # First GroupNorm
            self.total_macs += self.norm1.__macs__

            # Conv1
            self.total_macs += self.conv1.__macs__
            self.prunable_macs += self.conv1.__macs__

            # Time Embedding
            if self.time_emb_proj is not None:  # not necessary as it is always not None in the SD model
                self.total_macs += self.time_emb_proj.__macs__
                self.prunable_macs += self.time_emb_proj.__macs__

            # 2nd GroupNorm
            self.total_macs += self.norm2.__macs__
            self.prunable_macs += self.norm2.__macs__

            # Conv2
            self.total_macs += self.conv2.__macs__
            self.prunable_macs += self.conv2.__macs__

            # Skip Connection
            if self.conv_shortcut is not None:
                self.total_macs += self.conv_shortcut.__macs__

        hard_width_gate = hard_concrete(self.gate.gate_f)
        ratio = hard_width_gate.sum(dim=1, keepdim=True) / hard_width_gate.shape[1]
        return {"prunable_macs": self.prunable_macs,
                "total_macs": self.total_macs,
                "cur_prunable_macs": ratio * self.prunable_macs,
                "cur_total_macs": (ratio.detach()) * self.prunable_macs + (self.total_macs - self.prunable_macs)}

    def get_prunable_macs(self):
        return [self.prunable_macs]

    def get_block_utilization(self):
        return hard_concrete(self.gate.gate_f).mean(dim=1)

    @torch.no_grad()
    def prune(self):
        assert self.gate.gate_f.shape[0] == 1, "Pruning is only supported for single batch size"

        gate_hard = hard_concrete(self.gate.gate_f)[0]
        num_new_groups = gate_hard.sum().int().item()
        group_dim = self.conv1.out_channels // self.gate.width
        gate_hard = gate_hard.repeat_interleave(group_dim)

        conv1_cls = self.conv1.__class__
        conv1 = conv1_cls(self.conv1.in_channels, num_new_groups * group_dim, kernel_size=self.conv1.kernel_size,
                          stride=self.conv1.stride, padding=self.conv1.padding)
        conv1.weight.data = self.conv1.weight.data[gate_hard.bool(), :, :, :]
        if self.conv1.bias is not None:
            conv1.bias.data = self.conv1.bias.data[gate_hard.bool()]

        self.conv1 = conv1

        if self.time_emb_proj is not None:
            linear_cls = self.time_emb_proj.__class__
            time_emb_proj = linear_cls(self.time_emb_proj.in_features, conv1.out_channels,
                                       bias=self.time_emb_proj.bias is not None)
            time_emb_proj.weight.data = self.time_emb_proj.weight.data[gate_hard.bool(), :]
            if self.time_emb_proj.bias is not None:
                time_emb_proj.bias.data = self.time_emb_proj.bias.data[gate_hard.bool()]
            self.time_emb_proj = time_emb_proj

        norm2_cls = self.norm2.__class__
        norm2 = norm2_cls(num_new_groups, conv1.out_channels, self.norm2.eps, self.norm2.affine)
        norm2.weight.data = self.norm2.weight.data[gate_hard.bool()]
        norm2.bias.data = self.norm2.bias.data[gate_hard.bool()]
        self.norm2 = norm2

        conv2_cls = self.conv2.__class__
        conv2 = conv2_cls(norm2.num_channels, self.conv2.out_channels, kernel_size=self.conv2.kernel_size,
                          stride=self.conv2.stride, padding=self.conv2.padding)
        conv2.weight.data = self.conv2.weight.data[:, gate_hard.bool(), :, :]
        if self.conv2.bias is not None:
            conv2.bias.data = self.conv2.bias.data
        self.conv2 = conv2

        self.pruned = True


class ResnetBlock2DWidthDepthGated(ResnetBlock2D):
    def __init__(self, skip_connection_dim=None, is_input_concatenated=False, *args, **kwargs):
        # extract gate_flag from kwargs
        super().__init__(*args, **kwargs)

        self.gate = WidthGate(self.norm2.num_groups)
        self.depth_gate = DepthGate(1)
        self.is_input_concatenated = is_input_concatenated
        self.skip_connection_dim = skip_connection_dim
        self.structure = {'width': [], 'depth': []}
        self.prunable_macs, self.total_macs = 0., 0.
        self.dropped = False
        self.pruned = False

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = ("The `scale` argument is deprecated and will be ignored."
                                   " Please remove it, as passing it will raise an error in the future."
                                   " `scale` should directly be passed while calling the underlying pipeline component"
                                   " i.e., via `cross_attention_kwargs`.")
            deprecate("scale", "1.0.0", deprecation_message)

        assert (self.upsample is None) and (self.downsample is None)
        # Depth gate cannot be in the up/down sample blocks.
        if self.is_input_concatenated:  # We are in the upsample blocks, input is concatenated.
            # input_hidden_states = input_tensor.chunk(2, dim=1)[0]
            # [0] because the forward pass is hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            # in here:
            # https://github.com/huggingface/diffusers/blob/acd926f4f208e4cf12be69315787c450da48913b/src/diffusers/models/unet_2d_blocks.py#L2324
            assert input_tensor.ndim == 4
            assert self.skip_connection_dim is not None
            n_channels_concat = input_tensor.shape[1]
            input_hidden_states = input_tensor[:, :(n_channels_concat - self.skip_connection_dim), :, :]
        else:  # We are in the downsample blocks
            input_hidden_states = input_tensor

        if self.dropped:
            return input_hidden_states

        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb

            if not self.pruned:
                # ########## Apply Width Gate ##########
                assert self.norm2.num_groups == self.gate.gate_f.shape[1]
                hidden_states = self.gate(hidden_states)

            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)

            if not self.pruned:
                # ########## Apply Width Gate ##########
                assert self.norm2.num_groups == self.gate.gate_f.shape[1]
                hidden_states = self.gate(hidden_states)

            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            if not self.pruned:
                # ########## Apply Width Gate ##########
                assert self.norm2.num_groups == self.gate.gate_f.shape[1]
                hidden_states = self.gate(hidden_states)

            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        if not self.pruned:
            # ########## Apply Depth Gate ##########
            assert output_tensor.shape == input_hidden_states.shape
            output = self.depth_gate((input_hidden_states, output_tensor))

            return output
        else:
            return output_tensor

    def get_gate_structure(self):
        if not self.structure["width"]:
            self.structure = {"depth": [self.depth_gate.width], "width": [self.gate.width]}
        return self.structure

    def set_gate_structure(self, arch_vectors):
        assert len(arch_vectors['depth']) == 1
        assert len(arch_vectors['width']) == 1
        assert arch_vectors['width'][0].shape[1] == self.gate.width
        self.gate.set_structure_value(arch_vectors['width'][0])
        self.depth_gate.set_structure_value(arch_vectors['depth'][0])

    def calc_macs(self):
        if self.total_macs == 0. or self.prunable_macs == 0.:
            self.total_macs, self.prunable_macs = 0., 0.
            # First GroupNorm
            self.total_macs += self.norm1.__macs__

            # Conv1
            self.total_macs += self.conv1.__macs__
            self.prunable_macs += self.conv1.__macs__

            # Time Embedding
            self.total_macs += self.time_emb_proj.__macs__
            self.prunable_macs += self.time_emb_proj.__macs__

            # 2nd GroupNorm
            self.total_macs += self.norm2.__macs__
            self.prunable_macs += self.norm2.__macs__

            # Conv2
            self.total_macs += self.conv2.__macs__
            self.prunable_macs += self.conv2.__macs__

            # Skip Connection
            if self.conv_shortcut is not None:
                self.total_macs += self.conv_shortcut.__macs__

        hard_width_gate = hard_concrete(self.gate.gate_f)
        ratio = hard_width_gate.sum(dim=1, keepdim=True) / hard_width_gate.shape[1]
        depth_hard_gate = hard_concrete(self.depth_gate.gate_f).unsqueeze(1)
        depth_ratio = depth_hard_gate.sum(dim=1, keepdim=True) / depth_hard_gate.shape[1]
        return {"prunable_macs": self.prunable_macs,
                "total_macs": self.total_macs,
                "cur_prunable_macs": ((ratio * self.prunable_macs) + (
                        self.total_macs - self.prunable_macs)) * depth_ratio,
                "cur_total_macs": ((ratio.detach()) * self.prunable_macs + (
                        self.total_macs - self.prunable_macs)) * (depth_ratio.detach())}

    def get_prunable_macs(self):
        return [self.prunable_macs]

    def get_block_utilization(self):
        return hard_concrete(self.gate.gate_f).mean(dim=1) * hard_concrete(self.depth_gate.gate_f)

    @torch.no_grad()
    def prune(self):
        assert self.depth_gate.gate_f.shape[0] == self.gate.gate_f.shape[
            0] == 1, "Pruning is only supported for single batch size"
        hard_depth_gate = hard_concrete(self.depth_gate.gate_f)[0]
        if hard_depth_gate == 0:
            self.dropped = True
            # set all modules to identity
            self.norm1 = nn.Identity()
            self.conv1 = nn.Identity()
            if self.time_emb_proj is not None:
                self.time_emb_proj = nn.Identity()
            self.norm2 = nn.Identity()
            self.conv2 = nn.Identity()
            self.nonlinearity = nn.Identity()
            self.dropout = nn.Identity()
            if self.conv_shortcut is not None:
                self.conv_shortcut = nn.Identity()

        else:
            gate_hard = hard_concrete(self.gate.gate_f)[0]
            num_new_groups = gate_hard.sum().int().item()
            group_dim = self.conv1.out_channels // self.gate.width
            gate_hard = gate_hard.repeat_interleave(group_dim)

            conv1_cls = self.conv1.__class__
            conv1 = conv1_cls(self.conv1.in_channels, num_new_groups * group_dim, kernel_size=self.conv1.kernel_size,
                              stride=self.conv1.stride, padding=self.conv1.padding)
            conv1.weight.data = self.conv1.weight.data[gate_hard.bool(), :, :, :]
            if self.conv1.bias is not None:
                conv1.bias.data = self.conv1.bias.data[gate_hard.bool()]

            self.conv1 = conv1

            if self.time_emb_proj is not None:
                linear_cls = self.time_emb_proj.__class__
                time_emb_proj = linear_cls(self.time_emb_proj.in_features, conv1.out_channels,
                                           bias=self.time_emb_proj.bias is not None)
                time_emb_proj.weight.data = self.time_emb_proj.weight.data[gate_hard.bool(), :]
                if self.time_emb_proj.bias is not None:
                    time_emb_proj.bias.data = self.time_emb_proj.bias.data[gate_hard.bool()]
                self.time_emb_proj = time_emb_proj

            norm2_cls = self.norm2.__class__
            norm2 = norm2_cls(num_new_groups, conv1.out_channels, self.norm2.eps, self.norm2.affine)
            norm2.weight.data = self.norm2.weight.data[gate_hard.bool()]
            norm2.bias.data = self.norm2.bias.data[gate_hard.bool()]
            self.norm2 = norm2

            conv2_cls = self.conv2.__class__
            conv2 = conv2_cls(norm2.num_channels, self.conv2.out_channels, kernel_size=self.conv2.kernel_size,
                              stride=self.conv2.stride, padding=self.conv2.padding)
            conv2.weight.data = self.conv2.weight.data[:, gate_hard.bool(), :, :]
            if self.conv2.bias is not None:
                conv2.bias.data = self.conv2.bias.data
            self.conv2 = conv2
            self.pruned = True


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
            # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
            norm_eps: float = 1e-5,
            final_dropout: bool = False,
            attention_type: str = "default",
            positional_embeddings: Optional[str] = None,
            num_positional_embeddings: Optional[int] = None,
            ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
            ada_norm_bias: Optional[int] = None,
            ff_inner_dim: Optional[int] = None,
            ff_bias: bool = True,
            attention_out_bias: bool = True,
            gated_ff: bool = True,
            ff_gate_width: int = 32,
    ):
        super().__init__(dim=dim, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim,
                         dropout=dropout, cross_attention_dim=cross_attention_dim, activation_fn=activation_fn,
                         num_embeds_ada_norm=num_embeds_ada_norm, attention_bias=attention_bias,
                         only_cross_attention=only_cross_attention, double_self_attention=double_self_attention,
                         upcast_attention=upcast_attention, norm_elementwise_affine=norm_elementwise_affine,
                         norm_type=norm_type, norm_eps=norm_eps, final_dropout=final_dropout,
                         attention_type=attention_type, positional_embeddings=positional_embeddings,
                         num_positional_embeddings=num_positional_embeddings,
                         ada_norm_continous_conditioning_embedding_dim=ada_norm_continous_conditioning_embedding_dim,
                         ada_norm_bias=ada_norm_bias, ff_inner_dim=ff_inner_dim, ff_bias=ff_bias,
                         attention_out_bias=attention_out_bias)

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        if self.attn1 is not None:
            self.attn1 = GatedAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
            )

        if self.attn2 is not None:
            self.attn2 = GatedAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )

        self.gated_ff = gated_ff
        if gated_ff:
            self.ff = FeedForwardWidthGated(dim, dropout=dropout, activation_fn=activation_fn,
                                            final_dropout=final_dropout, gate_width=ff_gate_width)
        else:
            self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        self.structure = {'width': [], 'depth': []}
        self.prunable_macs, self.total_macs = 0., 0.

    def get_gate_structure(self):
        if len(self.structure['width']) == 0:
            self.structure['width'] = [self.attn1.gate.width, self.attn2.gate.width]
            if isinstance(self.ff, FeedForwardWidthGated):
                self.structure['width'].append(self.ff.net[0].gate.width)
            self.structure['depth'] = [0]
        return self.structure

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
            assert arch_vectors['width'][2].shape[1] == self.ff.net[0].gate.width
            self.ff.net[0].gate.set_structure_value(arch_vectors['width'][2])

    def calc_macs(self):
        out_dict = {"prunable_macs": 0., "total_macs": 0., "cur_prunable_macs": 0., "cur_total_macs": 0.}

        # Norm1
        out_dict["total_macs"] += self.norm1.__macs__
        out_dict["cur_total_macs"] += self.norm1.__macs__

        # Attention1
        attn1_macs = self.attn1.calc_macs()
        for k in out_dict.keys():
            out_dict[k] = out_dict[k] + attn1_macs[k]

        # Norm2
        out_dict["total_macs"] += self.norm2.__macs__
        out_dict["cur_total_macs"] += self.norm2.__macs__

        # Attention2
        if self.attn2 is not None:
            attn2_macs = self.attn2.calc_macs()
            for k in out_dict.keys():
                out_dict[k] = out_dict[k] + attn2_macs[k]

        # Norm3
        out_dict["total_macs"] += self.norm3.__macs__
        out_dict["cur_total_macs"] += self.norm3.__macs__

        # FeedForward
        if self.gated_ff:
            ff_macs = self.ff.calc_macs()
            for k in out_dict.keys():
                out_dict[k] = out_dict[k] + ff_macs[k]

        if self.total_macs == 0.:
            self.total_macs = out_dict["total_macs"]

        if self.prunable_macs == 0.:
            self.prunable_macs = out_dict["prunable_macs"]

        return out_dict

    def get_prunable_macs(self):
        macs = [self.attn1.prunable_macs, self.attn2.prunable_macs]
        if self.gated_ff:
            macs.append(self.ff.prunable_macs)
        return macs

    def get_block_utilization(self):
        attn1_ratio = hard_concrete(self.attn1.gate.gate_f).mean(dim=1)
        attn2_ratio = hard_concrete(self.attn2.gate.gate_f).mean(dim=1)
        total_prunable_macs = self.attn1.prunable_macs + self.attn2.prunable_macs + (self.ff.prunable_macs if
                                                                                     self.gated_ff else 0)
        if self.gated_ff:
            ff_ratio = hard_concrete(self.ff.net[0].gate.gate_f).mean(dim=1)
            return (
                    attn1_ratio * self.attn1.prunable_macs + attn2_ratio * self.attn2.prunable_macs +
                    ff_ratio * self.ff.prunable_macs) / total_prunable_macs
        else:
            return ((
                            attn1_ratio * self.attn1.prunable_macs + attn2_ratio * self.attn2.prunable_macs) /
                    total_prunable_macs)


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
            # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
            attention_type: str = "default",
            caption_channels: int = None,
            interpolation_scale: float = None,
            use_additional_conditions: Optional[bool] = None,
            gated_ff: bool = False,
            ff_gate_width: int = 32
    ):
        super().__init__(num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim,
                         in_channels=in_channels, out_channels=out_channels, num_layers=num_layers, dropout=dropout,
                         norm_num_groups=norm_num_groups, cross_attention_dim=cross_attention_dim,
                         attention_bias=attention_bias, sample_size=sample_size, num_vector_embeds=num_vector_embeds,
                         patch_size=patch_size, activation_fn=activation_fn, num_embeds_ada_norm=num_embeds_ada_norm,
                         use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention,
                         double_self_attention=double_self_attention, upcast_attention=upcast_attention,
                         norm_type=norm_type, norm_elementwise_affine=norm_elementwise_affine,
                         attention_type=attention_type, norm_eps=norm_eps,
                         caption_channels=caption_channels, interpolation_scale=interpolation_scale,
                         use_additional_conditions=use_additional_conditions)

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
                    gated_ff=gated_ff,
                    ff_gate_width=ff_gate_width
                )
                for _ in range(num_layers)
            ]
        )
        self.structure = {'width': [], 'depth': []}
        self.prunable_macs, self.total_macs = 0., 0.

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

    def calc_macs(self):
        out_dict = {"prunable_macs": 0., "total_macs": 0., "cur_prunable_macs": 0., "cur_total_macs": 0.}

        # Input
        if self.is_input_continuous:
            # Norm
            out_dict["total_macs"] += self.norm.__macs__
            out_dict["cur_total_macs"] += self.norm.__macs__

            # proj_in (conv or linear)
            out_dict["total_macs"] += self.proj_in.__macs__
            out_dict["cur_total_macs"] += self.proj_in.__macs__

        # Transformer blocks
        for tb in self.transformer_blocks:
            tb_macs = tb.calc_macs()
            for k in out_dict.keys():
                out_dict[k] = out_dict[k] + tb_macs[k]

        # Output
        if self.is_input_continuous:
            # proj_out (conv or linear)
            out_dict["total_macs"] += self.proj_out.__macs__
            out_dict["cur_total_macs"] += self.proj_out.__macs__

        if self.total_macs == 0.:
            self.total_macs = out_dict["total_macs"]

        if self.prunable_macs == 0:
            self.prunable_macs = out_dict["prunable_macs"]

        return out_dict

    def get_prunable_macs(self):
        macs = []
        for tb in self.transformer_blocks:
            macs += tb.get_prunable_macs()
        return macs

    def get_block_utilization(self):
        util = []
        for tb in self.transformer_blocks:
            util.append(tb.get_block_utilization())
        return torch.stack(util).mean(dim=0)


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
            # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
            attention_type: str = "default",
            caption_channels: int = None,
            interpolation_scale: float = None,
            use_additional_conditions: Optional[bool] = None,
            gated_ff: bool = False,
            ff_gate_width: int = 32
    ):
        super().__init__(num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim,
                         in_channels=in_channels, out_channels=out_channels, num_layers=num_layers, dropout=dropout,
                         norm_num_groups=norm_num_groups, cross_attention_dim=cross_attention_dim,
                         attention_bias=attention_bias, sample_size=sample_size, num_vector_embeds=num_vector_embeds,
                         patch_size=patch_size, activation_fn=activation_fn, num_embeds_ada_norm=num_embeds_ada_norm,
                         use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention,
                         double_self_attention=double_self_attention, upcast_attention=upcast_attention,
                         norm_type=norm_type, norm_elementwise_affine=norm_elementwise_affine,
                         attention_type=attention_type, norm_eps=norm_eps,
                         caption_channels=caption_channels, interpolation_scale=interpolation_scale,
                         use_additional_conditions=use_additional_conditions)

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
                    gated_ff=gated_ff,
                    ff_gate_width=ff_gate_width
                )
                for _ in range(num_layers)
            ]
        )

        self.depth_gate = DepthGate(1)
        self.structure = {'width': [], 'depth': []}
        self.prunable_macs, self.total_macs = 0., 0.
        self.dropped, self.pruned = False, False

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
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)`
            if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
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
                Whether to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        if self.dropped:
            if not return_dict:
                return (hidden_states,)

            return Transformer2DModelOutput(sample=hidden_states)

        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

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

        # 1. Input
        input_hidden_states = hidden_states
        if self.is_input_continuous:
            batch_size, _, height, width = hidden_states.shape
            residual = hidden_states
            hidden_states, inner_dim = self._operate_on_continuous_inputs(hidden_states)
        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[
                -1] // self.patch_size
            hidden_states, encoder_hidden_states, timestep, embedded_timestep = self._operate_on_patched_inputs(
                hidden_states, encoder_hidden_states, timestep, added_cond_kwargs
            )

        # 2. Blocks
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=",
                                                                                           "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
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

        # 3. Output
        if self.is_input_continuous:
            output = self._get_output_for_continuous_inputs(
                hidden_states=hidden_states,
                residual=residual,
                batch_size=batch_size,
                height=height,
                width=width,
                inner_dim=inner_dim,
            )
        elif self.is_input_vectorized:
            output = self._get_output_for_vectorized_inputs(hidden_states)
        elif self.is_input_patches:
            output = self._get_output_for_patched_inputs(
                hidden_states=hidden_states,
                timestep=timestep,
                class_labels=class_labels,
                embedded_timestep=embedded_timestep,
                height=height,
                width=width,
            )

        if not self.pruned:
            # ########## Apply Depth Gate ##########
            assert output.shape == input_hidden_states.shape
            output_tensor = self.depth_gate((input_hidden_states, output))

        else:
            output_tensor = output
        if not return_dict:
            return (output_tensor,)

        return Transformer2DModelOutput(sample=output_tensor)

    def get_gate_structure(self):
        if len(self.structure['width']) == 0:
            for tb in self.transformer_blocks:
                tb_structure = tb.get_gate_structure()
                self.structure['width'] = self.structure['width'] + tb_structure['width']
            self.structure['depth'].append(1)
        return self.structure

    def set_gate_structure(self, arch_vectors):
        if len(self.transformer_blocks) > 1:
            raise NotImplementedError

        assert len(arch_vectors['depth']) == 1
        self.depth_gate.set_structure_value(arch_vectors['depth'][0])
        self.transformer_blocks[0].set_gate_structure({'width': arch_vectors['width'], 'depth': []})

    def calc_macs(self):
        out_dict = {"prunable_macs": 0., "total_macs": 0., "cur_prunable_macs": 0., "cur_total_macs": 0.}

        # Input
        if self.is_input_continuous:
            # Norm
            out_dict["total_macs"] += self.norm.__macs__
            out_dict["cur_total_macs"] += self.norm.__macs__

            # proj_in (conv or linear)
            out_dict["total_macs"] += self.proj_in.__macs__
            out_dict["cur_total_macs"] += self.proj_in.__macs__

        # Transformer blocks
        for tb in self.transformer_blocks:
            tb_macs = tb.calc_macs()
            for k in out_dict.keys():
                out_dict[k] = out_dict[k] + tb_macs[k]

        # Output
        if self.is_input_continuous:
            # proj_out (conv or linear)
            out_dict["total_macs"] += self.proj_out.__macs__
            out_dict["cur_total_macs"] += self.proj_out.__macs__

        # return out_dict

        depth_hard_gate = hard_concrete(self.depth_gate.gate_f).unsqueeze(1)
        depth_ratio = depth_hard_gate.sum(dim=1, keepdim=True) / depth_hard_gate.shape[1]

        if self.total_macs == 0.:
            self.total_macs = out_dict["total_macs"]

        if self.prunable_macs == 0:
            self.prunable_macs = out_dict["prunable_macs"]

        out_dict["cur_prunable_macs"] = ((out_dict["cur_prunable_macs"] + self.total_macs - self.prunable_macs)
                                         * depth_ratio)
        out_dict["cur_total_macs"] = out_dict["cur_total_macs"] * (depth_ratio.detach())

        return out_dict

    def get_prunable_macs(self):
        macs = []
        for tb in self.transformer_blocks:
            macs += tb.get_prunable_macs()
        return macs

    def get_block_utilization(self):
        util = []
        for tb in self.transformer_blocks:
            util.append(tb.get_block_utilization())
        return torch.stack(util).mean(dim=0) * hard_concrete(self.depth_gate.gate_f)

    @torch.no_grad()
    def prune_module(self):
        assert self.depth_gate.gate_f.shape[0] == 1, "Pruning is only supported for single batch size"
        hard_depth_gate = hard_concrete(self.depth_gate.gate_f)[0]
        self.pruned = True
        if hard_depth_gate == 0:
            self.norm = nn.Identity()
            self.proj_in = nn.Identity()
            transformer_blocks = [nn.Identity() for _ in range(len(self.transformer_blocks))]
            self.transformer_blocks = nn.ModuleList(transformer_blocks)
            self.proj_out = nn.Identity()
            self.dropped = True


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
                Whether to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

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
            gated_ff: bool = True,
            ff_gate_width: int = 32
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
                        gated_ff=gated_ff,
                        ff_gate_width=ff_gate_width
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
            gated_ff: bool = False,
            ff_gate_width: int = 32
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
                        gated_ff=gated_ff,
                        ff_gate_width=ff_gate_width
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
                        gated_ff=gated_ff,
                        ff_gate_width=ff_gate_width
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
        self.total_macs, self.prunable_macs = 0., 0.

    def get_gate_structure(self):
        if len(self.structure['width']) == 0:
            structure = {'width': [], 'depth': []}
            for b in self.resnets:
                assert hasattr(b, "get_gate_structure")
                b_structure = b.get_gate_structure()
                structure['width'].append(b_structure['width'])
                structure['depth'].append(b_structure['depth'])

            for b in self.attentions:
                assert hasattr(b, "get_gate_structure")
                b_structure = b.get_gate_structure()
                structure['width'].append(b_structure['width'])
                structure['depth'].append(b_structure['depth'])

            self.structure = structure

        return self.structure

    def set_gate_structure(self, arch_vectors):
        # We first read the resnets and then attentions in the "get_gate_structure" method.
        # Thus, we do a similar approach here.
        width_vectors, depth_vectors = arch_vectors['width'], arch_vectors['depth']

        for b in self.resnets:
            b_structure = b.get_gate_structure()
            block_vectors = {'width': [], 'depth': []}
            for i in range(len(b_structure['width'])):
                assert b_structure['width'][i] == width_vectors[0].shape[1]
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

    def calc_macs(self):
        out_dict = {"prunable_macs": 0., "total_macs": 0., "cur_prunable_macs": 0., "cur_total_macs": 0.}

        blocks = list(zip(self.resnets, self.attentions))
        for (resnet, attention) in blocks:
            resnet_macs = resnet.calc_macs()

            for k in out_dict.keys():
                out_dict[k] = out_dict[k] + resnet_macs[k]

            attention_macs = attention.calc_macs()

            for k in out_dict.keys():
                out_dict[k] = out_dict[k] + attention_macs[k]

        for b in self.downsamplers:
            b_macs = 0
            for m in b.children():
                b_macs += m.__macs__
            out_dict["total_macs"] += b_macs
            out_dict["cur_total_macs"] += b_macs

        if self.total_macs == 0.:
            self.total_macs = out_dict["total_macs"]

        if self.prunable_macs == 0:
            self.prunable_macs = out_dict["prunable_macs"]

        return out_dict

    def get_prunable_macs(self):
        macs = []
        for b in self.resnets:
            macs.append(b.get_prunable_macs())
        for b in self.attentions:
            macs.append(b.get_prunable_macs())
        return macs

    def get_block_utilization(self):
        util = []
        blocks = list(zip(self.resnets, self.attentions))
        for (resnet, attention) in blocks:
            resnet_util = resnet.get_block_utilization()
            util.append(resnet_util)
            attention_util = attention.get_block_utilization()
            util.append(attention_util)
        return util


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
            gated_ff: bool = True,
            ff_gate_width: int = 32
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
                    skip_connection_dim=res_skip_channels,
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
                        gated_ff=gated_ff,
                        ff_gate_width=ff_gate_width
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
            gated_ff: bool = True,
            ff_gate_width: int = 32
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
                        gated_ff=gated_ff,
                        ff_gate_width=ff_gate_width
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
                    skip_connection_dim=res_skip_channels,
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
                        gated_ff=gated_ff,
                        ff_gate_width=ff_gate_width
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
        self.total_macs, self.prunable_macs = 0., 0.

    def get_gate_structure(self):
        if len(self.structure['width']) == 0:
            structure = {'width': [], 'depth': []}
            for b in self.resnets:
                assert hasattr(b, "get_gate_structure")
                b_structure = b.get_gate_structure()
                structure['width'].append(b_structure['width'])
                structure['depth'].append(b_structure['depth'])

            for b in self.attentions:
                assert hasattr(b, "get_gate_structure")
                b_structure = b.get_gate_structure()
                structure['width'].append(b_structure['width'])
                structure['depth'].append(b_structure['depth'])

            self.structure = structure

        return self.structure

    def set_gate_structure(self, arch_vectors):
        # We first read the resnets and then attentions in the "get_gate_structure" method.
        # Thus, we do a similar approach here.
        width_vectors, depth_vectors = arch_vectors['width'], arch_vectors['depth']

        for b in self.resnets:
            b_structure = b.get_gate_structure()
            block_vectors = {'width': [], 'depth': []}
            for i in range(len(b_structure['width'])):
                assert b_structure['width'][i] == width_vectors[0].shape[1]
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

    def calc_macs(self):
        out_dict = {"prunable_macs": 0., "total_macs": 0., "cur_prunable_macs": 0., "cur_total_macs": 0.}

        blocks = list(zip(self.resnets, self.attentions))
        for (resnet, attention) in blocks:
            resnet_macs = resnet.calc_macs()

            for k in out_dict.keys():
                out_dict[k] = out_dict[k] + resnet_macs[k]

            attention_macs = attention.calc_macs()

            for k in out_dict.keys():
                out_dict[k] = out_dict[k] + attention_macs[k]

        if self.upsamplers:
            for b in self.upsamplers:
                b_macs = 0
                for m in b.children():
                    b_macs += m.__macs__
                out_dict["total_macs"] += b_macs
                out_dict["cur_total_macs"] += b_macs

        if self.total_macs == 0.:
            self.total_macs = out_dict["total_macs"]

        if self.prunable_macs == 0:
            self.prunable_macs = out_dict["prunable_macs"]

        return out_dict

    def get_prunable_macs(self):
        macs = []
        for b in self.resnets:
            macs.append(b.get_prunable_macs())
        for b in self.attentions:
            macs.append(b.get_prunable_macs())
        return macs

    def get_block_utilization(self):
        util = []
        blocks = list(zip(self.resnets, self.attentions))
        for (resnet, attention) in blocks:
            resnet_util = resnet.get_block_utilization()
            util.append(resnet_util)
            attention_util = attention.get_block_utilization()
            util.append(attention_util)
        return util


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
        self.total_macs, self.prunable_macs = 0., 0.

    def get_gate_structure(self):
        if len(self.structure['width']) == 0:
            structure = {'width': [], 'depth': []}
            for b in self.resnets:
                assert hasattr(b, "get_gate_structure")
                b_structure = b.get_gate_structure()
                structure['width'].append(b_structure['width'])
                structure['depth'].append(b_structure['depth'])

            self.structure = structure
        return self.structure

    def set_gate_structure(self, arch_vectors):

        width_vectors, depth_vectors = arch_vectors['width'], arch_vectors['depth']
        for b in self.resnets:
            b_structure = b.get_gate_structure()
            block_vectors = {'width': [], 'depth': []}
            for i in range(len(b_structure['width'])):
                assert b_structure['width'][i] == width_vectors[0].shape[1]
                block_vectors['width'].append(width_vectors.pop(0))
            for i in range(len(b_structure['depth'])):
                if b_structure['depth'][i] == 1:
                    block_vectors['depth'].append(depth_vectors.pop(0))
            b.set_gate_structure(block_vectors)

    def calc_macs(self):
        out_dict = {"prunable_macs": 0., "total_macs": 0., "cur_prunable_macs": 0., "cur_total_macs": 0.}

        for resnet in self.resnets:
            resnet_macs = resnet.calc_macs()

            for k in out_dict.keys():
                out_dict[k] = out_dict[k] + resnet_macs[k]

        if self.downsamplers is not None:
            for b in self.downsamplers:
                b_macs = 0
                for m in b.children():
                    b_macs += m.__macs__

                out_dict["total_macs"] += b_macs
                out_dict["cur_total_macs"] += b_macs

        if self.total_macs == 0.:
            self.total_macs = out_dict["total_macs"]

        if self.prunable_macs == 0:
            self.prunable_macs = out_dict["prunable_macs"]

        return out_dict

    def get_prunable_macs(self):
        macs = []
        for b in self.resnets:
            macs.append(b.get_prunable_macs())
        return macs

    def get_block_utilization(self):
        util = []
        for b in self.resnets:
            util.append(b.get_block_utilization())
        return util


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
                    skip_connection_dim=res_skip_channels,
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
        self.total_macs, self.prunable_macs = 0., 0.

    def get_gate_structure(self):
        if len(self.structure['width']) == 0:
            structure = {'width': [], 'depth': []}
            for b in self.resnets:
                assert hasattr(b, "get_gate_structure")
                b_structure = b.get_gate_structure()
                structure['width'].append(b_structure['width'])
                structure['depth'].append(b_structure['depth'])

            self.structure = structure
        return self.structure

    def set_gate_structure(self, arch_vectors):

        width_vectors, depth_vectors = arch_vectors['width'], arch_vectors['depth']
        for b in self.resnets:
            b_structure = b.get_gate_structure()
            block_vectors = {'width': [], 'depth': []}
            for i in range(len(b_structure['width'])):
                assert b_structure['width'][i] == width_vectors[0].shape[1]
                block_vectors['width'].append(width_vectors.pop(0))
            for i in range(len(b_structure['depth'])):
                if b_structure['depth'][i] == 1:
                    block_vectors['depth'].append(depth_vectors.pop(0))
            b.set_gate_structure(block_vectors)

    def calc_macs(self):
        out_dict = {"prunable_macs": 0., "total_macs": 0., "cur_prunable_macs": 0., "cur_total_macs": 0.}

        for resnet in self.resnets:
            resnet_macs = resnet.calc_macs()

            for k in out_dict.keys():
                out_dict[k] = out_dict[k] + resnet_macs[k]

        if self.upsamplers is not None:
            for b in self.upsamplers:
                b_macs = 0
                for m in b.children():
                    b_macs += m.__macs__

                out_dict["total_macs"] += b_macs
                out_dict["cur_total_macs"] += b_macs

        if self.total_macs == 0.:
            self.total_macs = out_dict["total_macs"]

        if self.prunable_macs == 0:
            self.prunable_macs = out_dict["prunable_macs"]

        return out_dict

    def get_prunable_macs(self):
        macs = []
        for b in self.resnets:
            macs.append(b.get_prunable_macs())
        return macs

    def get_block_utilization(self):
        util = []
        for b in self.resnets:
            util.append(b.get_block_utilization())
        return util


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
            gated_ff: bool = False,
            ff_gate_width: int = 32
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
                        gated_ff=gated_ff,
                        ff_gate_width=ff_gate_width
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
        self.total_macs, self.prunable_macs = 0., 0.

    def get_gate_structure(self):
        if len(self.structure['width']) == 0:
            structure = {'width': [], 'depth': []}
            for b in self.resnets:
                assert hasattr(b, "get_gate_structure")
                b_structure = b.get_gate_structure()
                structure['width'].append(b_structure['width'])
                structure['depth'].append(b_structure['depth'])

            for b in self.attentions:
                assert hasattr(b, "get_gate_structure")
                b_structure = b.get_gate_structure()
                structure['width'].append(b_structure['width'])
                structure['depth'].append(b_structure['depth'])

            self.structure = structure

        return self.structure

    def set_gate_structure(self, arch_vectors):
        # We first read the resnets and then attentions in the "get_gate_structure" method.
        # Thus, we do a similar approach here.
        width_vectors, depth_vectors = arch_vectors['width'], arch_vectors['depth']

        for b in self.resnets:
            b_structure = b.get_gate_structure()
            block_vectors = {'width': [], 'depth': []}
            for i in range(len(b_structure['width'])):
                assert b_structure['width'][i] == width_vectors[0].shape[1]
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

    def calc_macs(self):
        out_dict = {"prunable_macs": 0., "total_macs": 0., "cur_prunable_macs": 0., "cur_total_macs": 0.}

        for resnet in self.resnets:
            resnet_macs = resnet.calc_macs()
            for k in out_dict.keys():
                out_dict[k] = out_dict[k] + resnet_macs[k]

        for attention in self.attentions:
            attention_macs = attention.calc_macs()
            for k in out_dict.keys():
                out_dict[k] = out_dict[k] + attention_macs[k]

        if self.total_macs == 0.:
            self.total_macs = out_dict["total_macs"]

        if self.prunable_macs == 0:
            self.prunable_macs = out_dict["prunable_macs"]

        return out_dict

    def get_prunable_macs(self):
        macs = []
        for b in self.resnets:
            macs.append(b.get_prunable_macs())
        for b in self.attentions:
            macs.append(b.get_prunable_macs())
        return macs

    def get_block_utilization(self):
        util = []
        for b in self.resnets:
            util.append(b.get_block_utilization())
        for b in self.attentions:
            util.append(b.get_block_utilization())
        return util
