# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import FluxTransformer2DModel
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention_processor import (
    Attention,
)
from diffusers.models.normalization import AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.utils import logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from ..gates import WidthGate
from ..attention import GatedFeedForward, GatedAttention, GatedFluxAttnProcessor2_0


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@maybe_allow_in_graph
class GatedFluxSingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        mlp_ratio (`float`, *optional*, defaults to 4.0): The ratio of the hidden layer size to the input size.

    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0, ff_gate_width=32):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        processor = GatedFluxAttnProcessor2_0()
        self.attn = GatedAttention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

        self.ff_gate = WidthGate(ff_gate_width)

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            image_rotary_emb=None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        mlp_hidden_states = self.ff_gate(mlp_hidden_states)

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states

    def get_structure(self):
        return {"width": [self.attn.gate.width, self.ff_gate.width]}


@maybe_allow_in_graph
class GatedFluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6, ff_gate_width=32):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)

        self.norm1_context = AdaLayerNormZero(dim)

        if hasattr(F, "scaled_dot_product_attention"):
            processor = GatedFluxAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = GatedFeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate", gate_width=ff_gate_width)

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = GatedFeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate",
                                           gate_width=ff_gate_width)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            image_rotary_emb=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def get_structure(self):
        return {"width": [self.attn.gate.width, self.ff.net[0].gate.width, self.ff_context.net[0].gate.width]}


class GatedFluxTransformer2DModel(FluxTransformer2DModel):
    """
    The Transformer model introduced in Flux with Gated Transformer blocks.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
        axes_dims_rope (`Tuple[int]`, *optional*, defaults to (16, 56, 56)): The dimensions of the axes in the
         Rotary Positional Embeddings.
        ff_gate_width (`int`, *optional*, defaults to 32): The width of the gate in the feed-forward layer.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["GatedFluxTransformerBlock", "GatedFluxSingleTransformerBlock"]

    @register_to_config
    def __init__(
            self,
            patch_size: int = 1,
            in_channels: int = 64,
            num_layers: int = 19,
            num_single_layers: int = 38,
            attention_head_dim: int = 128,
            num_attention_heads: int = 24,
            joint_attention_dim: int = 4096,
            pooled_projection_dim: int = 768,
            guidance_embeds: bool = False,
            axes_dims_rope: Tuple[int] = (16, 56, 56),
            ff_gate_width: int = 32,
    ):
        super().__init__(patch_size=patch_size, in_channels=in_channels, num_layers=num_layers,
                         num_single_layers=num_single_layers, attention_head_dim=attention_head_dim,
                         num_attention_heads=num_attention_heads, joint_attention_dim=joint_attention_dim,
                         pooled_projection_dim=pooled_projection_dim, guidance_embeds=guidance_embeds,
                         axes_dims_rope=axes_dims_rope)

        self.transformer_blocks = nn.ModuleList(
            [
                GatedFluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    ff_gate_width=self.config.ff_gate_width,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                GatedFluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    ff_gate_width=self.config.ff_gate_width,
                )
                for _ in range(self.config.num_single_layers)
            ]
        )

        self.structure = None

    def freeze(self):
        # Freeze all parameters except the gate_f parameters
        for name, param in self.named_parameters():
            if "gate_f" not in name:
                param.requires_grad = False

    def get_structure(self):
        if self.structure is None:
            structure = {"width": [], "depth": []}
            for block in self.transformer_blocks:
                structure["width"].append(block.get_structure()["width"])
            for block in self.single_transformer_blocks:
                structure["width"].append(block.ff_gate.get_structure()["width"])
            self.structure = structure
        return self.structure
