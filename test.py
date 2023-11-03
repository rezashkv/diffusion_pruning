import torchvision

from op_counter import count_ops_and_params
from blocks import ResnetBlock2DGated, BasicTransformerBlockGated
import torch
from diffusers.models.transformer_2d import Transformer2DModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler, EulerDiscreteScheduler

from diffusers import AutoencoderKL
from unet_2d_conditional import UNet2DConditionModelGated

# block = ResnetBlock2DGated(in_channels=320,temb_channels=None)
# attn_block = BasicTransformerBlockGated(16 * 88, 16, 88,
#                                         dropout=0.0,
#                                         attention_bias=False,
#                                         activation_fn="geglu",
#                                         num_embeds_ada_norm=None,
#                                         only_cross_attention=False,
#                                         double_self_attention=False,
#                                         upcast_attention=False,
#                                         norm_type="layer_norm",
#                                         norm_elementwise_affine=True,
#                                         attention_type="default",
#                                         cross_attention_dim=1408,
#                                         )


# class Net(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.blocks = torch.nn.ModuleList([block for _ in range(10)])
#
#     def forward(self, x, temb):
#         for block in self.blocks:
#             x = block(x, temb)
#         return x


# model = Transformer2DModel(num_attention_heads=16, attention_head_dim=88, in_channels=320)

# a random input for the model
# x = torch.randn(1, 320, 32, 32)
# timpestep = 100

# out = model(x, timestep=timpestep)
# out = block(x, temb=None)
# model = UNet2DConditionModel(cross_attention_dim=50)
# x = torch.randn(1, 4, 32, 32)
# encoder_hidden = torch.randn(1, 4, 50)
# print(count_ops_and_params(model, {"sample": x, "timestep": 100, "encoder_hidden_states": encoder_hidden}))
# print(attn_block.get_flops(x))
# out = attn_block(x, timestep=timpestep)
# print(out.shape)


unet = UNet2DConditionModelGated.from_pretrained(
    "stabilityai/stable-diffusion-2", subfolder="unet", down_block_types=(
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2DHalfGated",
        "CrossAttnDownBlock2DGated",
        "DownBlock2DGated",
    ), up_block_types=(
        "UpBlock2DHalfGated", "CrossAttnUpBlock2DGated", "CrossAttnUpBlock2DGated", "CrossAttnUpBlock2DHalfGated"),
    mid_block_type="UNetMidBlock2DCrossAttn")
# #
# # # model = UNet2DConditionModel(cross_attention_dim=50)
# x = torch.randn(1, 4, 64, 64)
# encoder_hidden = torch.randn(1, 4, 1024)
# print(count_ops_and_params(net, {"sample": x, "timestep": 100, "encoder_hidden_states": encoder_hidden}))

model_id = "stabilityai/stable-diffusion-2-1"
# scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

# pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"

# image = pipe(prompt, height=768, width=768).images[0]
#
# image.save("astronaut_rides_horse.png") # works fine

pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, unet=unet)
pipe = pipe.to("cuda")

prompt = ("Realistic painting of a dystopian industrial city with towering factories, pollution-filled air, and a gloomy sky")
image = pipe(prompt).images[0]
image.save("test.png")
# save the pil image


# dis
# 378438112384.0
# 378438112384.0
