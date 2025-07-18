import torch
from diffusers.utils import load_image, check_min_version
from diffusers.pipelines import StableDiffusion3ControlNetInpaintingPipeline
from diffusers.models.controlnet_sd3 import SD3ControlNetModel
from PIL import Image
from torchao.quantization import quantize_, int8_weight_only

controlnet = SD3ControlNetModel.from_pretrained(
    "alimama-creative/SD3-Controlnet-Inpainting",
    use_safetensors=True,
    extra_conditioning_channels=1)
quantize_(controlnet, int8_weight_only())

pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.text_encoder.to(torch.float16)
pipe.controlnet.to(torch.float16)
# pipe.to("cuda")

image = Image.open("render.png").convert("RGB")
mask = Image.open("mask_new.png").convert("L")

width = 1024
height = 1024
prompt = "a city town square"
generator = torch.Generator(device="cuda").manual_seed(24)
res_image = pipe(
    negative_prompt="",
    prompt=prompt,
    height=height,
    width=width,
    control_image=image,
    control_mask=mask,
    num_inference_steps=28,
    generator=generator,
    controlnet_conditioning_scale=0.95,
    guidance_scale=7,
).images[0]
res_image.save(f"sd3.png")
