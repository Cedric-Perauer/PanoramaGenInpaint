import torch
from diffusers.utils import load_image, check_min_version
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from torchao.quantization import quantize_, int8_weight_only
from PIL import Image

check_min_version("0.30.2")

# Set image path , mask path and prompt
image_path='bucket.png'
mask_path='bucket_mask.jpeg'
prompt='a person wearing a white shoe, carrying a white bucket with text "FLUX" on it'

# Build pipeline
controlnet = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha", torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder='transformer', torch_dtype=torch.bfloat16
    )
quantize_(transformer, int8_weight_only())
quantize_(controlnet, int8_weight_only())
pipe = FluxControlNetInpaintingPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cpu")


pipe.enable_model_cpu_offload()
pipe.transformer.to(torch.bfloat16)
pipe.controlnet.to(torch.bfloat16)


# Load image and mask
size = (768, 768)
image = Image.open(image_path).convert("RGB").resize(size)
mask = Image.open(mask_path).convert("RGB").resize(size)
generator = torch.Generator(device="cpu").manual_seed(24)

# Inpaint
result = pipe(
    prompt=prompt,
    height=size[1],
    width=size[0],
    control_image=image,
    control_mask=mask,
    num_inference_steps=28,
    generator=generator,
    controlnet_conditioning_scale=0.9,
    guidance_scale=3.5,
    negative_prompt="",
    true_guidance_scale=1.0 # default: 3.5 for alpha and 1.0 for beta
).images[0]

result.save('flux_inpaint.png')
print("Successfully inpaint image")
