import torch
from diffusers import FluxPipeline
from diffusers.utils import load_image

def preprocess_image(image):
    image = image.convert("RGB")
    image = transforms.CenterCrop((image.size[1] // 64 * 64, image.size[0] // 64 * 64))(image)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0).to("cuda")
    return image

def preprocess_mask(mask):
    mask = mask.convert("L")
    mask = transforms.CenterCrop((mask.size[1] // 64 * 64, mask.size[0] // 64 * 64))(mask)
    mask = transforms.ToTensor()(mask)
    mask = mask.to("cuda")
    return mask


pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

mask = preprocess_mask(
    load_image(
        "./overture-creations-5sI6fQgYIuo_mask.png"
    )
)

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    mask_image=mask,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-schnell.png")