import torch
import random
from pipeline_flux import FluxPipeline  # use our modifed flux pipeline to ensure close-loop.
from torchao.quantization import quantize_, int8_weight_only

lora_path = "lora_hubs/pano_lora_720*1440_v1.safetensors"  # download panorama lora in our huggingface repo and replace it to your path.
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights(lora_path)  # change this.
breakpoint()
quantize_(pipe.transformer, int8_weight_only())
quantize_(pipe.vae, int8_weight_only())
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
pipe.transformer.to(torch.bfloat16)
pipe.vae.to(torch.bfloat16)

prompt = "A vibrant city avenue, bustling traffic, towering skyscrapers"

pipe.enable_vae_tiling()
seed = 119223

# Select the same resolution as LoRA for inference
image = pipe(
    prompt,
    height=512,
    width=1024,
    generator=torch.Generator("cpu").manual_seed(seed),
    num_inference_steps=50,
    blend_extend=6,
    guidance_scale=7,
).images[0]

image.save("result.png")
