import torch
from diffusers.utils import load_image, check_min_version
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from torchao.quantization import quantize_, int8_weight_only
from PIL import Image
import numpy as np

check_min_version("0.30.2")

# Set image path , mask path and prompt
image_path='render.png'
mask_path='mask_new.png'
prompt='a city town square'


def load_contolnet_pipeline():
	import sys 
	sys.path.append('/home/cedric/PanoramaGenInpaint/FLUX-Controlnet-Inpainting')
	from diffusers.utils import load_image, check_min_version
	from controlnet_flux import FluxControlNetModel
	from transformer_flux import FluxTransformer2DModel
	from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
	from torchao.quantization import quantize_, int8_weight_only

	controlnet = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha", torch_dtype=torch.bfloat16)
	transformer = FluxTransformer2DModel.from_pretrained(
			"black-forest-labs/FLUX.1-dev", subfolder='transformer', torch_dtype=torch.bfloat16)
	
	quantize_(transformer, int8_weight_only())
	quantize_(controlnet, int8_weight_only())
	pipe = FluxControlNetInpaintingPipeline.from_pretrained(
		"black-forest-labs/FLUX.1-dev",
		controlnet=controlnet,
		transformer=transformer,
		torch_dtype=torch.bfloat16
	)
	pipe.enable_model_cpu_offload()
	pipe.transformer.to(torch.bfloat16)
	pipe.controlnet.to(torch.bfloat16)
	return pipe 



def outpaint_controlnet(pipe,image, mask,vis=False,num_steps=50,prompt='a city town square'):
	generator = torch.Generator(device="cpu").manual_seed(24)
	# Inpaint
	size = (768, 768)
	result = pipe(
		prompt=prompt,
		height=size[1],
		width=size[0],
		control_image=image,
		control_mask=mask,
		num_inference_steps=num_steps,
		generator=generator,
		controlnet_conditioning_scale=0.9,
		guidance_scale=3.5,
		negative_prompt="",
		true_guidance_scale=1.0 # default: 3.5 for alpha and 1.0 for beta
	).images[0]
	

	return result


pipe = load_contolnet_pipeline()

# Load image and mask
size = (768, 768)
image = Image.open(image_path).convert("RGB").resize(size)
mask = Image.open(mask_path).convert("L").resize(size)

result = outpaint_controlnet(pipe,image, mask, num_steps=100, prompt=prompt)

result.save('flux_inpaint.png')
print("Successfully inpaint image")

import numpy as np
mask_np = np.array(mask.convert("L"))
mask_binary = mask_np > 128  # Create binary mask (True where mask is white)
# Convert images to numpy arrays
original_np = np.array(image)
result_np = np.array(result)

# Combine images: use original image and replace only the masked area with the result
combined_np = original_np.copy()
combined_np[mask_binary] = result_np[mask_binary]

# Convert back to PIL Image
combined_result = Image.fromarray(combined_np)

# Save both the raw inpainting result and the combined result
result.save('flux_inpaint_full.png')
combined_result.save('flux_inpaint_combined.png')
print("Successfully inpainted image and combined with original using mask")