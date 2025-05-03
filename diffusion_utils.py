from diffusers import FluxFillPipeline, FluxPipeline,DiffusionPipeline, FluxFillPipeline, FluxTransformer2DModel
import torch 
from transformers import T5EncoderModel
from utils import *
import numpy as np
import copy
from image_utils import visualize_all_inpainting_masks

IMAGE_SIZE = 1024

def vis_inpaint_strategy(vis=False):
	try:
		initial_pano_pil = Image.open("initial_pano_with_back.png")
		side_pano = Image.open("initial_pano_center.png")
		initial_pano_np = np.array(initial_pano_pil)
		side_pano_np = np.array(side_pano)

		# Generate the visualization data
		all_views_data = visualize_all_inpainting_masks(
			initial_panorama=initial_pano_np,
			side_pano=side_pano_np,
			output_size=1024 # Should match inpainting model input size
		)

		# --- Plot the results ---
		num_views = len(all_views_data)
		# Create a 3-column plot: Render | Mask | Highlighted Pano Footprint
		fig, axes = plt.subplots(num_views, 3, figsize=(12, 3 * num_views)) # Width, Height
		fig.suptitle("Individual Inpainting View Masks and Projections", fontsize=16)

		if num_views == 1: # Handle case of single view if axes is not a 2D array
			axes = np.array([axes])

		for i, data in enumerate(all_views_data):
			ax_render = axes[i, 0]
			ax_mask = axes[i, 1]
			ax_highlight = axes[i, 2]

			# Column 1: Rendered Perspective
			ax_render.imshow(data['render'])
			ax_render.set_title(f"{data['label']}\nY={data['yaw']}, P={data['pitch']}, FoV={data['fov']}\nRendered View", fontsize=8)
			ax_render.axis('off')

			# Column 2: Generated Mask
			ax_mask.imshow(data['mask'], cmap='gray')
			ax_mask.set_title("Mask (White=Inpaint)", fontsize=8)
			ax_mask.axis('off')

			# Column 3: Highlighted Footprint on Equirectangular
			ax_highlight.imshow(data['highlighted_pano'])
			ax_highlight.set_title("Mask Footprint on Pano", fontsize=8)
			ax_highlight.axis('off')

		plt.tight_layout(rect=[0, 0.01, 1, 0.98]) # Adjust layout
		plt.show()
		return all_views_data

	except FileNotFoundError:
		print("Error: 'initial_pano_with_back.png' not found.")
		print("Please ensure you have created the initial panorama with front/back duplication first.")
	except Exception as e:
		print(f"An error occurred: {e}")


def fix_inpaint_mask(mask, contour_color=(0, 255, 0), fill_color=(0, 0, 0),extend_amount=100):
    mask_copy = mask.copy()
    if mask_copy.dtype != np.uint8:
    
        mask_copy = (mask_copy * 255).astype(np.uint8)
    
    
    inverted = cv2.bitwise_not(mask_copy)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(mask_copy, contours, -1, 0 , -1)
    # Now extend the mask into black regions
    if extend_amount > 0:
        kernel = np.ones((extend_amount, extend_amount), np.uint8)
        mask_copy = cv2.dilate(mask_copy, kernel, iterations=1)
        
    blur_amount = 20
    if blur_amount > 0:
        mask_copy = cv2.GaussianBlur(mask_copy, (blur_amount*2+1, blur_amount*2+1), 0)
        # Normalize back to proper range
        mask_copy = mask_copy.astype(np.float32) / 255.0
    
    return mask_copy
    
    return mask_copy 

def load_pipeline(four_bit=False):
	orig_pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
	print("Using four bit.")
	transformer = FluxTransformer2DModel.from_pretrained(
		"sayakpaul/FLUX.1-Fill-dev-nf4", subfolder="transformer", torch_dtype=torch.bfloat16
	)
	text_encoder_2 = T5EncoderModel.from_pretrained(
		"sayakpaul/FLUX.1-Fill-dev-nf4", subfolder="text_encoder_2", torch_dtype=torch.bfloat16
	)
	pipeline = FluxFillPipeline.from_pipe(
		orig_pipeline, transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=torch.bfloat16
	)

	pipeline.enable_model_cpu_offload()
	return pipeline

def generate_outpaint(pipe,image, mask,vis=False,use_flux=False,prompt='a city town square'):
	if use_flux:
		image = pipe(
			prompt=prompt,
			image=preprocess_image(image),
			mask_image=mask,
			height=IMAGE_SIZE,
			width=IMAGE_SIZE,
			guidance_scale=30,
			num_inference_steps=50,
			max_sequence_length=512,
			generator=torch.Generator("cpu").manual_seed(0)
		).images[0]
	else : 
		image = pipe(
			prompt=prompt,
			image=preprocess_image(image),
			mask_image=mask,
			height=IMAGE_SIZE,
			width=IMAGE_SIZE,
			max_sequence_length=512,
			generator=torch.Generator("cpu").manual_seed(0),
			).images[0]
	if vis:
		show_image_cv2(pil_to_cv2(image))
	return image


def fix_mask(image,source_image,fill_shape,offset,fill_size,left,pipe,vis=False):
	clear_gpu_memory()

	mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE+fill_shape), dtype=np.uint8)
	mask_width = 10
	if left : 
		mask[:,IMAGE_SIZE + offset -mask_width:IMAGE_SIZE + offset +mask_width] = 1 
	else : 
		mask[:,fill_size-mask_width:mask_width+fill_size] = 1
	blur_amount = 50
	blurred_mask = cv2.GaussianBlur(mask*255, (blur_amount*2+1, blur_amount*2+1), 0) * 255
	blurred_mask = blurred_mask.astype(np.float32)
	image_gen = pipe(
		prompt="",
		image=preprocess_image(cv2_to_pil(image)),
		mask_image=blurred_mask,
		height=IMAGE_SIZE,
		width=IMAGE_SIZE+fill_shape,
		guidance_scale=40,
		num_inference_steps=10,
		max_sequence_length=512,
		generator=torch.Generator("cpu").manual_seed(0)
	).images[0]
	return image_gen
