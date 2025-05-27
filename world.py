import torch
from diffusers import FluxPipeline
from diffusers.utils import load_image
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from utils import preprocess_image,pil_to_cv2,cv2_to_pil, show_image_cv2, clear_gpu_memory,run_with_conda_env
import copy
from diffusion_utils import load_pipeline,generate_outpaint,fix_mask, fix_inpaint_mask, vis_inpaint_strategy,composite_with_mask, load_contolnet_pipeline,outpaint_controlnet
import torch
from dust3r_infer import get_focals
from image_utils import *
import copy

GEN = False
USE_SDXL = False

IMAGE_SIZE = 1024
if GEN == True:
	pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
	pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
	prompt = "a town square in a city with a person standing in the center"
	image = pipe(
		prompt,
		guidance_scale=0.0,
		num_inference_steps=4,
		max_sequence_length=256,
		generator=torch.Generator("cpu").manual_seed(0)
	).images[0]
	image.save("middle.jpg")
	clear_gpu_memory()
	focals = get_focals('middle.jpg', 512)
	source_image_cv2 = pil_to_cv2(image)
	show_image_cv2(source_image_cv2)

all_views_data = vis_inpaint_strategy()

def load_contolnet_pipeline():
	import sys 
	import torch
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
	return pipe, controlnet, transformer


top_and_bottom_views = [all_views_data[0],all_views_data[3]]
side_views = all_views_data[6:]
clear_gpu_memory()

pipeline, controlnet, transformer = load_contolnet_pipeline()
 
'''
for view in top_and_bottom_views[:2] :
    mask = view['mask']
    mask[mask==255] = 1
    new_mask = fix_inpaint_mask(mask*255)
    #show_image_cv2(new_mask)
    inital_pano = Image.open("initial_pano_with_back.png")
    initial_pano_np = np.array(inital_pano)
    render_img = cv2.cvtColor(view['render'], cv2.COLOR_BGR2RGB)
    #show_image_cv2(render_img)
    render_img = cv2_to_pil(render_img)

    print(f"Render image shape: {render_img.size}{mask.shape}")
    #image = Image.open('top1.png')
    image = generate_outpaint(pipeline, render_img, new_mask,vis=True,prompt='floor of a city town square')
    final_pano = project_perspective_to_equirect(
        cv2.cvtColor(pil_to_cv2(image),cv2.COLOR_BGR2RGB), 
        initial_pano_np,
        yaw_deg=-view['yaw'], 
        pitch_deg=view['pitch'],
        h_fov_deg=view['fov'], 
        v_fov_deg=view['fov'],
        mask=mask  # Original mask - only project where we outpainted
    )

    # Visualize the updated panorama
    plt.figure(figsize=(20, 10))
    plt.imshow(final_pano)
    plt.title(f"Panorama after outpainting {view['label']}")
    plt.axis('off')
    plt.show()
    #break
'''


inital_pano = Image.open("imgs/initial_pano_with_back.png")
initial_pano_np = np.array(inital_pano)
output_size = 1024
'''
for view in top_and_bottom_views :
    show_image_cv2(cv2.cvtColor(initial_pano_np, cv2.COLOR_BGR2RGB))
    render_img = render_perspective(
                initial_pano_np, view['yaw'], -view['pitch'], view['fov'], view['vfov'], output_size
    )
    
    mask = create_mask_from_black(render_img, threshold=10)
    
    mask[mask==255] = 1
    new_mask = fix_inpaint_mask(mask*255)
    
    render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
    #show_image_cv2(render_img)
    render_img = cv2_to_pil(render_img)
    if view['label'] == 'Top' : 
        prompt = 'Floor of a city town square'      
    else : 
        prompt = 'Sky of a city town square, no buildings'
    
    
	
    print(f"Render image shape: {render_img.size}{mask.shape}")
    #image = Image.open('top1.png')
    image = generate_outpaint(pipeline, render_img, new_mask,vis=True,prompt=prompt)
    image.save("outpainted_test.png")
    #image = Image.open("outpainted_test.png")
    initial_pano_np = project_perspective_to_equirect(
        cv2.cvtColor(pil_to_cv2(image),cv2.COLOR_BGR2RGB), 
        initial_pano_np,
        yaw_deg=view['yaw'], 
        pitch_deg=view['pitch'],
        h_fov_deg=view['fov'], 
        v_fov_deg=view['fov'],
        mask=None,  # Original mask - only project where we outpainted
        mirror=False
    )

    # Visualize the updated panorama
    plt.figure(figsize=(20, 10))
    plt.imshow(initial_pano_np)
    plt.title(f"Panorama after outpainting {view['label']}")
    plt.axis('off')
    plt.show()
    #break
'''
side_view_pano = Image.open("imgs/initial_pano_center.png")
side_view_pano_np = np.array(side_view_pano)
for idx,view in enumerate(side_views[:2]):
    show_image_cv2(cv2.cvtColor(initial_pano_np, cv2.COLOR_BGR2RGB))
    render_img = render_perspective(
                side_view_pano_np, view['yaw'], -view['pitch'], view['fov'], view['vfov'], output_size
    )
    show_image_cv2(cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB))
    
    mask = create_mask_from_black(render_img, threshold=10)
    new_mask = fix_inpaint_mask(mask,extend_amount=100)
    new_mask = Image.fromarray(new_mask).convert("L")
    new_mask.save(f"new_mask_{idx}.png")
    render_img = Image.fromarray(render_img).convert("RGB")

    show_image_cv2(pil_to_cv2(new_mask))
    
    prompt = 'a different house on a market square'      
    
    print(f"Render image shape: {render_img.size}{mask.shape}")
    #image = Image.open('top1.png')
    image = outpaint_controlnet(pipeline, render_img, new_mask,vis=True,prompt=prompt,num_steps=30,guidance_scale=3.5,cond_scale=0.4)
    image = image.resize((1024, 1024), Image.LANCZOS)
    image.save(f"imgs/render.png")
    
    # Aggressive GPU memory clearing
    if torch.cuda.is_available():
        # Move pipeline and all its components to CPU
        try:
            pipeline.to("cpu")
            if hasattr(pipeline, 'controlnet'):
                pipeline.controlnet.to("cpu")
            if hasattr(pipeline, 'transformer'):
                pipeline.transformer.to("cpu")
            if hasattr(pipeline, 'unet'):
                pipeline.unet.to("cpu")
            if hasattr(pipeline, 'vae'):
                pipeline.vae.to("cpu")
        except:
            pass
            
        # Clear all CUDA caches and memory
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        # Force CUDA to release memory
        torch.cuda.set_per_process_memory_fraction(0.0)
        torch.cuda.set_per_process_memory_fraction(1.0)
    
    # Clear Python garbage collection multiple times
    import gc
    gc.collect()
    gc.collect()
    gc.collect()
    
    # Delete all possible references
    try:
        del pipeline
        del render_img
        del controlnet
        del transformer
        del new_mask
        del image
    except:
        pass
        
    clear_gpu_memory()
    breakpoint()
    run_with_conda_env('diffusers33', 'refiner.py imgs/render.png')
    image = Image.open("refined_output.png")
    

    new_mask = np.array(new_mask)
    #image = generate_outpaint(pipeline, image, np.ones_like(new_mask),vis=True,num_steps=20,prompt=prompt)
    #image = Image.open("outpainted_test.png")
    cur_mask = new_mask 
    initial_pano_np = project_perspective_to_equirect(
        cv2.cvtColor(pil_to_cv2(image),cv2.COLOR_BGR2RGB), 
        initial_pano_np,
        yaw_deg=view['yaw'], 
        pitch_deg=view['pitch'],
        h_fov_deg=view['fov'], 
        v_fov_deg=view['fov'],
        mask=new_mask,  # Original mask - only project where we outpainted,
        mirror=False
    )
    
    side_view_pano_np = project_perspective_to_equirect(
        cv2.cvtColor(pil_to_cv2(image),cv2.COLOR_BGR2RGB), 
        side_view_pano_np,
        yaw_deg=view['yaw'], 
        pitch_deg=view['pitch'],
        h_fov_deg=view['fov'], 
        v_fov_deg=view['fov'],
        mask=new_mask,  # Original mask - only project where we outpainted,
        mirror=False
    )

    # Visualize the updated panorama
    plt.figure(figsize=(20, 10))
    plt.imshow(initial_pano_np)
    plt.title(f"Panorama after outpainting {view['label']}")
    plt.axis('off')
    plt.show()

'''
for idx,view in enumerate(side_views) : 
    mask = view['mask']
    mask[mask==255] = 1
    #mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
    render_img = cv2.cvtColor(view['render'], cv2.COLOR_BGR2RGB)
    show_image_cv2(render_img)
    render_img = cv2_to_pil(render_img)
    
    print(f"Render image shape: {render_img.size}{mask.shape}")
    image = generate_outpaint(pipeline, render_img, mask,vis=True)
    if idx == 1 : 
    	break
'''