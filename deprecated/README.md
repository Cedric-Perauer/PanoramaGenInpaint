# Training free panorama generation with inpainting via equirectangular mapping

This code implements a 360 panorama generation pipeline based on the 2025 Meta paper ["A Recipe for Generating 3D Worlds From a Single Image"](https://arxiv.org/abs/2503.16611). 


## Example Outputs 

Prompt : "a market square in the 1800s" 
<img src="assets/cur_pano_v2.png" alt="My Image" width="1200">


Prompt : "a modern japanese garden with a pond and a waterfall" 
<img src="assets/garden.png" alt="My Image" width="1200">


## ToDo 
- [ ] readme explanation


## Install 

```bash
conda create -n pano_gen python=3.10
conda activate pano_gen
pip install -r requirements.txt
```

## Run the Code 
```bash
python world.py --scene_prompt "a modern japanese garden with a pond and a waterfall" --scene_prompt_sides "a modern japanese garden " --sky_prompt "a clear blue sky"
```

```bash
--scene_prompt : prompt for the middle center image generation 
--scene_prompt_sides : prompt for the sides of the image => should describe general scene layout, but can be less descriptive than scene_prompt
--sky_prompt : prompt for the sky image generation 
--debug : optional mode that stores all inpainting masks, etc.
```

