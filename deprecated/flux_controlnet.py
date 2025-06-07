from torchao.quantization import quantize_, int8_weight_only
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from transformer_flux import FluxTransformer2DModel
from controlnet_flux import FluxControlNetModel
from diffusers.utils import load_image, check_min_version
import torch
import sys

sys.path.append("/home/cedric/PanoramaGenInpaint/FLUX-Controlnet-Inpainting")


def load_controlnet_pipeline():
    controlnet = FluxControlNetModel.from_pretrained(
        "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha",
        torch_dtype=torch.bfloat16)
    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        torch_dtype=torch.bfloat16)

    quantize_(transformer, int8_weight_only())
    quantize_(controlnet, int8_weight_only())

    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)
    return pipe


if __name__ == "__main__":
    pipe = load_controlnet_pipeline()
