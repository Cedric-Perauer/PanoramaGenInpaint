"""Compute depth maps for images in the input folder.
"""

import os
import glob
import torch
import utils
import cv2
import argparse

from torchvision.transforms import Compose
from models.midas_net import MidasNet
from models.transforms import Resize, NormalizeImage, PrepareForNet


def run(input_path, output_path, model_path, process_res):
    """Run MonoDepthNN to compute depth maps.
    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda")
    print("device: %s" % device)

    # load network
    model = MidasNet(model_path, non_negative=True)

    transform = Compose(
        [
            Resize(
                process_res,
                process_res,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(
                mean=[
                    0.485,
                    0.456,
                    0.406],
                std=[
                    0.229,
                    0.224,
                    0.225]),
            PrepareForNet(),
        ])

    model.to(device)
    model.eval()

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")

    for ind, img_name in enumerate(img_names):

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

        # input

        img = utils.read_image(img_name)
        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # output
        filename = os.path.join(
            output_path, os.path.splitext(
                os.path.basename(img_name))[0])
        utils.write_depth(filename, prediction, bits=2)

    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--res", required=True, type=int)
    args = parser.parse_args()

    MODEL_PATH = "model.pt"

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(args.input_dir, args.output_dir, MODEL_PATH, args.res)
