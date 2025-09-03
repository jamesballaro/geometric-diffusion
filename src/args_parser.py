import argparse

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Find Jacobian of UNet prediction")
    parser.add_argument(
        "--path_to_pretrained_model",
        type=str,
        default=None,
        required=True,
        help=("Path of SDXL model checkpoints")
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        required=True,
        help=("Text prompt for SDXL image generation")
    )
    parser.add_argument(
        "--path_to_image",
        type=str,
        default=None,
        required=True,
        help=("Path to input image")
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
        help=("Path to output image")
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        required=False,
        help=("Number of denoising steps for inference")
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=8.0,
        required=False,
        help=("Guidance scale")
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.5,
        required=False,
        help=("Strength of image conditioning (0 = input image, 1 = full noise)")
    )
    parser.add_argument(
        "--high_noise_frac",
        type=float,
        default=0.5,
        required=False,
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args
