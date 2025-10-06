import argparse

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="BVP Image Interpolation Algorithm")

    # Model and paths
    parser.add_argument(
        "--path_to_pretrained_model",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Path or name of pretrained model"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Directory for output images"
    )

    # Generation parameters
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps for inference"
    )
    parser.add_argument(
        "--num_output_imgs",
        type=int,
        default=7,
        help="Number of output images to generate"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale for generation"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.002,
        help="Strength of image conditioning (0 = input image, 1 = full noise)"
    )
    parser.add_argument(
        "--high_noise_frac",
        type=float,
        default=0.6,
        help="Fraction of high noise to apply"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args
