from pipeline import CustomStableDiffusionPipeline
from args_parser import parse_args
from bvp_algorithm import BVPAlgorithm
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
import torch
import os
import yaml

def main(args):
    device = torch.device("cuda")
    torch_dtype = torch.float32

    scheduler = DDIMScheduler.from_pretrained(args.path_to_pretrained_model, subfolder='scheduler')
    pipe = CustomStableDiffusionPipeline.from_pretrained(args.path_to_pretrained_model, scheduler=scheduler, torch_dtype=torch.float32)

    pipe.to(device)
    pipe.set_resolution(args.resolution)
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)

    os.makedirs(args.output_dir, exist_ok=True)

    with open("configs/config.yaml") as f:
        cfg_dict = yaml.safe_load(f)

    cfg_dict.update({
        'device': device,
        'pipe': pipe,
        'resolution': args.resolution,
        'num_output_imgs': args.num_output_imgs,
        'output_dir': args.output_dir,
    })

    config = BVPConfig(**cfg_dict)
    # interpolation = BVPAlgorithm(device, pipe, config)
    interpolation = BVPAlgorithm(config)
    
    interpolation.init()
    interpolation.optimise()

if __name__ == "__main__":
    args = parse_args()
    main(args)
