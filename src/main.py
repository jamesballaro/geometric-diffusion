from pipeline import CustomStableDiffusionPipeline
from args_parser import parse_args
from bvp_algorithm import BVPAlgorithm
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
import torch
import os
import yaml

def main():
    device = torch.device("cuda")
    torch_dtype = torch.float32
    model_id = "stabilityai/stable-diffusion-2-1-base"

    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipe = CustomStableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32)

    pipe.set_resolution(512)

    pipe.to(device)
    pipe.scheduler.set_timesteps(50, device=device)
    pipe.set_seed(67280421310721)

    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)

    num_output_imgs = 7
    output_dir = 'results/'
    os.makedirs(output_dir, exist_ok=True)

    # with open("configs/config.yaml") as f:
    #     cfg = yaml.safe_load(f)

    interpolation = BVPAlgorithm(
        device=device,
        pipe = pipe,
        test_name = 'dog_test',
        output_dir = output_dir,
        image_path1 = 'assets/dog17_0.png',
        image_path2 ='assets/dog17_1.png',
        prompt1 = "a cute dog",
        prompt2 = "a cute dog",
        uncond_prompt = (""),
        neg_prompt = ('A doubling image, unrealistic, artifacts, distortions, unnatural blending, ghosting effects,\
                overlapping edges, harsh transitions, motion blur, poor resolution, low detail'),
        noise_level = 0.6,
        alpha = 0.002,
        guidance_scale = 1,
        use_neg_cfg = False,
        output_start_images = False,
        num_output_imgs = num_output_imgs,
        use_pu_sampling = False,
        grad_analysis_out = False,
        output_interval = -1,
        output_separate_images = False,
        bvp_opt_args = {
            "opt_max_iter": 400,
            "opt_lr": 0.1,
            "lr_scheduler": 'linear',
            "lr_divide": True,
        },
        text_inv_args = {
            "text_inv_lr" : 5e-3,
            "text_inv_bsz" : 2,
            "text_inv_path" : 'text_inversion_checkpoints/sd2-1/',
            "text_inv_steps" : 500
        },
        grad_args = {
            "grad_batch_size": 10,
            "grad_sample_range": 100,
            "grad_guidance_0": 1.0,
            "grad_guidance_1": 1.0
        },
        bisection_args = {
            "max_strength": 4,
            "bisect_interval": 100,
            "only_new_points": False,
        },
        semantic_edit_args = {
            "image_idx": 2,
            "op": 'mid',
            "vis_num": 2,
            "vis_num_pc": 2,
            "pca_rank": 3,
            "edit_prompt": ("big ears"),
            "x_guidance_step": 64,
            "x_guidance_strength": 0.3
        }
    )
    interpolation.init()
    interpolation.optimise()

if __name__ == "__main__":
    main()
