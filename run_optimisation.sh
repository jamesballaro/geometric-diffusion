!/bin/bash

ARGS = (
    "--path_to_pretrained_model", "stabilityai/stable-diffusion-2-1-base",
    "--output_dir", "results/",
    "--num_inference_steps", 50,
    "--num_output_imgs", 7,
    "--guidance_scale", 1,
    "--strength", 0.002,
    "--high_noise_frac", 0.6,
)

python src/main.py ${ARGS[@]}
