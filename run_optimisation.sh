!/bin/bash

ARGS = (
    "--path_to_pretrained_model", "stabilityai/stable-diffusion-2-1-base",
    "--prompt", "a cute dog",
    "--path_to_image", "assets/dog17_0.png",
    "--output_path", "results/dog_test.png",
    "--num_inference_steps", 50,
    "--guidance_scale", 1,
    "--strength", 0.002,
    "--high_noise_frac", 0.6,
)

python src/main.py ${ARGS[@]}
