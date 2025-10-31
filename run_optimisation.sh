#!/bin/bash

ARGS=(
  "--path_to_pretrained_model" "/model/stable-diffusion-2-1-base"
  "--output_dir" "results/"
  "--resolution" 512
  "--num_inference_steps" 50
  "--num_output_imgs" 13
  "--guidance_scale" 1
  "--seed 67280421310721"
)

python -m src.main ${ARGS[@]}
