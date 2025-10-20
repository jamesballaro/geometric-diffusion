import os
import time
import types
from typing import Tuple, Union, List

import functools
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as tvu

from tqdm import tqdm
from PIL import Image
from einops import rearrange, einsum

from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.models.attention_processor import AttnProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPFeatureExtractor,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

class CustomStableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True):

        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, image_encoder, requires_safety_checker)

        self.generator = torch.Generator(device=self.device)
        self.disable_xformers_memory_efficient_attention()

        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        # Latent downsizing factor for SD2.1 = 8
        if self.resolution:
            self.latent_dim = int(self.resolution[0] / 8)

    def set_seed(self, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        self.generator = generator

    def set_resolution(self, res):
        self.resolution = (res,res)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]]
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        prompt_tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        token_ids = prompt_tokens.input_ids
        prompt_embeds = self.text_encoder(
            token_ids.to(self.text_encoder.device),
        )[0]

        return prompt_embeds, prompt_embeds

    def preprocess_image(self, image_path) -> torch.Tensor:
        try:
            input_image = Image.open(image_path)
            print(f"Loaded existing {image_path}")
        except FileNotFoundError:
            print("Creating a placeholder 'input.jpg'. Please replace it with your own image.")
            input_image = Image.new('RGB', self.resolution, color='red')
            input_image.save("input.jpg")

        image = input_image.convert("RGB")
        image = image.resize(self.resolution)
        # image_tensor = torch.tensor(list(image.getdata()), dtype=self.vae.dtype).reshape(self.resolution[1], self.resolution[0], 3)
        image_tensor = (np.array(image).astype(np.float32) / 255.0) * 2.0 - 1.0
        image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).permute(0, 3, 1, 2)

        return image_tensor.clamp(-1,1)

    def encode_image(self, image_tensor):
        latents_dist = self.vae.encode(image_tensor)['latent_dist'].mean
        latents = latents_dist * self.vae.config.scaling_factor
        return latents

    def decode_latent(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image_tensor = self.vae.decode(latents)['sample']
        return image_tensor

    def noise_pred_cfg(self, latent, t, prompt_embed, guidance_scale=0.5):
        # model prediction with classifier free guidance
        latent_model_input = torch.cat([latent] * 2)
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embed).sample
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        return noise_pred

    def noise_pred(self, latent, t, prompt_embed):
        return self.unet(latent, t, encoder_hidden_states=prompt_embed).sample

    def get_timesteps(self, noise_level, return_single=False):
        # return the time step or steps for the given noise level

        if noise_level == 0:
            if return_single:
                return torch.tensor(0, device=self.device)
            return torch.tensor([], device=self.device)

        time_steps = self.scheduler.timesteps
        self.max_t = max(time_steps)

        #[500, .. , 1]
        time_stamp = max(int(len(time_steps) * noise_level), 1)


        #e.g noise_level =0.8, time_stamp = 400
        t = time_steps[-time_stamp]
        # t = 400

        if return_single:
            return t

        # return steps: [400, .. ,1]
        return time_steps[-time_stamp:]
