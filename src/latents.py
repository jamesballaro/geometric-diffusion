
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

class LatentProcessor():
    def __init__(self, pipe, device):
        self.pipe = pipe
        self.device = device
        self.text_inverter = TextInverter(pipe, device)

    def ddim_backward(self,
        noise_level,
        guidance_scale,
        latent,
        neg_prompt_embed,
        uncond_prompt_embed,
        prompt_embed,
        eta=0.0,
        use_neg_cfg=True,
        end_timestep=None,
        ):

        if noise_level > 0:

            time_steps = self.pipe.get_timesteps(noise_level)
            #TODO: extra_step_kwargs 
            extra_step_kwargs = self.prepare_extra_step_kwargs(self.generator, eta=eta)

            if guidance_scale>0:
                if use_neg_cfg:
                    prompt_embed_cfg = torch.cat([uncond_prompt_embed, prompt_embed-neg_prompt_embed])
                else:
                    prompt_embed_cfg = torch.cat([uncond_prompt_embed, prompt_embed])
            else:
                prompt_embed_cfg = uncond_prompt_embed

            assert (guidance_scale > 0 and prompt_embed_cfg.shape[0] == 2*latent.shape[0]) \
            or (guidance_scale == 0 and prompt_embed_cfg.shape[0] == latent.shape[0])


            for t in tqdm(time_steps):
                if end_timestep and t <= end_timestep:
                    break

                if guidance_scale > 0:
                    noise_pred = self.pipe.noise_pred_cfg(latent, t, prompt_embed_cfg, guidance_scale=guidance_scale)
                else:
                    noise_pred = self.pipe.noise_pred(latent, t, prompt_embed)

                latent = self.pipe.scheduler.step(noise_pred, t, latent, **extra_step_kwargs).prev_sample

        return latent

    def ddim_forward(self,
        noise_level,
        guidance_scale,
        latent,
        uncond_prompt_embed,
        prompt_embed,
        neg_prompt_embed,
        use_neg_cfg,
        start_timestep=None):
        """
        This function carries out the DDIM forward process, using the UNet's own noise prediction
        instead of Gaussian noise to ensure reversability (and accurate score prediction).

        Takes in a clean latent and prompt embedding and outputs the correctly noised latent for
        the noise_level.
        """
        if noise_level > 0: # Noise level (how far to sample) = t/T

            # Get the time step for the noise level e.g [400, . . . ,1]
            time_steps = self.pipe.get_timesteps(noise_level)
            time_steps = reversed(time_steps) # [1, . . . , 400]

            if start_timestep:
                time_steps = time_steps[start_timestep-1:]
                print("start:",start_timestep, time_steps[:start_timestep-1])

            for t in tqdm(time_steps):
                # Previous timestep index
                t_prev = max((t - self.pipe.scheduler.config.num_train_timesteps // self.pipe.scheduler.num_inference_steps), 0)

                # Cumulative alpha for equations
                alpha_prod_t_prev = self.pipe.scheduler.alphas_cumprod[t_prev]
                alpha_prod_t = self.pipe.scheduler.alphas_cumprod[t]
                # Predict noise for latent at time t

                if guidance_scale > 0:
                    prompt_embed_cfg = torch.cat([uncond_prompt_embed, prompt_embed])
                    if use_neg_cfg:
                        prompt_embed_cfg = torch.cat([uncond_prompt_embed, prompt_embed-neg_prompt_embed])

                    epsilon = self.pipe.noise_pred_cfg(latent, t, prompt_embed_cfg, guidance_scale=guidance_scale)

                else:  # if no classifier free guidance, save half of the computation
                    epsilon = self.pipe.noise_pred(latent, t, uncond_prompt_embed)

                # We invert the prediction to estimate x_0 from the PREV timestep
                x0 = (latent -  (1 - alpha_prod_t_prev)**0.5 * epsilon) / (alpha_prod_t_prev**0.5)

                # Add the unet's own noise pred to the clean latent to produce a noised version at THIS timestep
                latent = alpha_prod_t**0.5 * x0 + (1- alpha_prod_t)**0.5*epsilon

        return latent

class TextInverter():
    def __init__(self, pipe, device):
        self.pipe = pipe
        self.device = device

    def load_text_inversion(self,
        prompt_embed,
        image_latent,
        prefix,
        postfix='',
        text_inv_steps=None,
        text_inv_path=None,
        text_inv_lr=None,
        text_inv_bsz=None,
        **kwargs):
        """
        Look for checkpoints to load for the optimised text embedding
        """
        os.makedirs(text_inv_path, exist_ok=True)
        # Load the text inversion model
        checkpoint_path = '{}_{}_{}_{}.pt'.format(prefix, text_inv_steps, str(text_inv_lr).replace('.',''), postfix)
        checkpoint_path = os.path.join(text_inv_path, checkpoint_path)

        print(f"\nLooking for text inversion checkpoints at {checkpoint_path}")

        # If this test has been run already, load it
        if os.path.exists(checkpoint_path):
            prompt_embed = torch.load(checkpoint_path, weights_only=True).to(self.device)
            print(f"Text inversion checkpoints loaded successfully {checkpoint_path}")

        else:
            # If there is no path, run the text_inversion
            print(f"None found at {checkpoint_path} \nCreating new checkpoint.")
            prompt_embed = self.text_inversion(prompt_embed, image_latent, text_inv_steps, text_inv_lr, text_inv_bsz).to(self.device)
            torch.save(prompt_embed, checkpoint_path)
        return prompt_embed

    def text_inversion(self, prompt_embed, latent, text_inv_steps=None, text_inv_lr=None, text_inv_bsz=None):
        """
        This routine keeps the U-Net (and the latent sample) frozen and directly optimizes the
        *context embedding* that conditions the U-Net. On each step it samples a random diffusion
        timestep t, generates Gaussian noise ε, forms the noised latent x_t via the scheduler’s
        forward process q(x_t | x_0=latent, t), and updates the embedding so that the U-Net's
        prediction matches the ground-truth target. The result is a learned embedding tensor that
        “pulls” the model toward reconstructing the given latent when conditioned by this embedding.
        """

        latent.requires_grad_(False)
        prompt_embed_opt = prompt_embed.clone().requires_grad_(True)
        optimizer = torch.optim.AdamW([prompt_embed_opt], lr=text_inv_lr)

        print("\nOptimising latent conditional embedding:")
        print(f"\ttotal steps: {text_inv_steps}")
        print(f"\tlr: {text_inv_lr}")
        print(f"\tbatch_size: {text_inv_bsz}")

        progress_bar = tqdm(range(text_inv_steps), desc="Text Inversion Training")

        # Batch size defaults to 2
        if latent.shape[0]< text_inv_bsz:
            latent = latent.repeat(text_inv_bsz,1,1,1)

        # steps from config.yaml file
        for i in progress_bar:
            #Freeze optimizer
            optimizer.zero_grad()

            # Sample latent batch randomly
            indices = torch.randperm(latent.shape[0])[:text_inv_bsz]
            lat = latent[indices]

            # Noise the shape of the latent
            noise = torch.randn_like(lat)

            # Random int from 0 to max train steps (500)
            t = torch.randint(self.pipe.scheduler.config.num_train_timesteps, (text_inv_bsz,), device=self.pipe.unet.device)

            # Add noise to the latent according to timestep (FWD)
            lat_t = self.pipe.scheduler.add_noise(lat, noise, t)

            # Predict the noise added to the latent
            model_pred = self.pipe.unet(lat_t, t, encoder_hidden_states=prompt_embed_opt.repeat(text_inv_bsz, 1,1)).sample

            # Calc loss and propagate back
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction='mean')
            # accelerator.backward(loss)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        prompt_embed_opt.requires_grad_(False)
        return prompt_embed_opt
