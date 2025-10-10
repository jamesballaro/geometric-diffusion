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

from semantic_utils import get_h, local_encoder_pullback_zt

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

        self.init_semantic_unet()

    def set_seed(self, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        self.generator = generator

    def set_resolution(self, res):
        self.resolution = (res,res)

    def init_semantic_unet(self,):
        # monkey patch (basic method)
        self.unet.get_h                        = types.MethodType(get_h, self.unet)
        self.unet.local_encoder_pullback_zt    = types.MethodType(local_encoder_pullback_zt, self.unet)
        self.unet.set_attn_processor(AttnProcessor())

    @torch.no_grad()
    def run_edit_local_encoder_pullback_zt(
            self,
            config,
            latent_t,
            latent_T,
            idx,
            backward_fn=None,
            output_dir=None,
            vis_vT=False,
        ):
        block_idx = 0
        num_steps = 100
        seed = self.generator.initial_seed()

        print(
            f"current experiment: idx: {idx}, "
            f"op: {config.semantic_edit_args['op']}, "
            f"block_idx: {block_idx}, "
            f"vis_num: {config.semantic_edit_args['vis_num']}, "
            f"vis_num_pc: {config.semantic_edit_args['vis_num_pc']}, "
            f"pca_rank: {config.semantic_edit_args['pca_rank']}, 
            f"edit_prompt: {config.semantic_edit_args['edit_prompt']}, "
            f"x_guidance_steps: {config.semantic_edit_args['x_guidance_step']}"
        )

        # set edit prompt
        if  config.semantic_edit_args['edit_prompt']is not None:
            self.edit_prompt_emb = self.encode_prompt(config.semantic_edit_args['edit_prompt'])[0]

        # get local basis
        local_basis_name = f'local_basis-_{idx}-{noise_level}T-"{self.edit_prompt}"-{op}-block_{block_idx}-seed_{seed}'


        save_dir = f'./inputs/local_encoder_pullback_stable_diffusion-dataset_-num_steps_{num_steps}-pca_rank_{pca_rank}'
        os.makedirs(save_dir, exist_ok=True)

        u_path = os.path.join(save_dir, 'u-' + local_basis_name + '.pt')
        s_path = os.path.join(save_dir, 's-' + local_basis_name + '.pt')
        vT_path = os.path.join(save_dir, 'vT-' + local_basis_name + '.pt')

        # self.scheduler.set_timesteps(num_steps)

        t = self.get_timesteps(config.noise_level, return_single=True)

        # load pre-computed local basis
        if os.path.exists(u_path) and os.path.exists(vT_path):
            u = torch.load(u_path, map_location=self.device).type(self.dtype)
            vT = torch.load(vT_path, map_location=self.device).type(self.dtype)

        # computed local basis
        else:
            print('Run local pullback')
            u, s, vT = self.unet.local_encoder_pullback_zt(
                sample=latent_t, 
                timestep=t, 
                encoder_hidden_states=self.edit_prompt_emb, 
                config.semantic_edit_args['op'],
                block_idx=block_idx,
                pca_rank=config.semantic_edit_args['pca_rank'], 
                chunk_size=5, 
                min_iter=10, 
                max_iter=50, 
                convergence_threshold=1e-4,
            )

            vT = vT.to(device=self.device, dtype=self.dtype)

            # save semantic direction in h-space
            torch.save(u, u_path)
            torch.save(s, s_path)
            torch.save(vT, vT_path)

            # visualize vT (using pca basis)
            pca_vT = vT.view(-1, *latent_T.shape[1:]).permute(0, 2, 3, 1)
            pca_vT = pca_vT.reshape(-1, 4)
            _, _, pca_basis = torch.pca_lowrank(pca_vT, q=3, center=True, niter=2)
            vis_vT = einsum(vT.view(-1, *latent_T.shape[1:]), pca_basis, 'b c w h, c p -> b p w h')

            vis_vT = vis_vT - vis_vT.min()
            vis_vT = vis_vT / vis_vT.max()

            pca_vis_dir = os.path.join('obs')
            os.makedirs(pca_vis_dir, exist_ok=True)

            pca_vis_path = os.path.join(pca_vis_dir, f'vT-{local_basis_name}.png')

            tvu.save_image(vis_vT, pca_vis_path)

            del pca_vT, vis_vT, pca_basis
            torch.cuda.empty_cache()

        u = u / u.norm(dim=0, keepdim=True)
        vT = vT / vT.norm(dim=1, keepdim=True)

        original_latent_t = latent_t.clone()
        denoised_edited_latents = []

        for pc_idx in range(vis_num_pc):
            latent_dir = []
            dir_count = 0
            for direction in [1, -1]: # +v, -v

                vk = direction*vT[pc_idx, :].view(-1, *latent_T.shape[1:])

                # edit latent_t along vk direction with **x-space guidance**
                latent_t_list = [original_latent_t.clone()]
                for _ in tqdm(range(x_guidance_step), desc=f'x_space_guidance edit, pc_idx: {pc_idx}, dir: {dir_count}'):
                    latent_t_edit = self.x_space_guidance(
                        latent_t_list[-1], t=t, vk=vk,
                        single_edit_step=1,
                        use_edit_prompt=True,
                    )
                    latent_t_list.append(latent_t_edit)
                latent_t = torch.cat(latent_t_list, dim=0)
                latent_t = latent_t[::(latent_t.size(0) // vis_num)]

                if backward_fn is not None:
                    for i, latent in enumerate(latent_t):
                        print(f"\tx_guidance denoising latent {i+1}/{latent_t.size(0)}")
                        denoised = backward_fn(noise_level=config.noise_level, latent=latent.unsqueeze(0))
                        latent_dir.append(denoised)
                        print("")
                else:
                    raise RuntimeError("No DDIM backward function supplied")
                dir_count += 1

            denoised_edited_latents.append(latent_dir) # [pca direction 1 [+v, -v], pca direction 2[+v,-v], pca direction 3[+v,-v]]

        return denoised_edited_latents

    @torch.no_grad()
    def run_encoder_pullback_image_latent(self,
        config,
        latent_t,
        uncond_prompt_embed,
        neg_prompt_embed,
        interp_prompt,
        semantic_edit_args,
        edit_prompt,
        output_dir=None,
        ):

        full_noise_level = 1
        start_timestep =  self.get_timesteps(config.noise_level, return_single=True) # e.g 400

        # latents are noised to t = 0.6 * T
        # we find fully denoised latents with deterministic ddim_forward ( t = 1.0 *T)
        latent_T = self.ddim_forward(
            full_noise_level,
            config.guidance_scale,
            latent_t,
            uncond_prompt_embed,
            interp_prompt,
            neg_prompt_embed,
            config.use_neg_cfg,
            start_timestep,
        )

        ddim_backward_fn = functools.partial(
            self.ddim_backward,
            guidance_scale=config.guidance_scale,
            neg_prompt_embed=neg_prompt_embed,
            uncond_prompt_embed=uncond_prompt_embed,
            prompt_embed=interp_prompt,
            eta=0.0,
            use_neg_cfg=config.use_neg_cfg,
        )

        torch.cuda.empty_cache()
        denoised_edited_latents = self.run_edit_local_encoder_pullback_zt(
            config,
            latent_t,
            latent_T,
            0,
            backward_fn=ddim_backward_fn,
            output_dir=output_dir
        )
        return denoised_edited_latents


    @torch.no_grad()
    def x_space_guidance(self, zt, t, vk, single_edit_step, use_edit_prompt=False):
        # edit zt with vk
        zt_edit = zt + single_edit_step * vk

        # predict the noise residual
        et = self.unet(
            torch.cat([zt, zt_edit], dim=0), t,
            encoder_hidden_states=self.edit_prompt_emb.repeat(2, 1, 1)
            # cross_attention_kwargs=None,
        ).sample

        # DDS regularization
        et_null, et_edit = et.chunk(2)
        zt_edit = zt + (et_edit - et_null) * 2 # * x_space_guidance_scale
        return zt_edit

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
            t = torch.randint(self.scheduler.config.num_train_timesteps, (text_inv_bsz,), device=self.unet.device)

            # Add noise to the latent according to timestep (FWD)
            lat_t = self.scheduler.add_noise(lat, noise, t)

            # Predict the noise added to the latent
            model_pred = self.unet(lat_t, t, encoder_hidden_states=prompt_embed_opt.repeat(text_inv_bsz, 1,1)).sample

            # Calc loss and propagate back
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction='mean')
            # accelerator.backward(loss)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        prompt_embed_opt.requires_grad_(False)
        return prompt_embed_opt

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
            if guidance_scale>0:
                if use_neg_cfg:
                    prompt_embed_cfg = torch.cat([uncond_prompt_embed, prompt_embed-neg_prompt_embed])
                else:
                    prompt_embed_cfg = torch.cat([uncond_prompt_embed, prompt_embed])
            else:
                prompt_embed_cfg = uncond_prompt_embed
            assert (guidance_scale > 0 and prompt_embed_cfg.shape[0] == 2*latent.shape[0]) \
            or (guidance_scale == 0 and prompt_embed_cfg.shape[0] == latent.shape[0])

            time_steps = self.get_timesteps(noise_level)
            extra_step_kwargs = self.prepare_extra_step_kwargs(self.generator, eta=eta)

            for t in tqdm(time_steps):
                if end_timestep and t <= end_timestep:
                    break

                if guidance_scale > 0:
                    noise_pred = self.noise_pred_cfg(latent, t, prompt_embed_cfg, guidance_scale=guidance_scale)
                else:
                    noise_pred = self.noise_pred(latent, t, prompt_embed)

                latent = self.scheduler.step(noise_pred, t, latent, **extra_step_kwargs).prev_sample

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
            time_steps = self.get_timesteps(noise_level)
            time_steps = reversed(time_steps) # [1, . . . , 400]

            if start_timestep:
                time_steps = time_steps[start_timestep-1:]
                print("start:",start_timestep, time_steps[:start_timestep-1])


            for t in tqdm(time_steps):
                # Previous timestep index
                t_prev = max((t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps), 0)

                # Cumulative alpha for equations
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[t_prev]
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                # Predict noise for latent at time t

                if guidance_scale > 0:
                    prompt_embed_cfg = torch.cat([uncond_prompt_embed, prompt_embed])
                    if use_neg_cfg:
                        prompt_embed_cfg = torch.cat([uncond_prompt_embed, prompt_embed-neg_prompt_embed])

                    epsilon = self.noise_pred_cfg(latent, t, prompt_embed_cfg, guidance_scale=guidance_scale)

                else:  # if no classifier free guidance, save half of the computation
                    epsilon = self.noise_pred(latent, t, uncond_prompt_embed)

                # We invert the prediction to estimate x_0 from the PREV timestep
                x0 = (latent -  (1 - alpha_prod_t_prev)**0.5 * epsilon) / (alpha_prod_t_prev**0.5)

                # Add the unet's own noise pred to the clean latent to produce a noised version at THIS timestep
                latent = alpha_prod_t**0.5 * x0 + (1- alpha_prod_t)**0.5*epsilon

        # print("\nlatent stats:", latent.min().item(), latent.max().item(),latent.mean().item())
        return latent


