import gc
import os
import math
import json
import types
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as tvu

import numpy as np
from einops import rearrange, einsum

from latents import LatentProcessor

class SemanticEditor():
    def __init__(self, generator, pipe, state, config):
        self.generator = generator
        self.pipe = pipe
        self.latent_proc = LatentProcessor(self.pipe, self.generator.device)
        self.state = state
        self.config = config

    # -----Semantic Editing
    def calculate_semantic_edit_latent(self, latent):
        """
        This function uses the local encoder pullback to generate a latent space image sequence
        """

        latent_dim = int(self.config.resolution[0] / 8)
        # latents are noised to t = 0.6 * T
        latent = latent.reshape(-1, 4, latent_dim, latent_dim)

        edit_prompt = self.config.semantic_edit_args["edit_prompt"]
        edit_prompt_embed = self.pipe.encode_prompt(edit_prompt)[1].to(self.device) #TODO device

        interp_prompt = self.state.spline.lerp(self.state.timesteps_out, self.state.prompt_embed_opt1, self.state.prompt_embed_opt2)

        # Help with reverse CFG for semantic editing
        interp_prompt = interp_prompt + edit_prompt_embed.expand_as(interp_prompt)

        denoised_edited_latents = self.editor.run_encoder_pullback_image_latent(
            self.config,
            latent,
            self.state.uncond_prompt_embed,
            self.state.neg_prompt_embed,
            interp_prompt[self.state.edit_idx:self.state.edit_idx+1,:,:],
            edit_prompt,
            output_dir=self.output_dir,
        )

        return denoised_edited_latents
    
    def calculate_semantic_edit_input_image(self, image_path, select):
        """
        This function circumvents the optimsation and uses algorithm 2 to semantically edit the input image
        """
        
        image_tensor = self.pipe.preprocess_image(image_path).to(self.config.device)
        image_latent = self.pipe.encode_image(image_tensor)

        full_noise_level = 1
        end_timestep =  self.pipe.get_timesteps(self.config.noise_level, return_single=True) # e.g 400


        if select == 1:
            prompt_embed = self.state.prompt_embed_opt1
        else:
            prompt_embed = self.state.prompt_embed_opt2

        edit_prompt = self.config.semantic_edit_args["edit_prompt"]
        edit_prompt_embed = self.pipe.encode_prompt(edit_prompt)[1].to(self.config.device)

        interp_prompt = prompt_embed + edit_prompt

        noised_latent_T = self.editor.latent_proc.ddim_forward(
            full_noise_level,
            self.config.guidance_scale,
            image_latent,
            self.state.uncond_prompt_embed,
            prompt_embed,
            self.state.neg_prompt_embed,
            self.config.use_neg_cfg
        )

        ddim_backward_fn = functools.partial(
            self.editor.latent_proc.ddim_backward,
            guidance_scale=self.config.guidance_scale,
            neg_prompt_embed=self.state.neg_prompt_embed,
            uncond_prompt_embed=self.state.uncond_prompt_embed,
            prompt_embed=prompt_embed,
            eta=0.0,
            use_neg_cfg=self.config.use_neg_cfg,
        )

        noised_latent_t = ddim_backward_fn(
            noise_level=1,
            latent=noised_latent_T,
            end_timestep=end_timestep,
        )

        denoised_edited_latents = self.editor.run_edit_local_encoder_pullback_zt(
            self.config
            noised_latent_t,
            noised_latent_T,
            0,
            backward_fn=ddim_backward_fn,
            output_dir=self.config.output_dir
        )

        return denoised_edited_latents

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
            f"block_idx: {block_idx}, "
            f"vis_num: {config.semantic_edit_args['vis_num']}, "
            f"vis_num_pc: {config.semantic_edit_args['vis_num_pc']}, "
            f"pca_rank: {config.semantic_edit_args['pca_rank']},
            f"edit_prompt: {config.semantic_edit_args['edit_prompt']}, "
            f"x_guidance_steps: {config.semantic_edit_args['x_guidance_step']}"
        )

        edit_prompt = config.semantic_edit_args['edit_prompt']
        # set edit prompt
        if  edit_prompt is not None:
            self.edit_prompt_emb = self.pipe.encode_prompt(edit_prompt)[0]

        # get local basis
        local_basis_name = f'local_basis-_{idx}-{noise_level}T-"{edit_prompt}"-{op}-block_{block_idx}-seed_{seed}'


        save_dir = f'./inputs/local_encoder_pullback_stable_diffusion-dataset_-num_steps_{num_steps}-pca_rank_{pca_rank}'
        os.makedirs(save_dir, exist_ok=True)

        u_path = os.path.join(save_dir, 'u-' + local_basis_name + '.pt')
        s_path = os.path.join(save_dir, 's-' + local_basis_name + '.pt')
        vT_path = os.path.join(save_dir, 'vT-' + local_basis_name + '.pt')

        # self.scheduler.set_timesteps(num_steps)

        t = self.pipe.get_timesteps(config.noise_level, return_single=True)

        # load pre-computed local basis
        if os.path.exists(u_path) and os.path.exists(vT_path):
            u = torch.load(u_path, map_location=self.device).type(self.dtype)
            vT = torch.load(vT_path, map_location=self.device).type(self.dtype)

        # computed local basis
        else:
            print('Run local pullback')
            u, s, vT = self.pipe.unet.local_encoder_pullback_zt(
                sample=latent_t,
                timestep=t,
                encoder_hidden_states=self.edit_prompt_emb,
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
        start_timestep =  self.pipe.get_timesteps(config.noise_level, return_single=True) # e.g 400


        # latents are noised to t = 0.6 * T
        # we find fully denoised latents with deterministic ddim_forward ( t = 1.0 *T)
        latent_T = self.latent_proc.ddim_forward(
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
            self.latent_proc.ddim_backward,
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
        et = self.pipe.unet(
            torch.cat([zt, zt_edit], dim=0), t,
            encoder_hidden_states=self.edit_prompt_emb.repeat(2, 1, 1)
            # cross_attention_kwargs=None,
        ).sample

        # DDS regularization
        et_null, et_edit = et.chunk(2)
        zt_edit = zt + (et_edit - et_null) * 2 # * x_space_guidance_scale
        return zt_edit

    def get_h(
            self, sample=None, timestep=None, encoder_hidden_states=None,
            block_idx=None, verbose=False,
        ):
        '''
        Args
            - sample : zt
            - block_idx :op == mid : [0]
        Returns
            - h : hidden feature
        '''
        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # pre-process
        sample = self.conv_in(sample)

        # mid
        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
        if (op == 'mid') & (block_idx == 0):
            if verbose:
                print(f'op : {op}, block_idx : {block_idx}, return h.shape : {sample.shape}')
            return sample


        raise ValueError(f'(op, block_idx) = ({op, block_idx}) is not valid')

# monkey patch (local method)
    def local_encoder_pullback_zt(
            self, sample, timestep, encoder_hidden_states=None, block_idx=None,
            pca_rank=50, chunk_size=25, min_iter=10, max_iter=100, convergence_threshold=1e-3,
        ):
        '''
        Args
            - sample : zt
            - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
            - pooling : ['pixel-sum', 'channel-sum', 'single-channel', 'multiple-channel']
        Returns
            - h : hidden feature
        '''
        # get h samples
        time_s = time.time()

        # necessary variables
        h_shape = self.get_h(
            sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states,
            block_idx=block_idx,
        ).shape

        c_i, w_i, h_i = sample.size(1), sample.size(2), sample.size(3)
        c_o, w_o, h_o = h_shape[1], h_shape[2], h_shape[3]

        a = torch.tensor(0., device=sample.device, dtype=sample.dtype)

        # Algorithm 1
        vT = torch.randn(c_i*w_i*h_i, pca_rank, device=sample.device, dtype=torch.float)
        vT, _ = torch.linalg.qr(vT)
        v = vT.T
        v = v.view(-1, c_i, w_i, h_i)

        time_s = time.time()
        for i in range(max_iter):
            v = v.to(device=sample.device, dtype=sample.dtype)
            v_prev = v.detach().cpu().clone()

            u = []
            if v.size(0) // chunk_size != 0:
                v_buffer = list(v.chunk(v.size(0) // chunk_size))
            else:
                v_buffer = [v]

            for vi in v_buffer:
                g = lambda a : self.get_h(
                    sample + a*vi, timestep=timestep, encoder_hidden_states=encoder_hidden_states.repeat(vi.size(0), 1, 1),
                    block_idx=block_idx,
                )

                ui = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a)
                u.append(ui.detach().cpu().clone())
            u = torch.cat(u, dim=0)
            u = u.to(sample.device, sample.dtype)

            g = lambda sample : einsum(
                u, self.get_h(
                    sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states,
                    block_idx=block_idx,
                ), 'b c w h, i c w h -> b'
            )
            v_ = torch.autograd.functional.jacobian(g, sample)
            v_ = v_.view(-1, c_i*w_i*h_i)

            _, s, v = torch.linalg.svd(v_, full_matrices=False)
            v = v.view(-1, c_i, w_i, h_i)
            u = u.view(-1, c_o, w_o, h_o)

            convergence = torch.dist(v_prev, v.detach().cpu()).item()
            print(f'power method : {i}-th step convergence : ', convergence)

            if torch.allclose(v_prev, v.detach().cpu(), atol=convergence_threshold) and (i > min_iter):
                print('reach convergence threshold : ', convergence)
                break

        u, s, vT = u.view(-1, c_o*w_o*h_o).T.detach(), s.sqrt().detach(), v.view(-1, c_i*w_i*h_i).detach()

        time_e = time.time()
        print('power method runtime ==', time_e - time_s)

        return u, s, vT
