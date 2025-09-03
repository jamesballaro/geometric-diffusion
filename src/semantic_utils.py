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

def get_h(
        self, sample=None, timestep=None, encoder_hidden_states=None, 
        op=None, block_idx=None, verbose=False,
    ):
    '''
    Args
        - sample : zt
        - op : ['down', 'mid', 'up']
        - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
    Returns
        - h : hidden feature
    '''
    # time
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        # This would be a good case for the `match` statement (Python 3.10+)
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
    # down
    down_block_res_samples = (sample,)
    for down_block_idx, downsample_block in enumerate(self.down_blocks):
        if (op == 'down') & (block_idx == down_block_idx):
            sample, res_samples, h_space = down_block_forward(
                downsample_block, hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states, timesteps=timestep, uk=None, after_sa=True,
            )
            return h_space
        
        elif hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(
                hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

        if (op == 'down') & (block_idx == down_block_idx):
            return sample

        down_block_res_samples += res_samples

    # mid
    sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
    if (op == 'mid') & (block_idx == 0):
        if verbose:
            print(f'op : {op}, block_idx : {block_idx}, return h.shape : {sample.shape}')
        return sample
    
    # up
    for up_block_idx, upsample_block in enumerate(self.up_blocks):
        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample = upsample_block(
                hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=None,
            )

        if verbose:
            print(f'down_block_idx : {down_block_idx}, sample.shape : {sample.shape}')

        # return h
        if (op == 'up') & (block_idx == up_block_idx):
            if verbose:
                print(f'op : {op}, block_idx : {block_idx}, return h.shape : {sample.shape}')
            return sample

    raise ValueError(f'(op, block_idx) = ({op, block_idx}) is not valid')

# monkey patch (local method)
def local_encoder_pullback_zt(
        self, sample, timestep, encoder_hidden_states=None, op=None, block_idx=None,
        pca_rank=50, chunk_size=25, min_iter=10, max_iter=100, convergence_threshold=1e-3,
    ):
    '''
    Args
        - sample : zt
        - op : ['down', 'mid', 'up']
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
        op=op, block_idx=block_idx, 
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
                op=op, block_idx=block_idx, 
            )

            ui = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a)
            u.append(ui.detach().cpu().clone())
        u = torch.cat(u, dim=0)
        u = u.to(sample.device, sample.dtype)

        g = lambda sample : einsum(
            u, self.get_h(
                sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states,
                op=op, block_idx=block_idx, 
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

# non-monkey patch (basic method)
def down_block_forward(down_block, hidden_states, temb, encoder_hidden_states, timestep, uk=None, after_res=False, after_sa=False):
    assert after_res != after_sa
    
    output_states = ()

    for resnet, attn in zip(down_block.resnets, down_block.attentions):
        hidden_states = resnet(hidden_states, temb)

        if after_res:
            h_space = hidden_states.clone()
            
            if uk is not None:
                hidden_states = hidden_states + uk.view(-1, *hidden_states.shape[1:])
            
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=None,
            ).sample

        elif after_sa:
            # 1. Input
            if attn.is_input_continuous:
                batch, _, height, width = hidden_states.shape
                residual = hidden_states

                hidden_states = attn.norm(hidden_states)
                if not attn.use_linear_projection:
                    hidden_states = attn.proj_in(hidden_states)
                    inner_dim = hidden_states.shape[1]
                    hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                else:
                    inner_dim = hidden_states.shape[1]
                    hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                    hidden_states = attn.proj_in(hidden_states)
            elif attn.is_input_vectorized:
                hidden_states = attn.latent_image_embedding(hidden_states)
            elif attn.is_input_patches:
                hidden_states = attn.pos_embed(hidden_states)

            # 2. Blocks
            for block_idx, block in enumerate(attn.transformer_blocks):
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    cross_attention_kwargs=None,
                    class_labels=None,
                )

                if block_idx == 1:
                    h_space = hidden_states.clone()
                    
                    if uk is not None:
                        hidden_states = hidden_states + uk.view(-1, *hidden_states.shape[1:])

            # 3. Output
            if attn.is_input_continuous:
                if not attn.use_linear_projection:
                    hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                    hidden_states = attn.proj_out(hidden_states)
                else:
                    hidden_states = attn.proj_out(hidden_states)
                    hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

                output = hidden_states + residual
            elif attn.is_input_vectorized:
                raise ValueError()
            elif attn.is_input_patches:
                raise ValueError()
            hidden_states = output
            
        output_states += (hidden_states,)

    if down_block.downsamplers is not None:
        for downsampler in down_block.downsamplers:
            hidden_states = downsampler(hidden_states)

        output_states += (hidden_states,)
    
    return hidden_states, output_states, h_space

