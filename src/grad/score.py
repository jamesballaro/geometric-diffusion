import torch

from .geodesic import SphericalCubicSpline, BisectionSampler
from .geodesic import norm_fix, norm_fix_batch, o_project, o_project_batch

class ScoreProcessor():
    def __init__(self, pipe, config, state):

        self.pipe = pipe
        self.device = config.device
        self.state = state
        self.grad_batch_size = config.grad_args['grad_batch_size']
        self.grad_sample_range = config.grad_args['grad_sample_range']
        #self.uncond_prompt_embed = state.uncond_prompt_embed
        #self.neg_prompt_embed = state.neg_prompt_embed
        self.grad_guidance_0 = config.grad_args['grad_guidance_0']
        self.grad_guidance_1 = config.grad_args['grad_guidance_1']
        self.time_step = pipe.get_timesteps(config.noise_level, return_single=True)

    def grad_prepare(self, latent):
        range_t = self.grad_sample_range

        # We get this from the get_t in pipe (t from noise level)
        max_t = self.time_step + range_t
        min_t = self.time_step - range_t

        # Random sample timestep
        rand_t = torch.randint(min_t, max_t, (1,), device=self.device)

        # DDIM ODE
        alpha_t_ = self.pipe.scheduler.alphas_cumprod[self.time_step]
        noise = torch.randn_like(latent)

        if self.time_step == 0:
            clean_latent = latent
        else:
            clean_latent = (latent - (1-alpha_t_)**0.5 * noise) / (alpha_t_**0.5)

        noisy_latent = self.pipe.scheduler.add_noise(clean_latent, noise, rand_t)

        return noisy_latent, rand_t

    def grad_compute(self, latent, prompt_embed):

        assert latent.shape[0] == prompt_embed.shape[0]
        batch_size = latent.shape[0]

        #broadcast the conditions
        uncond_prompt_embed = self.state.uncond_prompt_embed.repeat(batch_size, 1, 1)
        neg_prompt_embed = self.state.neg_prompt_embed.repeat(batch_size, 1, 1)

        latent, t = self.grad_prepare(latent)

        grad_c, grad_d = 0, 0

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # Save computation (ep_uncond cancels out if guidance coeffs are equal)
            if self.grad_guidance_0 == self.grad_guidance_1:
                grad_c = -self.pipe.noise_pred(latent, t, prompt_embed)
                grad_d = self.pipe.noise_pred(latent, t, neg_prompt_embed)
            else:
                ep_uncond = self.pipe.noise_pred(latent, t, uncond_prompt_embed)

                if self.grad_guidance_0 > 0:
                    ep_cond = self.pipe.noise_pred(latent, t, prompt_embed)
                    grad_c = ep_uncond - ep_cond

                if self.grad_guidance_1 > 0:
                    ep_neg = self.pipe.noise_pred(latent, t, neg_prompt_embed)
                    grad_d = ep_neg - ep_uncond

        norm_constant = 1/(abs(self.grad_guidance_0)+ abs(self.grad_guidance_1))
        grad = norm_constant * (self.grad_guidance_0*grad_c + self.grad_guidance_1*grad_d)
        return grad

    def grad_compute_batch(self, latents, prompt_embed):
        assert latents.shape[0] == prompt_embed.shape[0]

        batch_size = latents.shape[0]
        grad_out = None

        batch_idx = 0

        while batch_size > 0:
            #Specific batch size for this iteration (ensures we dont take more batches than there are left)
            sub_bsz = min(batch_size, self.grad_batch_size)

            lats = latents[batch_idx:batch_idx + sub_bsz, :,:,:]

            grad = self.grad_compute(lats, prompt_embed[batch_idx:batch_idx + sub_bsz,:,:])

            grad_out = grad if grad_out is None else torch.cat([grad_out, grad])

            batch_size -= sub_bsz
            batch_idx += sub_bsz

        return grad_out
