from .bvp_structs import BVPConfig, BVPState
from .bvp_output import BVP_OutputModule
from ..grad.geodesic import o_project_batch

import torch
import torch.nn as nn
import torch.nn.functional as F

class BVP_GradientModule():
    def __init__(self, bvp_io_unit, state: BVPState, config: BVPConfig):
        self.bvp_io_unit = bvp_io_unit
        self.config = config
        self.state = state

    # Main functions
    def __call__(self, X, V, A, t_opt) :
        # Latent downsizing factor for SD2.1 = 8
        latent_dim = int(self.config.resolution / 8)

        latents = X.reshape(-1, 4, latent_dim, latent_dim)

        # Linear interpolation of text prompt-embeddings
        prompt_embed = self.state.spline.lerp(t_opt, self.state.prompt_embed_opt1, self.state.prompt_embed_opt2)

        # Compute score functions ( ∇ log p) from latent and prompt embedding
        scores = self.state.score_unit.grad_compute_batch(latents, prompt_embed)

        B, C, H, W = scores.shape

        # Flatten the scores
        scores = scores.reshape(B,-1)

        # This is the (I - ŷŷ) in the functional derivative which ensures that only the normal component of the score function affects the geodesic
        if self.config.project_to_sphere:
            scores = o_project_batch(scores, X)
            A = o_project_batch(A, X)

        # Compute the scaled acceleration term
        V_norm2 = torch.sum(V * V, dim=-1)
        A_scaled = A / V_norm2[:,None]

        # Project back to the hypersphere, these are the inner terms in the FuncDeriv
        term1 = o_project_batch(scores, V)
        term2 = o_project_batch(A_scaled, V) * (1/self.config.alpha)

        # Assemble the functional derivative
        grad_all = -(term1 + term2)

        # Grad analysis
        mean_all_norm, mean_norm1, mean_norm2, mean_angle = self.bvp_io_unit.grad_analysis(B, t_opt, self.state.iter, term1, term2, grad_all)

        if mean_norm1 < mean_norm2:
            # This is a heuristic, if the acceleration term too big, it will go to the wrong direction
            # Setting the learning rate to be super small can avoid this issue
            return None, mean_all_norm, mean_angle

        return grad_all, mean_all_norm, mean_angle
