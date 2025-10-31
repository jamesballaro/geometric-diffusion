import os
import torch
import torch.nn.functional as F
import functools

from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Any

from .bvp_structs import BVPConfig, BVPState
from .bvp_output import BVP_OutputModule
from .bvp_gradient import BVP_GradientModule

from ..grad.geodesic import SphericalCubicSpline, BisectionSampler
from ..grad.geodesic import norm_fix, norm_fix_batch, o_project, o_project_batch
from ..grad.score import ScoreProcessor
from ..image.image_io import ImageProcessor
from ..latent.semantic import SemanticEditor

class BVPOptimiser():
    def __init__(self, config, lr_scheduler='linear', lr_divide=True):
        self.opt_max_iter = config.bvp_opt_args['opt_max_iter']
        self.lr_init = config.bvp_opt_args['opt_lr']
        self.lr_scheduler = lr_scheduler
        self.lr_divide = lr_divide

    def get_learning_rate(self, it, t):
        if self.lr_scheduler == 'constant':
            return self.lr_init
        scale = it / self.opt_max_iter
        if self.lr_scheduler == 'linear':
            cur_lr =  self.lr_init * (1 - scale)
        elif self.lr_scheduler == 'cosine':
            cur_lr = self.lr_init * 0.5 *(1 + torch.cos(torch.pi * scale))
        elif self.lr_scheduler == 'polynomial':
            cur_lr = self.lr_init * (1 - scale)**2
        else:
            raise ValueError('lr_scheduler not recognized')
        if self.lr_divide:
            cur_lr = cur_lr / len(t)
        return cur_lr

class BVPAlgorithm():
    def __init__(self, pipe, config: BVPConfig):
        self.pipe = pipe
        self.config = config
        self.state = BVPState()

        # Copy all config attributes to self
        for key, value in config.__dict__.items():
            setattr(self, key, value)

        self.init_submodules()

    def init_submodules(self):
        # Initialize optimiser sub units
        self.sampler = BisectionSampler(self.device, **self.bisection_args)
        self.score_unit = ScoreProcessor(self.pipe, self.config, self.state)
        self.optimizer = BVPOptimiser(self.config)
        self.timesteps_out = torch.linspace(0, 1, self.num_output_imgs).to(self.device)
        self.editor = SemanticEditor(self.pipe, self.state, self.config)
        self.image_proc = ImageProcessor(self.pipe, self.config, self.state, self.editor)
        self.bvp_io_unit = BVP_OutputModule(self.pipe, self.editor, self.image_proc, self.state, self.config)
        self.gradient_unit = BVP_GradientModule(self.bvp_io_unit, self.state, self.config)

    # Main functions
    def step(self):
        # Get the control points. Well have 1, 3, 15 for each progressing strength level
        query_points = self.sampler.get_query_points()

        if query_points is None:
            print()
            print("Optimization finished")
            print("+"*50)
            return True # means the optimisation finished, strength to max
        query_points = query_points.to(self.device)

        #Brevity:
        qp = query_points

        # get the spline node locations, velocities and accelerations as x, dy/dx, dy2/dx2
        # These will contain points on the latent manifold (the number of them is determined by the current optimisation strength)
        X_opt = self.spline(qp)  # [num_query_points, flattened_latent_dim] -> first iteration: [1, 1024]
        V_opt = self.spline(qp, 1)
        A_opt = self.spline(qp, 2)

        # Compute gradient
        grad_all, mean_all_norm, mean_angle = self.gradient_unit(X_opt, V_opt, A_opt, query_points)

        # If the acceleration term is too big, we will get pulled away from the geodesic, this just accounts for that by making the optimisation finer
        if grad_all is None:
            self.sampler.add_strength(None)
            self.iter += 1
            return False

        #Learning rate
        it_lr = self.optimizer.get_learning_rate(self.iter, qp)

        control_t = qp.detach().cpu().numpy()

        #Logging
        print('\r\titeration: {}, grad_norm: {:.4f}, angle: {:.4f}, \t strength: {}'.format(
                self.iter, mean_all_norm, mean_angle, self.sampler.cur_strength), end='')

        # Optimisation step:
        X_opt = X_opt - it_lr * grad_all

        if self.project_to_sphere:
            X_opt = norm_fix_batch(X_opt, torch.tensor([self.radius]*X_opt.shape[0]).to(self.device))

        self.iter += 1

        # Update path
        for i, t in enumerate(qp):
            self.path[t.item()] = X_opt[i]

        # Refit the spline with newly optimised points
        new_control_points = torch.tensor(sorted(self.path.keys())).to(self.device)
        new_end_points = torch.stack([self.path[node.item()] for node in new_control_points], dim=0)

        self.spline = SphericalCubicSpline(new_control_points, new_end_points)

        self.sampler.add_strength(self.iter)

        return False

    # Initialisation
    def init(self):
        # Process prompts
        self.prompt1 = self.pipe.encode_prompt(self.config.prompt1)[1].to(self.config.device)
        self.prompt2 = self.pipe.encode_prompt(self.config.prompt2)[1].to(self.config.device)
        self.uncond_prompt_embed = self.pipe.encode_prompt(self.config.uncond_prompt)[1].to(self.config.device)
        self.neg_prompt_embed = self.pipe.encode_prompt(self.config.neg_prompt)[1].to(self.config.device)

        # Process images
        self.image_tensor1 = self.pipe.preprocess_image(self.image_path1).to(self.device)
        self.image_tensor2 = self.pipe.preprocess_image(self.image_path2).to(self.device)

        # Encode the images
        self.image_latent1 = self.pipe.encode_image(self.image_tensor1)
        self.image_latent2 = self.pipe.encode_image(self.image_tensor2)

        # We start by inverting the prompts
        self.prompt_embed_opt1 = self.editor.text_inverter.load_text_inversion(self.prompt1, self.image_latent1, self.test_name, 'A', **self.text_inv_args)
        self.prompt_embed_opt2 = self.editor.text_inverter.load_text_inversion(self.prompt2, self.image_latent2, self.test_name, 'B', **self.text_inv_args)

        print("Starting DDIM noising")
        # Then we deterministically noise the latents:
        noised_latent1 = self.editor.latent_proc.ddim_forward(self.noise_level, self.guidance_scale, self.image_latent1, self.uncond_prompt_embed, self.prompt_embed_opt1, self.neg_prompt_embed, self.use_neg_cfg).reshape(-1)
        noised_latent2 = self.editor.latent_proc.ddim_forward(self.noise_level, self.guidance_scale, self.image_latent2, self.uncond_prompt_embed, self.prompt_embed_opt2, self.neg_prompt_embed, self.use_neg_cfg).reshape(-1)

        # for brevity (these are the points on the latent space which we interpolate between)
        p1 = noised_latent1
        p2 = noised_latent2

        # Fix the endpoints to be on the radius of the hypersphere:
        if self.project_to_sphere:
            self.radius = 0.5 * (torch.norm(p1) + torch.norm(p2))
            p1 = norm_fix(p1, self.radius)
            p2 = norm_fix(p2, self.radius)

        self.path = dict()
        self.path[0] = p1
        self.path[1] = p2

        # Now we can initialize the spline for geodesic optimisation.
        control_points = torch.tensor([0.0,1.0]).to(self.device)
        end_points = torch.stack([p1, p2], dim=0)

        torch.cuda.empty_cache()
        self.spline = SphericalCubicSpline(control_points, end_points)

   # Run Optimisation
    def optimise(self):
        self.iter = 0
        self.edit_idx = self.semantic_edit_args["image_idx"]

        self.update_state()

        self.bvp_io_unit.output_spline_images('start')
        
        print()
        print("+"*50)
        print(f"Testing: {self.test_name}, maximum iterations: {self.optimizer.opt_max_iter} \nStarting spline optimisation:")
        print("+"*50)

        for i in range(self.optimizer.opt_max_iter):
            # Main loop occurs here!
            finish = self.step()

            #This stuff is just I/O
            if finish or i == self.optimizer.opt_max_iter - 1:

                torch.cuda.empty_cache()

                spline_latent =  self.spline(self.timesteps_out)[self.edit_idx]

                # Run the pullback diffusion method to move the latent in semantically meaningful directions
                # self.bvp_io_unit.output_semantic_edit_latent(spline_latent, 'semantic')

                #self.bvp_io_unit.output_semantic_edit_input_image(self.image_path1, 1)
                #self.bvp_io_unit.output_semantic_edit_input_image(self.image_path2, 2)

                # Output (un-edited) images from the spline interpolation
                self.bvp_io_unit.output_spline_images('final')
                break
            else:
                self.bvp_io_unit.output_spline_images(str(self.iter))

            self.update_state()
            torch.cuda.empty_cache()

        self.bvp_io_unit.save_optimisation(self.path)

        ts = torch.linspace(0, 1, self.num_output_imgs, device=self.device)
        torch.save(self.spline(ts,1), 'checkpoints/final_vs.pt')

    def update_state(self):
        state = self.state  # local alias for readability

        state.iter = self.iter
        state.image_latent1 = self.image_latent1
        state.image_latent2 = self.image_latent2
        state.image_tensor1 = self.image_tensor1
        state.image_tensor2 = self.image_tensor2
        state.prompt_embed_opt1 = self.prompt_embed_opt1
        state.prompt_embed_opt2 = self.prompt_embed_opt2
        state.uncond_prompt_embed = self.uncond_prompt_embed
        state.neg_prompt_embed = self.neg_prompt_embed
        state.timesteps_out = self.timesteps_out
        state.spline = self.spline
        state.score_unit = self.score_unit
        state.edit_idx = self.edit_idx
