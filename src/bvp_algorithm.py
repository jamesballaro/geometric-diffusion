import os
import torch
import torch.nn.functional as F
import functools

from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Any

from geodesic import SphericalCubicSpline, BisectionSampler
from geodesic import norm_fix, norm_fix_batch, o_project, o_project_batch
from score import ScoreProcessor
from image_io import ImageProcessor, IO

"""
    This file will contain:
    - text inversion
    - sphere constraining
    - gradient calculations
"""
class BVPOptimiser():
    def __init__(self, device, opt_max_iter, opt_lr, lr_scheduler='linear', lr_divide=True):
        self.opt_max_iter = opt_max_iter
        self.lr_init = opt_lr
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

@dataclass
class BVPConfig:
    """Configuration for BVP Algorithm"""
    # Test setting
    test_name: str
    image_path1: str
    image_path2: str
    prompt1: str
    prompt2: str
    output_dir: str

    # CFG
    uncond_prompt: str
    neg_prompt: str
    noise_level: float
    alpha: float
    guidance_scale: float
    use_neg_cfg: bool

    # Output settings
    output_start_images: bool
    num_output_imgs: int
    use_pu_sampling: bool
    grad_analysis_out: bool
    output_interval: int
    output_separate_images: bool = False
    project_to_sphere: bool = True

    # Grouped args
    grad_args: Dict[str, Any] = None
    bvp_opt_args: Dict[str, Any] = None
    bisection_args: Dict[str, Any] = None
    text_inv_args: Dict[str, Any] = None
    semantic_edit_args: Dict[str, Any] = Non

class BVPAlgorithm():
    def __init__(self, device, pipe, config: BVPConfig):
        self.device = device
        self.pipe = pipe
        self.config = config
        # Copy all config attributes to self
        for key, value in config.__dict__.items():
            setattr(self, key, value)

        # Process prompts
        self.prompt1 = pipe.encode_prompt(config.prompt1)[1].to(device)
        self.prompt2 = pipe.encode_prompt(config.prompt2)[1].to(device)
        self.uncond_prompt_embed = pipe.encode_prompt(config.uncond_prompt)[1].to(device)
        self.neg_prompt_embed = pipe.encode_prompt(config.neg_prompt)[1].to(device)

        self.init_submodules()

    def init_submodules(self):
        # Initialize optimiser sub units
        self.sampler = BisectionSampler(self.device, **self.bisection_args)
        self.score_unit = ScoreProcessor(
            self.pipe,
            self.device,
            self.uncond_prompt_embed,
            self.neg_prompt_embed,
            self.noise_level,
            **self.grad_args
        )
        self.optimizer = BVPOptimiser(
            self.device,
            **self.bvp_opt_args
        )
        self.io_unit = IO(
            self.output_dir,
            resolution=self.pipe.resolution,
            output_separate_images=self.output_separate_images
            )

        self.timesteps_out = torch.linspace(0, 1, self.num_output_imgs).to(self.device)
        self.image_proc = ImageProcessor(
            self.pipe,
            self.device,
            self.guidance_scale,
            self.noise_level,
            self.uncond_prompt_embed,
            self.neg_prompt_embed,
            self.timesteps_out,
            use_neg_cfg=self.use_neg_cfg,
            use_pu_sampling=self.use_pu_sampling,
            output_start_images=self.output_start_images
        )

    # Main functions
    def output_spline_images(self, out_name):
        """
        Decide whether to save the BVP image sequence based on iteration number and settings.
        """
        # Conditions for saving
        is_start = (self.iter == 0 and self.output_start_images)
        is_interval = (self.output_interval > 0 and self.iter > 0 and self.iter % self.output_interval == 0)
        is_final = (out_name == 'final')

        if not (is_start or is_interval or is_final):
            return None

        if is_start: print("Output start images.")
        elif is_interval: print(f"Output image sequence at iteration {self.iter}")
        elif is_final: print("Output final image sequence")

        # Generate
        images = self.image_proc.produce_images(
            self.spline,
            self.prompt_embed_opt1, self.prompt_embed_opt2,
            self.image_tensor1, self.image_tensor2,
            out_name
        )

        # Save
        image_list = [img for img in images]
        return self.io_unit.save_images(image_list, out_name, edit_idx=self.edit_idx)

    def output_semantic_edit_input_image(self, image_path, select):

        image_tensor = self.pipe.preprocess_image(image_path).to(self.device)
        image_latent = self.pipe.encode_image(image_tensor)

        full_noise_level = 1
        end_timestep =  self.pipe.get_timesteps(self.noise_level, return_single=True) # e.g 400


        if select == 1:
            prompt_embed = self.prompt_embed_opt1
        else:
            prompt_embed = self.prompt_embed_opt2

        edit_prompt = self.semantic_edit_args["edit_prompt"]
        edit_prompt_embed = self.pipe.encode_prompt(edit_prompt)[1].to(self.device)

        interp_prompt = prompt_embed + edit_prompt

        noised_latent_T = self.pipe.ddim_forward(
            full_noise_level,
            self.guidance_scale,
            image_latent,
            self.uncond_prompt_embed,
            prompt_embed,
            self.neg_prompt_embed,
            self.use_neg_cfg
        )

        ddim_backward_fn = functools.partial(
            self.pipe.ddim_backward,
            guidance_scale=self.guidance_scale,
            neg_prompt_embed=self.neg_prompt_embed,
            uncond_prompt_embed=self.uncond_prompt_embed,
            prompt_embed=prompt_embed,
            eta=0.0,
            use_neg_cfg=self.use_neg_cfg,
        )

        noised_latent_t = ddim_backward_fn(
            noise_level=1,
            latent=noised_latent_T,
            end_timestep=end_timestep,
        )

        denoised_edited_latents = self.pipe.run_edit_local_encoder_pullback_zt(
            self.noise_level,
            noised_latent_t,
            noised_latent_T,
            0,
            self.semantic_edit_args['op'],
            self.semantic_edit_args["vis_num"],
            self.semantic_edit_args["vis_num_pc"],
            self.semantic_edit_args["pca_rank"],
            self.semantic_edit_args["x_guidance_step"],
            self.semantic_edit_args["x_guidance_strength"],
            backward_fn=ddim_backward_fn,
            edit_prompt=self.semantic_edit_args["edit_prompt"],
            output_dir=self.output_dir
        )

        # Save images
        for rank, pca_rank_latent in enumerate(denoised_edited_latents):
            dir_latent = torch.cat(pca_rank_latent)
            images = self.pipe.decode_latent(dir_latent)
            image_list = [img.unsqueeze(0) for img in images]

            image_list = self.image_proc.normalize_image_batch(image_list)
            self.io_unit.save_images(image_list, f'from_{image_path[:-4]}_pca_rank_{rank}')

        return

    def output_semantic_edit_latent(self, latent, out_name):

        # latents are noised to t = 0.6 * T
        latent = latent.reshape(-1, 4, self.latent_dim, self.latent_dim)

        edit_prompt = self.semantic_edit_args["edit_prompt"]
        edit_prompt_embed = self.pipe.encode_prompt(edit_prompt)[1].to(self.device)

        interp_prompt = self.spline.lerp(self.timesteps_out, self.prompt_embed_opt1, self.prompt_embed_opt2)

        # Help with reverse CFG for semantic editing
        interp_prompt = interp_prompt + edit_prompt_embed.expand_as(interp_prompt)

        denoised_edited_latents = self.pipe.run_encoder_pullback_image_latent(
            latent,
            self.noise_level,
            self.guidance_scale,
            self.uncond_prompt_embed,
            self.neg_prompt_embed,
            interp_prompt[self.edit_idx:self.edit_idx+1,:,:],
            self.use_neg_cfg,
            self.semantic_edit_args['op'],
            self.semantic_edit_args["vis_num"],
            self.semantic_edit_args["vis_num_pc"],
            self.semantic_edit_args["pca_rank"],
            edit_prompt,
            self.semantic_edit_args["x_guidance_step"],
            self.semantic_edit_args["x_guidance_strength"],
            output_dir=self.output_dir,
        )

        # Save images
        for pca_rank_latent in denoised_edited_latents:
            dir_latent = torch.cat(pca_rank_latent)

            images = self.pipe.decode_latent(dir_latent)
            image_list = [img.unsqueeze(0) for img in images]
            image_list = self.image_proc.normalize_image_batch(image_list)

        print(len(image_list))

        self.io_unit.save_images(image_list, out_name)

        return

    def output_interp_images(self, method):

        # Then we deterministically noise the latents:
        noised_latent1 = self.pipe.ddim_forward(self.noise_level, self.guidance_scale, self.image_latent1, self.uncond_prompt_embed, self.prompt_embed_opt1, self.neg_prompt_embed, self.use_neg_cfg)
        noised_latent2 = self.pipe.ddim_forward(self.noise_level, self.guidance_scale, self.image_latent2, self.uncond_prompt_embed, self.prompt_embed_opt2, self.neg_prompt_embed, self.use_neg_cfg)

        interp_lat =[]
        if method == 'slerp':
            for a in self.timesteps_out:
                interp_lat.append(self.spline.slerp(self.timesteps_out, noised_latent1, noised_latent2, a))
        else:
            interp_lat = self.spline.lerp(self.timesteps_out, noised_latent1, noised_latent2)

        interp_prompt = self.spline.lerp(self.timesteps_out, self.prompt_embed_opt1, self.prompt_embed_opt2)
        images=[]
        for i in range(self.num_output_imgs):
            print(f"\nImage {i+1}/{self.num_output_imgs} | noise_level: {self.noise_level} | guidance_scale: {self.guidance_scale} | using negative cfg: {self.use_neg_cfg}| ")
            latent = self.pipe.ddim_backward(
                self.noise_level,
                self.guidance_scale,
                interp_lat[i:i+1],
                self.neg_prompt_embed,
                self.uncond_prompt_embed,
                interp_prompt[i:i+1,:,:],
                eta=0.0,
                use_neg_cfg=self.use_neg_cfg
            )

            image = self.pipe.decode_latent(latent)
            images.append(image)
        images = self.image_proc.normalize_image_batch(images)

        self.io_unit.save_images(images, f'{method}_{self.test_name}')
        return

    def bvp_gradient(self, X, V, A, t_opt) :
        # Latent downsizing factor for SD2.1 = 8
        self.latent_dim = int(self.pipe.resolution[0] / 8)

        latents = X.reshape(-1, 4, self.latent_dim, self.latent_dim)

        # Linear interpolation of text prompt-embeddings
        prompt_embed = self.spline.lerp(t_opt, self.prompt_embed_opt1, self.prompt_embed_opt2)

        # Compute score functions ( ∇ log p) from latent and prompt embedding
        scores = self.score_unit.grad_compute_batch(latents, prompt_embed)

        B, C, H, W = scores.shape

        # Flatten the scores
        scores = scores.reshape(B,-1)

        # This is the (I - ŷŷ) in the functional derivative which ensures that only the normal component of the score function affects the geodesic
        if self.project_to_sphere:
            scores = o_project_batch(scores, X)
            A = o_project_batch(A, X)

        # Compute the scaled acceleration term
        V_norm2 = torch.sum(V * V, dim=-1)
        A_scaled = A / V_norm2[:,None]

        # Project back to the hypersphere, these are the inner terms in the FuncDeriv
        term1 = o_project_batch(scores, V)
        term2 = o_project_batch(A_scaled, V) * (1/self.alpha)

        # Assemble the functional derivative
        grad_all = -(term1 + term2)

        # Grad analysis
        mean_all_norm, mean_norm1, mean_norm2, mean_angle = self.io_unit.grad_analysis(B, t_opt, self.iter, term1, term2, grad_all)

        if mean_norm1 < mean_norm2:
            # This is a heuristic, if the acceleration term too big, it will go to the wrong direction
            # Setting the learning rate to be super small can avoid this issue
            return None, mean_all_norm, mean_angle

        return grad_all, mean_all_norm, mean_angle

    def step(self):
        # Get the control points. Well have 1, 3, 15 for each progressing strength level
        query_points = self.sampler.get_query_points()

        if query_points is None:
            print("t_opt is none")
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
        grad_all, mean_all_norm, mean_angle = self.bvp_gradient(X_opt, V_opt, A_opt, query_points)

        # If the acceleration term is too big, we will get pulled away from the geodesic, this just accounts for that by making the optimisation finer
        if grad_all is None:
            self.sampler.add_strength(None)
            self.iter += 1
            return False

        #Learning rate
        it_lr = self.optimizer.get_learning_rate(self.iter, qp)

        control_t = qp.detach().cpu().numpy()

        #Logging
        if self.iter % 5 == 0:
            print('optimise {} iteration: {}, grad_norm: {}, angle: {}'.format(
                        self.test_name, self.iter, mean_all_norm, mean_angle))
            print(f't = [{control_t}]')

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

    def init(self):
        self.image_tensor1 = self.pipe.preprocess_image(self.image_path1).to(self.device)
        self.image_tensor2 = self.pipe.preprocess_image(self.image_path2).to(self.device)

        # Encode the images
        self.image_latent1 = self.pipe.encode_image(self.image_tensor1)
        self.image_latent2 = self.pipe.encode_image(self.image_tensor2)

        # We start by inverting the prompts
        self.prompt_embed_opt1 = self.pipe.load_text_inversion(self.prompt1, self.image_latent1, self.test_name, 'A', **self.text_inv_args)
        self.prompt_embed_opt2 = self.pipe.load_text_inversion(self.prompt2, self.image_latent1, self.test_name, 'B', **self.text_inv_args)

        print("Starting DDIM noising")
        # Then we deterministically noise the latents:
        noised_latent1 = self.pipe.ddim_forward(self.noise_level, self.guidance_scale, self.image_latent1, self.uncond_prompt_embed, self.prompt_embed_opt1, self.neg_prompt_embed, self.use_neg_cfg).reshape(-1)
        noised_latent2 = self.pipe.ddim_forward(self.noise_level, self.guidance_scale, self.image_latent2, self.uncond_prompt_embed, self.prompt_embed_opt2, self.neg_prompt_embed, self.use_neg_cfg).reshape(-1)

        print(f"Image 1: min {noised_latent1.min().item()}, max {noised_latent1.max().item()}, mean {noised_latent1.mean().item()}")
        print(f"Image 2: min {noised_latent2.min().item()}, max {noised_latent2.max().item()}, mean {noised_latent2.mean().item()}")

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

        print("\nInitializing Spline")
        torch.cuda.empty_cache()
        self.spline = SphericalCubicSpline(control_points, end_points)

    def optimise(self):

        self.iter = 0

        self.output_spline_images('start')

        for i in range(self.optimizer.opt_max_iter):

            # Main loop occurs here!
            finish = self.step()

            #This stuff is just I/O
            if finish or i == self.optimizer.opt_max_iter - 1:

                torch.cuda.empty_cache()
                self.edit_idx = self.semantic_edit_args["image_idx"]
                print("edit_idx", self.edit_idx)

                spline_latent =  self.spline(self.timesteps_out)[self.edit_idx]

                # Run the pullback diffusion method to move the latent in semantically meaningful directions
                self.output_semantic_edit_latent(spline_latent, 'semantic')

                self.output_semantic_edit_input_image(self.image_path1, 1)
                self.output_semantic_edit_input_image(self.image_path2, 2)

                # Output (un-edited) images from the spline interpolation
                self.output_spline_images('final')
                break
            else:
                self.output_spline_images(str(self.iter))

            torch.cuda.empty_cache()


        self.io_unit.save_optimisation(self.path)

        ts = torch.linspace(0, 1, self.num_output_imgs, device=self.device)
        torch.save(self.spline(ts,1), os.path.join(self.output_dir, 'final_vs.pt'))


