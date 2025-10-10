from bvp_structs import BVPConfig, BVPState

class BVP_OutputModule():
    def __init__(self, config: BVPConfig):
        self.config = config
    # Main functions
    def output_spline_images(self, state: BVPState, out_name):
        """
        Decide whether to save the BVP image sequence based on iteration number and settings.
        """
        it = state.iter
        # Conditions for saving
        is_start = (it == 0 and self.config.output_start_images)
        is_interval = (self.config.output_interval > 0 and it > 0 and it % self.config.output_interval == 0)
        is_final = (out_name == 'final')

        if not (is_start or is_interval or is_final):
            return None

        if is_start: print("Output start images.")
        elif is_interval: print(f"Output image sequence at iteration {it}")
        elif is_final: print("Output final image sequence")

        # Generate
        images = state.image_proc.produce_images(
            state.spline,
            state.prompt_embed_opt1, state.prompt_embed_opt2,
            state.image_tensor1, state.image_tensor2,
            out_name
        )

        # Save
        image_list = [img for img in images]
        state.io_unit.save_images(image_list, out_name, edit_idx=state.edit_idx)

    def output_semantic_edit_latent(self, state: BVPState,  latent, out_name):
        """
        This function uses the local encoder pullback to generate a latent space image sequence
        """

        latent_dim = int(self.config.resolution[0] / 8)
        # latents are noised to t = 0.6 * T
        latent = latent.reshape(-1, 4, latent_dim, latent_dim)

        edit_prompt = self.config.semantic_edit_args["edit_prompt"]
        edit_prompt_embed = self.config.pipe.encode_prompt(edit_prompt)[1].to(self.device) #TODO device

        interp_prompt = state.spline.lerp(state.timesteps_out, state.prompt_embed_opt1, state.prompt_embed_opt2)

        # Help with reverse CFG for semantic editing
        interp_prompt = interp_prompt + edit_prompt_embed.expand_as(interp_prompt)

        denoised_edited_latents = self.config.pipe.run_encoder_pullback_image_latent(
            self.config,
            latent,
            state.uncond_prompt_embed,
            state.neg_prompt_embed,
            interp_prompt[state.edit_idx:state.edit_idx+1,:,:],
            edit_prompt,
            output_dir=self.output_dir,
        )

        # Save images
        for pca_rank_latent in denoised_edited_latents:
            dir_latent = torch.cat(pca_rank_latent)

            images = self.config.pipe.decode_latent(dir_latent)
            image_list = [img.unsqueeze(0) for img in images]
            image_list = state.image_proc.normalize_image_batch(image_list)

        state.io_unit.save_images(image_list, out_name)

        return

    def output_semantic_edit_input_image(self, state: BVPState, image_path, select):
        """
        This function circumvents the optimsation and uses algorithm 2 to semantically edit the input image
        """
        image_tensor = self.config.pipe.preprocess_image(image_path).to(self.config.device)
        image_latent = self.config.pipe.encode_image(image_tensor)

        full_noise_level = 1
        end_timestep =  self.config.pipe.get_timesteps(self.config.noise_level, return_single=True) # e.g 400


        if select == 1:
            prompt_embed = state.prompt_embed_opt1
        else:
            prompt_embed = state.prompt_embed_opt2

        edit_prompt = self.config.semantic_edit_args["edit_prompt"]
        edit_prompt_embed = self.config.pipe.encode_prompt(edit_prompt)[1].to(self.config.device)

        interp_prompt = prompt_embed + edit_prompt

        noised_latent_T = self.config.pipe.ddim_forward(
            full_noise_level,
            self.config.guidance_scale,
            image_latent,
            state.uncond_prompt_embed,
            prompt_embed,
            state.neg_prompt_embed,
            self.config.use_neg_cfg
        )

        ddim_backward_fn = functools.partial(
            self.config.pipe.ddim_backward,
            guidance_scale=self.config.guidance_scale,
            neg_prompt_embed=state.neg_prompt_embed,
            uncond_prompt_embed=state.uncond_prompt_embed,
            prompt_embed=prompt_embed,
            eta=0.0,
            use_neg_cfg=self.config.use_neg_cfg,
        )

        noised_latent_t = ddim_backward_fn(
            noise_level=1,
            latent=noised_latent_T,
            end_timestep=end_timestep,
        )

        denoised_edited_latents = self.config.pipe.run_edit_local_encoder_pullback_zt(
            self.config
            noised_latent_t,
            noised_latent_T,
            0,
            backward_fn=ddim_backward_fn,
            output_dir=self.config.output_dir
        )

        # Save images
        for rank, pca_rank_latent in enumerate(denoised_edited_latents):
            dir_latent = torch.cat(pca_rank_latent)
            images = self.config.pipe.decode_latent(dir_latent)
            image_list = [img.unsqueeze(0) for img in images]
            image_list = state.image_proc.normalize_image_batch(image_list)
            state.io_unit.save_images(image_list, f'from_{image_path[:-4]}_pca_rank_{rank}')

        return

    def output_interp_images(self, state: BVPState,  method):
        """
        This functions uses standard linear interpolation to generate a continuous image sequence
        """"
        # Then we deterministically noise the latents:
        noised_latent1 = self.config.pipe.ddim_forward(self.config.noise_level, self.config.guidance_scale, state.image_latent1, state.uncond_prompt_embed, state.prompt_embed_opt1, state.neg_prompt_embed, self.config.use_neg_cfg)
        noised_latent2 = self.config.pipe.ddim_forward(self.config.noise_level, self.config.guidance_scale, state.image_latent2, state.uncond_prompt_embed, state.prompt_embed_opt2, state.neg_prompt_embed, self.config.use_neg_cfg)

        interp_lat =[]
        if method == 'slerp':
            for a in self.timesteps_out:
                interp_lat.append(state.spline.slerp(state.timesteps_out, noised_latent1, noised_latent2, a))
        else:
            interp_lat = state.spline.lerp(state.timesteps_out, noised_latent1, noised_latent2)

        interp_prompt = state.spline.lerp(state.timesteps_out, state.prompt_embed_opt1, state.prompt_embed_opt2)
        images=[]
        for i in range(self.config.num_output_imgs):
            print(f"\nImage {i+1}/{self.config.num_output_imgs} | noise_level: {self.config.noise_level} | guidance_scale: {self.config.guidance_scale} | using negative cfg: {self.config.use_neg_cfg}| ")
            latent = self.config.pipe.ddim_backward(
                self.config.noise_level,
                self.config.guidance_scale,
                interp_lat[i:i+1],
                self.state.neg_prompt_embed,
                self.state.uncond_prompt_embed,
                interp_prompt[i:i+1,:,:],
                eta=0.0,
                use_neg_cfg=self.config.use_neg_cfg
            )

            image = self.config.pipe.decode_latent(latent)
            images.append(image)
        images = state.image_proc.normalize_image_batch(images)

        state.io_unit.save_images(images, f'{method}_{self.config.test_name}')
        return
