from bvp_structs import BVPConfig, BVPState

class BVP_OutputModule():
    def __init__(self, pipe, editor, state: BVPState, config: BVPConfig):
        self.pipe = pipe
        self.editor = editor
        self.image_proc = image_proc
        self.state = state
        self.config = config

        self.output_dir = config.output_dir
        self.output_separate_images = config.output_separate_images
        self.resolution = config.resolution
        self.grad_analysis_out = config.grad_analysis_out

    # Main functions
    def output_spline_images(self, out_name):
        """
        Decide whether to save the BVP image sequence based on iteration number and settings.
        """
        it = self.state.iter
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
        images = self.image_proc.produce_images(
            self.state.spline,
            self.state.prompt_embed_opt1, self.state.prompt_embed_opt2,
            self.state.image_tensor1, self.state.image_tensor2,
            out_name
        )

        # Save
        image_list = [img for img in images]
        self.save_images(image_list, out_name, edit_idx=self.state.edit_idx)

    def output_semantic_edit_latent(self, latent):

        denoised_edited_latents = self.editor.calculate_semantic_edit_latent(latent, out_name)

        # Save images
        for pca_rank_latent in denoised_edited_latents:
            dir_latent = torch.cat(pca_rank_latent)

            images = self.pipe.decode_latent(dir_latent)
            image_list = [img.unsqueeze(0) for img in images]
            image_list = self.image_proc.normalize_image_batch(image_list)

        self.save_images(image_list, out_name)

        return

    def output_semantic_edit_input_image(self, image_path, select):

        denoised_edited_latents = self.editor.calculate_semantic_edit_input_image(image_path, select)

        # Save images
        for rank, pca_rank_latent in enumerate(denoised_edited_latents):
            dir_latent = torch.cat(pca_rank_latent)
            images = self.pipe.decode_latent(dir_latent)
            image_list = [img.unsqueeze(0) for img in images]
            image_list = self.image_proc.normalize_image_batch(image_list)
            self.save_images(image_list, f'from_{image_path[:-4]}_pca_rank_{rank}')

        return

    def output_interp_images(self,  method):
        """
        This functions uses standard linear interpolation to generate a continuous image sequence
        """"
        # Then we deterministically noise the latents:
        noised_latent1 = self.editor.latent_proc.ddim_forward(self.config.noise_level, self.config.guidance_scale, self.state.image_latent1, self.state.uncond_prompt_embed, self.state.prompt_embed_opt1, self.state.neg_prompt_embed, self.config.use_neg_cfg)
        noised_latent2 = self.editor.latent_proc.ddim_forward(self.config.noise_level, self.config.guidance_scale, self.state.image_latent2, self.state.uncond_prompt_embed, self.state.prompt_embed_opt2, self.state.neg_prompt_embed, self.config.use_neg_cfg)

        interp_lat =[]
        if method == 'slerp':
            for a in self.timesteps_out:
                interp_lat.append(self.state.spline.slerp(self.state.timesteps_out, noised_latent1, noised_latent2, a))
        else:
            interp_lat = self.state.spline.lerp(self.state.timesteps_out, noised_latent1, noised_latent2)

        interp_prompt = self.state.spline.lerp(self.state.timesteps_out, self.state.prompt_embed_opt1, self.state.prompt_embed_opt2)
        images=[]
        
        for i in range(self.config.num_output_imgs):
            print(f"\nImage {i+1}/{self.config.num_output_imgs} | noise_level: {self.config.noise_level} | guidance_scale: {self.config.guidance_scale} | using negative cfg: {self.config.use_neg_cfg}| ")
            latent = self.editor.latent_proc.ddim_backward(
                self.config.noise_level,
                self.config.guidance_scale,
                interp_lat[i:i+1],
                self.state.neg_prompt_embed,
                self.state.uncond_prompt_embed,
                interp_prompt[i:i+1,:,:],
                eta=0.0,
                use_neg_cfg=self.config.use_neg_cfg
            )

            image = self.pipe.decode_latent(latent)
            images.append(image)
        images = self.image_proc.normalize_image_batch(images)

        self.save_images(images, f'{method}_{self.config.test_name}')
        return

    def grad_analysis(self, B, t_opt, iter, grad_term1, grad_term2, grad_all):
        """
        Analyze and log gradient norms and angles between gradient components.
        """
        # Flatten to ensure consistent shape [batch, features]
        grad_term1 = grad_term1.reshape(B, -1)
        grad_term2 = grad_term2.reshape(B, -1)
        grad_all = grad_all.reshape(B, -1)

        # Cosine similarity → angle between term1 and term2
        cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        cos_vals = cos_sim(grad_term1, grad_term2)
        angles = torch.arccos(cos_vals) * 180 / torch.pi  # in degrees

        # Norms
        norm1 = torch.norm(grad_term1, dim=-1)
        norm2 = torch.norm(grad_term2, dim=-1)
        norm_all = torch.norm(grad_all, dim=-1)

        # Mean stats (for return)
        mean_all_norm = norm_all.mean().item()
        mean_norm1 = norm1.mean().item()
        mean_norm2 = norm2.mean().item()
        mean_angle = angles.mean().item()

        # If not writing logs, just return means
        if not self.grad_analysis_out:
            return mean_all_norm, mean_norm1, mean_norm2, mean_angle

        # Otherwise, round and write per-sample values to file
        def to_list(tensor):
            return [round(x.item(), 4) for x in tensor]

        log_data = {
            "t": to_list(t_opt),
            "g_t1": to_list(norm1),
            "g_t2": to_list(norm2),
            "g_all": to_list(norm_all),
            "g_angle": to_list(angles),
        }

        log_path = os.path.join(self.output_dir, 'analysis.txt')
        with open(self.output_dir, 'a') as f:
            f.write(f"iter:{iter}\n")
            for key, values in log_data.items():
                f.write(f"{key}:{values}\n")
            f.write("\n")

        return mean_all_norm, mean_norm1, mean_norm2, mean_angle

    def save_images(self, image_list, out_name, edit_idx=None):
        if self.output_separate_images:
            for i, image in enumerate(image_list):
                if out_name == 'start':
                    image.save(os.path.join(self.output_dir, 'start_imgs',  f'{i:02d}.png'))
                else:
                    image.save(os.path.join(self.output_dir, 'out_imgs',  f'{i:02d}.png'))

        image_long = self.display_alongside(image_list, edit_idx)
        image_path = os.path.join(self.output_dir, f'long_{out_name}.png')
        image_long.save(image_path)
        print(f'Image sequence saved to {self.output_dir}long_{out_name}.png')
        return image_list

    def display_alongside(self, image_list, edit_idx=None, padding=10, frame_color=(255, 255, 255), edit_color=(255, 0, 0), edit_width=10):
        padded_width = self.resolution[0] + 2 * padding
        padded_height = self.resolution[1] + 2 * padding
        res = Image.new("RGB", (padded_width * len(image_list), padded_height), frame_color)
        draw = ImageDraw.Draw(res)

        for i, image in enumerate(image_list):
            x_offset = i * padded_width + padding
            y_offset = padding
            img_resized = image.resize(self.resolution)
            res.paste(img_resized, (x_offset, y_offset))

            # Draw a border if the image is edited
            if edit_idx:
                if i == edit_idx:
                    rect_start = (x_offset - edit_width//2, y_offset - edit_width//2)
                    rect_end = (x_offset + self.resolution[0] + edit_width//2, y_offset + self.resolution[1] + edit_width//2)
                    draw.rectangle([rect_start, rect_end], outline=edit_color, width=edit_width)

        return res

    def save_optimisation(self, path):
        torch.save(path, os.path.join(self.output_dir, 'opt_points.pth'))
        print(f'Optimisation points saved to {self.output_dir}opt_points.pth')
