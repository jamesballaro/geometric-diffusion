import torch
from scheduler import Scheduler
from PIL import Image, ImageDraw
import os
from torchvision import transforms



"""
To process all the outputs from the BVP algorithm
"""

class ImageProcessor():
    def __init__(self,
        pipe,
        device,
        guidance_scale,
        noise_level,
        uncond_prompt_embed,
        neg_prompt_embed,
        timesteps_out,
        use_neg_cfg=True,
        output_interval=-1,
        use_pu_sampling=True,
        output_start_images=True,
        ):

        self.pipe = pipe
        self.device = device
        self.guidance_scale = guidance_scale
        self.noise_level = noise_level
        self.uncond_prompt_embed = uncond_prompt_embed
        self.neg_prompt_embed = neg_prompt_embed
        self.use_neg_cfg = use_neg_cfg
        self.output_interval = output_interval            
        self.use_pu_sampling = use_pu_sampling              
        self.output_start_images = output_start_images

        self.timesteps_out = timesteps_out

    def decode_spline_latents(self, latent_spline, interp_prompt):
        assert latent_spline.shape[0] == interp_prompt.shape[0]
        images = []
        num_images = latent_spline.shape[0]
        latent_dim = int(self.pipe.resolution[0] / 8)

        print(f"\nDecoding {num_images} latents into image...")
        for i in range(num_images):
            print(f"\nImage {i+1}/{num_images} | noise_level: {self.noise_level} | guidance_scale: {self.guidance_scale} | using negative cfg: {self.use_neg_cfg}| ")
            latent = self.pipe.ddim_backward(
                self.noise_level, 
                self.guidance_scale, 
                latent_spline[i].reshape(1,4, latent_dim, latent_dim), 
                self.neg_prompt_embed, 
                self.uncond_prompt_embed,
                interp_prompt[i:i+1,:,:],
                eta=0.0,
                use_neg_cfg=self.use_neg_cfg
            )

            image = self.pipe.decode_latent(latent)
            images.append(image)

        return images
    
    def get_spline_images(self, spline, interp_prompt, image1, image2):
        # Do we want the start images too?
        images = self.decode_spline_latents(spline(self.timesteps_out), interp_prompt)

        if not self.output_start_images:
            assert image1 is not None and image2 is not None
            images = [image1] + images[1:-1] + [image2]

        return images

    def produce_images(self, spline, prompt_embed1, prompt_embed2, image1, image2, out_name):
        # output the images from spline and given t, embed_cond, and save them
        # consider if we want to do perceptual uniform sampling, and save the images separately
        interp_prompt = spline.lerp(self.timesteps_out, prompt_embed1, prompt_embed2)

        images = self.get_spline_images(spline, interp_prompt, image1, image2)

        if self.use_pu_sampling and out_name == 'final':

            print('Perceptual Uniform sampling ...')
            scheduler = Scheduler(self.pipe.device)
            
            # images_pt = [transforms.ToTensor()(image).unsqueeze(0)
            #                     for image in images]
            
            images_pt = [image.to(self.pipe.device) for image in images]
            
            scheduler.from_images(images_pt)
            timesteps_out = scheduler.get_list() # start from 0 to 1
            
            print('p-sampled t: ',list(timesteps_out))
            
            interp_prompt = spline.lerp(timesteps_out, prompt_embed1, prompt_embed2)[1:-1]
            
            out_tensor = torch.tensor(timesteps_out).to(self.device).clip(0,1)
            
            X = spline(out_tensor)   

            images[1:-1] = self.decode_spline_latents(X[1:-1], interp_prompt)
        
        images = self.normalize_image_batch(images)
        return images

    def normalize_image(self, image_tensor):
        image = (image_tensor / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        pil_image = Image.fromarray(image[0])
        return pil_image
    
    def normalize_image_batch(self, image_list):
        images = []
        for image_tensor in image_list:
            images.append(self.normalize_image(image_tensor))
        return images


class IO():
    def __init__(
        self,
        output_dir,
        resolution,
        output_separate_images=False,
        grad_analysis_out=False,
        ):
        self.output_dir = output_dir
        self.output_separate_images = output_separate_images
        self.resolution = resolution
        self.grad_analysis_out = grad_analysis_out

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
