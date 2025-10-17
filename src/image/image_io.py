import torch
from scheduler import Scheduler
from PIL import Image, ImageDraw
import os
from torchvision import transforms

class ImageProcessor():
    def __init__(self, pipe, config, state):

        self.pipe = pipe
        self.device = config.device
        self.guidance_scale = config.guidance_scale
        self.noise_level = config.noise_level
        self.uncond_prompt_embed = state.uncond_prompt_embed
        self.neg_prompt_embed = state.neg_prompt_embed
        self.use_neg_cfg = config.use_neg_cfg
        self.output_interval = config.output_interval
        self.use_pu_sampling = config.use_pu_sampling
        self.output_start_images = config.output_start_images

        self.timesteps_out = state.timesteps_out

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
