import torch
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from .. import UNet as CustomUNet

from diffusers.configuration_utils import FrozenDict
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

class CondDDPMPipeline(DiffusionPipeline):
    def __init__(
        self,
        unet: CustomUNet,
        scheduler: KarrasDiffusionSchedulers
    ):
        super().__init__()
        
        self.unet = unet
        self.scheduler = scheduler
        
        self.register_modules(
            unet=unet,
            scheduler=scheduler
        )
        
        # vars that should be (but aren't) inherited from DiffusionPipeline
        self._execution_device = "cuda"
        
    def decode_image(self, raw_img):
        image = (raw_img / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    @torch.no_grad()
    def __call__(
        self,
        input_classes: torch.FloatTensor = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil", # pil or tensor
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1
    ):
        # 0. Default height and width to unet
        height = height or self.unet.default_sample_size
        width = width or self.unet.default_sample_size

        # 2. Define call parameters
        batch_size = input_classes.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare noise variables
        images = torch.randn((batch_size, self.unet.in_channels, height, width), device=device)
        
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                model_input = torch.cat([images] * 2) if do_classifier_free_guidance else images
                model_input = self.scheduler.scale_model_input(model_input, t)
                
                t_batch = torch.ones((batch_size,)).to(device) * t

                # predict the noise residual
                noise_pred = self.unet(
                    model_input,
                    t_batch.to(dtype=torch.float32),
                    input_classes.to(dtype=torch.float32),
                    debug=False
                )

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                images = self.scheduler.step(noise_pred, t, images, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        images = self.decode_image(images)
        if output_type == "pil":
            images = self.numpy_to_pil(images)
            
        if not return_dict:
            return (images, False)

        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=False)