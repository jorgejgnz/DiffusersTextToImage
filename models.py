import logging
import math
import os

import random
from pathlib import Path
import warnings
from tqdm.auto import tqdm

import diffusers
import accelerate
import datasets
from diffusers.optimization import get_scheduler

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchsummary import summary
from transformers import Pipeline

from pprint import pprint

#

class BaseModel:
    def __init__(self, name="BaseModel", subfolder=None, condition_type="text"):
        self.name = name
        if subfolder is None: subfolder = "unet"
        self.subfolder = subfolder
        self.condition_type = condition_type

    def setup_optimizer(self, args):
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        self.optimizer = optimizer_cls(
            self.get_trainable().parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon
        )

    def setup_scheduler(self, args):
        if args.lr_scheduler == "reduceonplateau":
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        else:
            self.lr_scheduler = get_scheduler(
                args.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
                num_training_steps=args.num_train_epochs
            )

    def setup_parts(self):
        # load parts, freeze untrainable parts and move parts to device
        raise NotImplementedError

    def setup_accelerator(self, accelerator, train_dataloader):

        # Prepare everything with our `accelerator`.
        self.unet, self.optimizer, new_train_dataloader, self.lr_scheduler = accelerator.prepare(
            self.unet, self.optimizer, train_dataloader, self.lr_scheduler
        )

        weight_dtype = self.get_weight_dtype(accelerator)

        # Move text_encode and vae to gpu and cast to weight_dtype
        self.text_encoder.to(accelerator.device, dtype=weight_dtype)

        return weight_dtype, new_train_dataloader
    
    def setup_noaccelerator(self, weight_dtype="fp32", device="cpu"):
        self.unet = self.unet.to(device, dtype=weight_dtype)
        self.text_encoder = self.text_encoder.to(device, dtype=weight_dtype)

    def get_loss(self, args, batch):

        images = batch["pixel_values"]
        input_ids = batch["input_ids"]

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(images)
        if args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += args.noise_offset * torch.randn(
                (images.shape[0], images.shape[1], 1, 1), device=images.device
            )

        bsz = images.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=images.device)
        timesteps = timesteps.long()

        # Add noise to the images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(input_ids)[0]

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(images, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        
        #print(f"noisy_images:{noisy_images.shape} \n \
        #      timesteps:{timesteps.shape} \n \
        #      input_ids:{input_ids.shape} \n \
        #      encoder_hidden_state:{encoder_hidden_states.shape}") # [b, 512, 4096]

        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_images, timesteps, encoder_hidden_states).sample

        #print(f"model_pred.shape:{model_pred.shape}\ntarget.shape:{target.shape}")

        if args.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss

    def get_trainable(self):
        return self.unet

    def get_pipeline(self, args, accelerator=None, dtype=None):

        if dtype is None:
            dtype = self.get_weight_dtype(accelerator)

        if accelerator is None:
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=self.text_encoder,
                unet=self.unet,
                revision=args.revision,
                #torch_dtype=dtype
            )
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(self.text_encoder),
                tokenizer=self.tokenizer,
                unet=accelerator.unwrap_model(self.unet),
                revision=args.revision,
                torch_dtype=dtype
            )
        
        pipeline.safety_checker = None

        return pipeline
    
    def load_trainable_model(self, input_dir):
        return UNet2DConditionModel.from_pretrained(input_dir, subfolder=self.subfolder)
    
    def get_weight_dtype(self, accelerator):
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        return weight_dtype
    
    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def save(self, args, accelerator=None, dtype=None):
        # get pipeline
        pipeline = self.get_pipeline(args, accelerator=accelerator, dtype=dtype)
         ## replace unet in pipeline by unwrapped unet
         #pipeline.unet = accelerator.unwrap_model(self.unet)
        # pipeline.save_pretrained(args.output_dir)
        pipeline.save_pretrained(args.output_dir)


#####################
# StableDiffusion
#####################

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

class StableDiffusion(BaseModel):
    def __init__(self, name="StableDiffusion", subfolder=None, condition_type="text"):
        self.name = name
        if subfolder is None: subfolder = "unet"
        self.subfolder = subfolder
        self.condition_type = condition_type

    def setup_parts(self, args):

        # Load parts
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False, revision=args.revision
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
        )

        # Freeze parts we don't want to train
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Enable gradient checkpointing
        if hasattr(args, 'gradient_checkpointing'):
            if args.gradient_checkpointing: self.unet.enable_gradient_checkpointing()

        print("Counting parameters...")
        param_info = {
            'text_encoder':{
                'total':sum(p.numel() for p in self.text_encoder.parameters()),
                'trainable':sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
            },
            'vae':{
                'total':sum(p.numel() for p in self.vae.parameters()),
                'trainable':sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
            },
            'unet':{
                'total':sum(p.numel() for p in self.unet.parameters()),
                'trainable':sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
            },
        }
        pprint(param_info,depth=4)
        
    def setup_accelerator(self, accelerator, train_dataloader):

        # Prepare everything with our `accelerator`.
        self.unet, self.optimizer, new_train_dataloader, self.lr_scheduler = accelerator.prepare(
            self.unet, self.optimizer, train_dataloader, self.lr_scheduler
        )

        weight_dtype = self.get_weight_dtype(accelerator)

        # Move text_encode and vae to gpu and cast to weight_dtype
        self.text_encoder.to(accelerator.device, dtype=weight_dtype)
        self.vae.to(accelerator.device, dtype=weight_dtype)

        return weight_dtype, new_train_dataloader
    
    def setup_noaccelerator(self, weight_dtype="fp32", device="cpu"):
        self.unet = self.unet.to(device, dtype=weight_dtype)
        self.vae = self.vae.to(device, dtype=weight_dtype)
        self.text_encoder = self.text_encoder.to(device, dtype=weight_dtype)

    def get_loss(self, args, batch):

        images = batch["pixel_values"]
        input_ids = batch["input_ids"]

        # Convert images to latent space
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += args.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )

        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)    

        # Get the text embedding for conditioning
        #encoder_hidden_states.shape = [batch, 77, 768]
        encoder_hidden_states = self.text_encoder(input_ids)[0]

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            # StableDiff2: AttributeError: 'EulerDiscreteScheduler' object has no attribute 'get_velocity'
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        
        #print(f"noisy_latents:{noisy_latents.shape} \n \
        #      timesteps:{timesteps.shape} \n \
        #      input_ids:{input_ids.shape} \n \
        #      encoder_hidden_state:{encoder_hidden_states.shape}")

        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if args.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss
    
    def get_pipeline(self, args, accelerator=None, dtype=None, overwrite_current_weights=True):

        if dtype is None:
            dtype = self.get_weight_dtype(accelerator)

        if overwrite_current_weights:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                revision=args.revision,
            )
        elif accelerator is None:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=self.text_encoder,
                vae=self.vae,
                unet=self.unet,
                revision=args.revision,
                #torch_dtype=dtype
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=accelerator.unwrap_model(self.vae),
                text_encoder=accelerator.unwrap_model(self.text_encoder),
                tokenizer=self.tokenizer,
                unet=accelerator.unwrap_model(self.unet),
                revision=args.revision,
                torch_dtype=dtype
            )
        
        pipeline.safety_checker = None

        return pipeline


#####################
# NanoStableDiffusion
#####################

class NanoStableDiffusion(StableDiffusion):
    def __init__(self, name="NanoStableDiffusion", subfolder=None, condition_type="text"):
        self.name = name
        if subfolder is None: subfolder = "unet"
        self.subfolder = subfolder
        self.condition_type = condition_type

    def setup_parts(self, args):

        super().setup_parts(args)

        self.unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=4,
            out_channels=4,
            cross_attention_dim=768,
            block_out_channels=[32, 64, 128, 128]
        )

        if hasattr(args, 'gradient_checkpointing'):
            if args.gradient_checkpointing: self.unet.enable_gradient_checkpointing()

#####################
# Nano21StableDiffusion
#####################

class Nano21StableDiffusion(StableDiffusion):
    def __init__(self, name="Nano21StableDiffusion", subfolder=None, condition_type="text"):
        self.name = name
        if subfolder is None: subfolder = "unet"
        self.subfolder = subfolder
        self.condition_type = condition_type

    def setup_parts(self, args):

        super().setup_parts(args)

        self.unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=4,
            out_channels=4,
            center_input_sample=False,
            flip_sin_to_cos=True,
            freq_shift=0,
            down_block_types=['CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'],
            mid_block_type='UNetMidBlock2DCrossAttn',
            up_block_types=['UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'],
            only_cross_attention=False,
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            downsample_padding=1,
            mid_block_scale_factor=1,
            act_fn='silu',
            norm_num_groups=32,
            norm_eps=1e-05,
            cross_attention_dim=1024,
            encoder_hid_dim=None,
            attention_head_dim=[5, 10, 20, 20],
            dual_cross_attention=False,
            use_linear_projection=True,
            class_embed_type=None,
            addition_embed_type=None,
            num_class_embeds=None,
            upcast_attention=False,
            resnet_time_scale_shift='default',
            resnet_skip_time_act=False,
            resnet_out_scale_factor=1.0,
            time_embedding_type='positional',
            time_embedding_dim=None,
            time_embedding_act_fn=None,
            timestep_post_act=None,
            time_cond_proj_dim=None,
            conv_in_kernel=3,
            conv_out_kernel=3,
            projection_class_embeddings_input_dim=None,
            class_embeddings_concat=False,
            mid_block_only_cross_attention=None,
            cross_attention_norm=None,
            addition_embed_type_num_heads=64
        )

        if hasattr(args, 'gradient_checkpointing'):
            if args.gradient_checkpointing: self.unet.enable_gradient_checkpointing()

#####################
# MiniStableDiffusion
#####################

class MiniStableDiffusion(StableDiffusion):
    def __init__(self, name="MiniStableDiffusion", subfolder=None, condition_type="text"):
        self.name = name
        if subfolder is None: subfolder = "unet"
        self.subfolder = subfolder
        self.condition_type = condition_type

    def setup_parts(self, args):

        super().setup_parts(args)

        self.unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=4,
            out_channels=4,
            cross_attention_dim=768,
            block_out_channels=[256, 512, 1024, 1024]
        )

        if hasattr(args, 'gradient_checkpointing'):
            if args.gradient_checkpointing: self.unet.enable_gradient_checkpointing()


#####################
# StableDiffusionLoRA
#####################

from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

class StableDiffusionLoRA(StableDiffusion):
    def __init__(self, name="StableDiffusionLoRA", subfolder=None, condition_type="text"):
        self.name = name
        if subfolder is None: subfolder = "unet"
        self.subfolder = subfolder
        self.condition_type = condition_type

    def setup_parts(self, args):
        super().setup_parts(args)

        print("############################ Antes de LoRA (StableDiffusion) ############################")
        dummy_noisy_latent = torch.ones((1, 4, 8, 8))
        dummy_timesteps = torch.ones((1,))
        dummy_encoder_hidden_states = torch.ones((1, 77, 768))
        #summary(self.unet, input_data=[dummy_noisy_latent, dummy_timesteps, dummy_encoder_hidden_states], depth=12)

        # LoRA specific
        self.unet.requires_grad_(False)
        # Set correct lora layers
        lora_attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

            print("------------------")
            print(f"{name}\nblock_id:{block_id} hidden_size:{hidden_size} cross_attention_dim:{cross_attention_dim if cross_attention_dim is not None else 'None'}")
            print("------------------")

        self.unet.set_attn_processor(lora_attn_procs)

        print("############################ Después de LoRA (StableDiffusion) ############################")
        #summary(self.unet, input_data=[dummy_noisy_latent, dummy_timesteps, dummy_encoder_hidden_states], depth=12)

        print("Counting parameters...")
        param_info = {
            'text_encoder':{
                'total':sum(p.numel() for p in self.text_encoder.parameters()),
                'trainable':sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
            },
            'vae':{
                'total':sum(p.numel() for p in self.vae.parameters()),
                'trainable':sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
            },
            'unet':{
                'total':sum(p.numel() for p in self.unet.parameters()),
                'trainable':sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
            },
        }
        pprint(param_info,depth=4)

    def get_trainable(self):
        return AttnProcsLayers(self.unet.attn_processors)
    
    def get_pipeline(self, args, accelerator=None, dtype=None, overwrite_current_weights=False):
        pipeline = super().get_pipeline(args, accelerator=accelerator, dtype=dtype, overwrite_current_weights=overwrite_current_weights)
        # scale 0.0 to not use LoRA weights
        # scale 1.0 to use only LoRA weights
        pipeline.cross_attention_kwargs={"scale": 1.0}
        return pipeline
    
    def save(self, args, accelerator=None, dtype=None):
        unet = self.unet.to(torch.float32)
        unet.save_attn_procs(args.output_dir)


#####################
# DeepFloydIF
#####################

from transformers import T5Tokenizer, T5EncoderModel
from diffusers import DiffusionPipeline

class DeepFloydIF(BaseModel):
    def __init__(self, name="DeepFloydIF", subfolder=None, condition_type="text"):
        self.name = name
        if subfolder is None: subfolder = "unet"
        self.subfolder = subfolder
        self.condition_type = condition_type

    def setup_parts(self, args):

        # Load parts
        self.noise_scheduler = DDPMScheduler.from_pretrained(
             args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.tokenizer = T5Tokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False, revision=args.revision
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
        )

        # Freeze parts we don't want to train
        self.text_encoder.requires_grad_(False)

        # Enable gradient checkpointing
        if args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()


#####################
# NanoDeepFloydIF
#####################

class NanoDeepFloydIF(DeepFloydIF):
    def __init__(self, name="NanoDeepFloydIF", subfolder=None, condition_type="text"):
        self.name = name
        if subfolder is None: subfolder = "unet"
        self.subfolder = subfolder
        self.condition_type = condition_type

    def setup_parts(self, args):

        super().setup_parts(args)

        self.unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=3,
            out_channels=6,
            cross_attention_dim=768,
            down_block_types=['ResnetDownsampleBlock2D', 'SimpleCrossAttnDownBlock2D', 'SimpleCrossAttnDownBlock2D'],
            mid_block_type='UNetMidBlock2DSimpleCrossAttn',
            up_block_types=['SimpleCrossAttnUpBlock2D', 'SimpleCrossAttnUpBlock2D', 'ResnetUpsampleBlock2D'],
            block_out_channels=[192, 256, 256],
            layers_per_block=2,
            encoder_hid_dim=4096,
            attention_head_dim=64,
            addition_embed_type='text',
            resnet_time_scale_shift='scale_shift',
            cross_attention_norm='group_norm'
        )

        if args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()


#####################
# MiniDeepFloydIF
#####################

class MiniDeepFloydIF(DeepFloydIF):
    def __init__(self, name="MiniDeepFloydIF", subfolder=None, condition_type="text"):
        self.name = name
        if subfolder is None: subfolder = "unet"
        self.subfolder = subfolder
        self.condition_type=condition_type

    def setup_parts(self, args):

        super().setup_parts(args)

        self.unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=3,
            out_channels=6,
            cross_attention_dim=768,
            down_block_types=['ResnetDownsampleBlock2D', 'SimpleCrossAttnDownBlock2D', 'SimpleCrossAttnDownBlock2D', 'SimpleCrossAttnDownBlock2D'],
            mid_block_type='UNetMidBlock2DSimpleCrossAttn',
            up_block_types=['SimpleCrossAttnUpBlock2D', 'SimpleCrossAttnUpBlock2D', 'SimpleCrossAttnUpBlock2D', 'ResnetUpsampleBlock2D'],
            block_out_channels=[192, 256, 512, 512],
            layers_per_block=2,
            encoder_hid_dim=4096,
            attention_head_dim=64,
            addition_embed_type='text',
            resnet_time_scale_shift='scale_shift',
            cross_attention_norm='group_norm'
        )

        if args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()


#####################
# DeepFloydIFLoRA
#####################

from diffusers import Transformer2DModel

class DeepFloydIFLoRA(DeepFloydIF):
    def __init__(self, name="DeepFloydIFLoRA", subfolder=None, condition_type="text"):
        self.name = name
        if subfolder is None: subfolder = "unet"
        self.subfolder = subfolder
        self.condition_type = condition_type

    def setup_parts(self, args):
        super().setup_parts(args)

        print("############################ Antes de LoRA (DeepFloyd IF) ############################")
        dummy_noisy_latent = torch.ones((1, 3, 64, 64))
        dummy_timesteps = torch.ones((1,))
        dummy_encoder_hidden_states = torch.ones((1, 512, 4096))
        #summary(self.unet, input_data=[dummy_noisy_latent, dummy_timesteps, dummy_encoder_hidden_states], depth=12)

        # LoRA specific
        self.unet.requires_grad_(False)
        # Set correct lora layers
        lora_attn_procs = {}
        for name in self.unet.attn_processors.keys():
            block_id = None
            if name.endswith("attn1.processor"):
                cross_attention_dim = None
            else:
                cross_attention_dim = self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            print("------------------")
            print(f"{name}\nblock_id:{block_id} hidden_size:{hidden_size} cross_attention_dim:{cross_attention_dim if cross_attention_dim is not None else 'None'}")
            print("------------------")

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

        self.unet.set_attn_processor(lora_attn_procs)

        print("############################ Después de LoRA (DeepFloyd IF) ############################")
        #summary(self.unet, input_data=[dummy_noisy_latent, dummy_timesteps, dummy_encoder_hidden_states], depth=12)

    def get_trainable(self):
        return AttnProcsLayers(self.unet.attn_processors)
    
    def get_pipeline(self, args, accelerator=None, dtype=None):
        pipeline = super().get_pipeline(args, accelerator=accelerator, dtype=dtype)
        # scale 0.0 to not use LoRA weights
        # scale 1.0 to use only LoRA weights
        pipeline.cross_attention_kwargs={"scale": 1.0}
        return pipeline
    
    def save(self, args, accelerator=None, dtype=None):
        unet = self.unet.to(torch.float32)
        unet.save_attn_procs(args.output_dir)
    

#####################
# VQDiffusion
#####################

from diffusers import VQDiffusionPipeline, VQDiffusionScheduler, VQModel

class VQDiffusion(BaseModel):
    def __init__(self, name="VQDiffusion", subfolder=None, condition_type="text"):
        self.name = name
        if subfolder is None: subfolder = "transformer"
        self.subfolder = subfolder
        self.condition_type=condition_type

    def setup_parts(self, args):

        # Load parts
        self.noise_scheduler = VQDiffusionScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False, revision=args.revision
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        self.vqvae = VQModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vqvae", revision=args.non_ema_revision
        )
        self.transformer = Transformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.non_ema_revision
        )

        # Freeze parts we don't want to train
        self.text_encoder.requires_grad_(False)
        self.vqvae.requires_grad_(False)

        # Enable gradient checkpointing
        if args.gradient_checkpointing:
            print("WARNING: Transformer2DModel does not support gradient checkpointing")

    def setup_accelerator(self, accelerator, train_dataloader):

        # Prepare everything with our `accelerator`.
        self.transformer, self.optimizer, new_train_dataloader, self.lr_scheduler = accelerator.prepare(
            self.transformer, self.optimizer, train_dataloader, self.lr_scheduler
        )

        weight_dtype = self.get_weight_dtype(accelerator)

        # Move text_encode and vae to gpu and cast to weight_dtype
        self.text_encoder.to(accelerator.device, dtype=weight_dtype)

        return weight_dtype, new_train_dataloader

    def get_loss(self, args, batch):

        images = batch["pixel_values"]
        input_ids = batch["input_ids"]
        
        batch_size = images.shape[0]

        # get the initial completely masked latents unless the user supplied it
        latents_shape = (batch_size, self.transformer.num_latent_pixels)

        mask_class = self.transformer.num_vector_embeds - 1
        latents = torch.full(latents_shape, mask_class).to("cpu")

        # Sample noise that we'll add to the latents
        noise = torch.randint_like(latents, mask_class)
        if args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += args.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        print(f"latents:{latents.device} noise:{noise.device} timesteps:{timesteps.device}")
        noisy_latents = self.noise_scheduler.q_posterior(latents, noise, timesteps).to(images.device)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(input_ids)[0]

        # Predict the noise residual and compute loss
        model_pred = self.transformer(noisy_latents, encoder_hidden_states=encoder_hidden_states, timestep=timesteps).sample

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        if args.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss

    def get_trainable(self):
        return self.transformer

    def get_pipeline(self, args, accelerator=None, dtype=None):

        if dtype is None:
            dtype = self.get_weight_dtype(accelerator)

        if accelerator is None:
            pipeline = VQDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=self.text_encoder,
                transformer=self.transformer,
                vqvae=self.vqvae,
                revision=args.revision,
                #torch_dtype=dtype
            )
        else:
            pipeline = VQDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(self.text_encoder),
                tokenizer=self.tokenizer,
                transformer=accelerator.unwrap_model(self.transformer),
                vqvae=accelerator.unwrap_model(self.vqvae),
                revision=args.revision,
                torch_dtype=dtype
            )
        
        pipeline.safety_checker = None

        return pipeline
    
    def load_trainable_model(self, input_dir):
        return Transformer2DModel.from_pretrained(input_dir, subfolder=self.subfolder)
    
#####################
# MyDiffusion
#####################

from pipelines.custom import MyDiffusionPipeline

class MyDiffusion(BaseModel):
    def __init__(self, name="MyDiffusion", subfolder=None, condition_type="text", block_out_channels=None):
        self.name = name
        if subfolder is None: subfolder = "unet"
        self.subfolder = subfolder
        self.condition_type = condition_type
        self.block_out_channels = block_out_channels

    def setup_parts(self, args):

        # Load parts
        self.noise_scheduler = DDPMScheduler.from_pretrained(
             args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False, revision=args.revision
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        self.unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            center_input_sample=False,
            flip_sin_to_cos=True,
            freq_shift=0,
            down_block_types=['CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'],
            mid_block_type='UNetMidBlock2DCrossAttn',
            up_block_types=['UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'],
            only_cross_attention=False,
            block_out_channels=self.block_out_channels,
            layers_per_block=3,
            downsample_padding=1,
            mid_block_scale_factor=1,
            act_fn='silu',
            norm_num_groups=32,
            norm_eps=1e-05,
            cross_attention_dim=1024,
            encoder_hid_dim=None,
            attention_head_dim=[5, 10, 20, 20],
            dual_cross_attention=False,
            use_linear_projection=True,
            class_embed_type=None,
            addition_embed_type=None,
            num_class_embeds=None,
            upcast_attention=False,
            resnet_time_scale_shift='default',
            resnet_skip_time_act=False,
            resnet_out_scale_factor=1.0,
            time_embedding_type='positional',
            time_embedding_dim=None,
            time_embedding_act_fn=None,
            timestep_post_act=None,
            time_cond_proj_dim=None,
            conv_in_kernel=3,
            conv_out_kernel=3,
            projection_class_embeddings_input_dim=None,
            class_embeddings_concat=False,
            mid_block_only_cross_attention=None,
            cross_attention_norm=None,
            addition_embed_type_num_heads=64
        )

        # Freeze parts we don't want to train
        self.text_encoder.requires_grad_(False)

        # Enable gradient checkpointing
        if hasattr(args, 'gradient_checkpointing'):
            if args.gradient_checkpointing: self.unet.enable_gradient_checkpointing()

        print("Counting parameters...")
        param_info = {
            'text_encoder':{
                'total':sum(p.numel() for p in self.text_encoder.parameters()),
                'trainable':sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
            },
            'unet':{
                'total':sum(p.numel() for p in self.unet.parameters()),
                'trainable':sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
            },
        }
        pprint(param_info,depth=4)
    
    def get_pipeline(self, args, accelerator=None, dtype=None, overwrite_current_weights=True):

        if dtype is None:
            dtype = self.get_weight_dtype(accelerator)

        if overwrite_current_weights:
            pipeline = MyDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                revision=args.revision,
            )
        elif accelerator is None:
            pipeline = MyDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=self.text_encoder,
                unet=self.unet,
                revision=args.revision,
                #torch_dtype=dtype
            )
        else:
            pipeline = MyDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(self.text_encoder),
                tokenizer=self.tokenizer,
                unet=accelerator.unwrap_model(self.unet),
                revision=args.revision,
                torch_dtype=dtype
            )
        
        pipeline.safety_checker = None

        return pipeline
    
#####################
# DiT
#####################

from diffusers import DiTPipeline, DPMSolverMultistepScheduler

class DiT(BaseModel):
    def __init__(self, name="DiT", subfolder=None, condition_type="class"):
        self.name = name
        if subfolder is None: subfolder = "transformer"
        self.subfolder = subfolder
        self.condition_type = condition_type

    def setup_parts(self, args, num_classes=16):
        
        # Load parts
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae"
        )

        pprint(self.vae.config)
        assert False, "STOP! Hay que reducir sample_size de 256 a 64 pero eso implicaría reentrenar el VAE"

        self.transformer = Transformer2DModel(
            num_attention_heads=16,
            attention_head_dim=72,
            in_channels=4,
            out_channels=8,
            num_layers=28,
            dropout=0.0,
            norm_num_groups=32,
            cross_attention_dim=None,
            attention_bias=True,
            sample_size=32,
            num_vector_embeds=None,
            patch_size=2,
            activation_fn='gelu-approximate',
            num_embeds_ada_norm=num_classes,
            use_linear_projection=False,
            only_cross_attention=False,
            upcast_attention=False,
            norm_type='ada_norm_zero',
            norm_elementwise_affine=False
        )

        # Freeze parts we don't want to train
        self.vae.requires_grad_(False)

        # Enable gradient checkpointing
        if hasattr(args, 'gradient_checkpointing'):
            if args.gradient_checkpointing: self.transformer.enable_gradient_checkpointing()

    def setup_accelerator(self, accelerator, train_dataloader):

        # Prepare everything with our `accelerator`.
        self.transformer, self.optimizer, new_train_dataloader, self.lr_scheduler = accelerator.prepare(
            self.transformer, self.optimizer, train_dataloader, self.lr_scheduler
        )

        weight_dtype = self.get_weight_dtype(accelerator)

        # Move text_encode and vae to gpu and cast to weight_dtype
        self.vae.to(accelerator.device, dtype=weight_dtype)

        return weight_dtype, new_train_dataloader
    
    def setup_noaccelerator(self, weight_dtype="fp32", device="cpu"):
        self.transformer = self.transformer.to(device, dtype=weight_dtype)
        self.vae = self.vae.to(device, dtype=weight_dtype)

    def get_loss(self, args, batch):

        images = batch["pixel_values"]
        input_classes= batch["input_classes"]

        # Convert images to latent space
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += args.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )

        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            # StableDiff2: AttributeError: 'EulerDiscreteScheduler' object has no attribute 'get_velocity'
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        
        #print(f"noisy_latents:{noisy_latents.shape} \n \
        #      timesteps:{timesteps.shape} \n \
        #      input_ids:{input_ids.shape} \n \
        #      encoder_hidden_state:{encoder_hidden_states.shape}")

        # Predict the noise residual and compute loss
        model_pred = self.transformer(noisy_latents, timestep=timesteps, class_labels=input_classes).sample

        if args.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss

    def get_trainable(self):
        return self.transformer
    
    def get_pipeline(self, args, accelerator=None, dtype=None):

        if dtype is None:
            dtype = self.get_weight_dtype(accelerator)

        if accelerator is None:
            pipeline = DiTPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=self.vae,
                transformer=self.transformer,
                #torch_dtype=dtype
            )
        else:
            pipeline = DiTPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=accelerator.unwrap_model(self.vae),
                transformer=accelerator.unwrap_model(self.transformer),
                torch_dtype=dtype
            )
        
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    def load_trainable_model(self, input_dir):
        return Transformer2DModel.from_pretrained(input_dir, subfolder=self.subfolder)


#####################
# DiTnoVAE
#####################

class DiTnoVAE(BaseModel):
    def __init__(self, name="DiTnoVAE", subfolder=None, condition_type="class"):
        self.name = name
        if subfolder is None: subfolder = "transformer"
        self.subfolder = subfolder
        self.condition_type = condition_type

    def setup_parts(self, args, num_classes=16):
        
        # Load parts
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.transformer = Transformer2DModel(
            num_attention_heads=16,
            attention_head_dim=72,
            in_channels=3,
            out_channels=3,
            num_layers=28,
            dropout=0.0,
            norm_num_groups=32,
            cross_attention_dim=None,
            attention_bias=True,
            sample_size=64,
            num_vector_embeds=None,
            patch_size=2,
            activation_fn='gelu-approximate',
            num_embeds_ada_norm=num_classes,
            use_linear_projection=False,
            only_cross_attention=False,
            upcast_attention=False,
            norm_type='ada_norm_zero',
            norm_elementwise_affine=False
        )

        #File "/home/jjuagon/jjuagon_env/lib/python3.9/site-packages/diffusers/models/attention.py", line 335, in forward
        #x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        #RuntimeError: The size of tensor a (1152) must match the size of tensor b (6912) at non-singleton dimension 3
        print("[TO SOLVE] RuntimeError: The size of tensor a (1152) must match the size of tensor b (6912) at non-singleton dimension 3...")
        raise NotImplementedError

        # Enable gradient checkpointing
        if hasattr(args, 'gradient_checkpointing'):
            if args.gradient_checkpointing: self.transformer.enable_gradient_checkpointing()

    def setup_accelerator(self, accelerator, train_dataloader):

        # Prepare everything with our `accelerator`.
        self.transformer, self.optimizer, new_train_dataloader, self.lr_scheduler = accelerator.prepare(
            self.transformer, self.optimizer, train_dataloader, self.lr_scheduler
        )

        weight_dtype = self.get_weight_dtype(accelerator)

        return weight_dtype, new_train_dataloader
    
    def setup_noaccelerator(self, weight_dtype="fp32", device="cpu"):
        self.transformer = self.transformer.to(device, dtype=weight_dtype)

    def get_loss(self, args, batch):

        images = batch["pixel_values"]
        input_classes= batch["input_classes"]

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(images)
        if args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += args.noise_offset * torch.randn(
                (images.shape[0], images.shape[1], 1, 1), device=images.device
            )

        bsz = images.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=images.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(images, noise, timesteps)

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            # StableDiff2: AttributeError: 'EulerDiscreteScheduler' object has no attribute 'get_velocity'
            target = self.noise_scheduler.get_velocity(images, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        
        #print(f"noisy_latents:{noisy_latents.shape} \n \
        #      timesteps:{timesteps.shape} \n \
        #      input_ids:{input_ids.shape} \n \
        #      encoder_hidden_state:{encoder_hidden_states.shape}")

        # Predict the noise residual and compute loss
        model_pred = self.transformer(noisy_latents, timestep=timesteps, class_labels=input_classes).sample

        if args.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss

    def get_trainable(self):
        return self.transformer
    
    def get_pipeline(self, args, accelerator=None, dtype=None):

        raise NotImplementedError

        if dtype is None:
            dtype = self.get_weight_dtype(accelerator)

        if accelerator is None:
            pipeline = DiTPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=self.vae,
                transformer=self.transformer,
                #torch_dtype=dtype
            )
        else:
            pipeline = DiTPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=accelerator.unwrap_model(self.vae),
                transformer=accelerator.unwrap_model(self.transformer),
                torch_dtype=dtype
            )
        
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    def load_trainable_model(self, input_dir):
        return Transformer2DModel.from_pretrained(input_dir, subfolder=self.subfolder)
    

#####################
# DDPM
#####################

import types
from ddpm_torch import UNet as CustomUNet
from ddpm_torch import CondDDPMPipeline

class DDPM(BaseModel):
    def __init__(self, name="DDPM", subfolder=None, condition_type="class"):
        self.name = name
        if subfolder is None: subfolder = "unet"
        self.subfolder = subfolder
        self.condition_type = condition_type

        self.in_channels = 3
        self.hid_channels = 128
        self.out_channels = 3
        self.ch_multipliers = (1,2,4,4)
        self.num_res_blocks = 2
        self.apply_attn = (False,True,True,True)

        self.t_embed_dim = None
        self.c_embed_dim = None
        self.c_in_dim = 16
        
        # for pipeline
        self.default_sample_size = 64
        
        self.unet = None
        self.noise_scheduler = None

    def setup_parts(self, args):
        
        if hasattr(args,'mixed_precision') and args.mixed_precision != "no":
            raise NotImplementedError

        # Load parts
        self.noise_scheduler = DDPMScheduler.from_pretrained(
             args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.unet = self.load_trainable_model(None)

        # Enable gradient checkpointing
        if hasattr(args, 'gradient_checkpointing') and args.gradient_checkpointing:
            raise NotImplementedError
        
        print("Counting parameters...")
        param_info = {
            'unet':{
                'total':sum(p.numel() for p in self.unet.parameters()),
                'trainable':sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
            },
        }
        pprint(param_info,depth=4)
    
    def setup_accelerator(self, accelerator, train_dataloader):
        raise NotImplementedError
    
    def setup_noaccelerator(self, weight_dtype="fp32", device="cpu"):
        self.unet = self.unet.to(device, dtype=weight_dtype)

    def get_loss(self, args, batch):

        images = batch["pixel_values"]

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(images)
        if args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += args.noise_offset * torch.randn(
                (images.shape[0], images.shape[1], 1, 1), device=images.device
            )

        bsz = images.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=images.device)
        timesteps = timesteps.long()

        # Add noise to the images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(images, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        c = batch["input_classes"]
        model_pred = self.unet(noisy_images, timesteps, c)

        #print(f"model_pred.shape:{model_pred.shape}\ntarget.shape:{target.shape}")

        if args.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss

    def get_pipeline(self, args, accelerator=None, dtype=None, unet_weights=None):

        if dtype is None:
            dtype = self.get_weight_dtype(accelerator)

        if self.unet is None or unet_weights is not None:
            self.unet = self.load_trainable_model(unet_weights=unet_weights)

        if accelerator is None:
            pipeline = CondDDPMPipeline(
                unet=self.unet,
                scheduler=DDPMScheduler.from_pretrained(
                     args.pretrained_model_name_or_path, subfolder="scheduler"
                )
            )
        else:
            raise NotImplementedError

        return pipeline
    
    def load_trainable_model(self, unet_weights=None):

        model = CustomUNet(
            self.in_channels,
            self.hid_channels,
            self.out_channels,
            self.ch_multipliers,
            self.num_res_blocks,
            self.apply_attn,
            t_embed_dim=self.t_embed_dim,
            c_embed_dim=self.c_embed_dim,
            c_in_dim=self.c_in_dim,
            num_groups=32,
            drop_rate=0.0,
            resample_with_conv=True
        )

        if unet_weights is not None:
            model.load_state_dict(torch.load(unet_weights), strict=False)

        return model
    
    def save(self, args, accelerator=None, dtype=None):
        # get pipeline
        pipeline = self.get_pipeline(args, accelerator=accelerator, dtype=dtype)
        #pipeline.save_pretrained(args.output_dir)
        torch.save(pipeline.unet.state_dict(), f"{args.output_dir}/unet.pth.tar")