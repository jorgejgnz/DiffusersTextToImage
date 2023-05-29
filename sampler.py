import argparse
import logging
import math
import os
import sys
import time
import random
from pathlib import Path
import warnings

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torch.utils.data import IterableDataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers

from models import NanoStableDiffusion, MiniStableDiffusion, StableDiffusionLoRA, MyDiffusion, DDPM

# importing the module
import json
  
# declaringa a class
class obj:   
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)
  
def dict2obj(dict1):
    # using json.loads method and passing json.dumps
    # method and custom object hook as arguments
    return json.loads(json.dumps(dict1), object_hook=obj)

def tokenize_caption(some_model, caption):
    inputs = some_model.tokenizer(
        [caption], max_length=some_model.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids[0]

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--main_folder",
        type=str,
        default="text"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="text"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="text"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="text"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no"
    )
    parser.add_argument(
        "--unet_weights",
        type=str,
        default=None
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    save_path = "samples"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # get pipeline from pretrained
    if args.model_type == "StableDiffusionLoRA":
        some_model = StableDiffusionLoRA()
    elif args.model_type == "MyDiffusion":
        block_out_channels = [256,512,768,768]
        some_model = MyDiffusion(block_out_channels=block_out_channels)
    elif args.model_type == "DDPM":
        some_model = DDPM()

    some_model.setup_parts(args)
  
    if args.model_type == "StableDiffusionLoRA":
        pipeline = some_model.get_pipeline(args, dtype=torch.float32)
        pipeline.unet.load_attn_procs(f"{args.main_folder}/{args.experiment}")
    elif args.model_type == "MyDiffusion":
        pipeline = some_model.get_pipeline(args, dtype=torch.float32, overwrite_current_weights=True)
    elif args.model_type == "DDPM":
        pipeline = some_model.get_pipeline(args, dtype=torch.float32, unet_weights=args.unet_weights)

    pipeline = pipeline.to(device)

    # inference
    kwargs = {
        'num_inference_steps':args.num_inference_steps,
        'guidance_scale':args.guidance_scale,
        'height':64,
        'width':64
    }

    # save image
    save_path = f"{args.output_dir}/{args.experiment}/inference-steps-{kwargs['num_inference_steps']}_guidance-scale{kwargs['guidance_scale']}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if some_model.condition_type == "text":
        prompts = [
            "with Hair, without Eyeglasses, Female, Smiling",
            "with Hair, without Eyeglasses, Female, Serious",
            "with Hair, without Eyeglasses, Male, Smiling",
            "with Hair, without Eyeglasses, Male, Serious",
            "with Hair, with Eyeglasses, Female, Smiling",
            "with Hair, with Eyeglasses, Female, Serious",
            "with Hair, with Eyeglasses, Male, Smiling",
            "with Hair, with Eyeglasses, Male, Serious",
            "without Hair, without Eyeglasses, Female, Smiling",
            "without Hair, without Eyeglasses, Female, Serious",
            "without Hair, without Eyeglasses, Male, Smiling",
            "without Hair, without Eyeglasses, Male, Serious",
            "without Hair, with Eyeglasses, Female, Smiling",
            "without Hair, with Eyeglasses, Female, Serious",
            "without Hair, with Eyeglasses, Male, Smiling",
            "without Hair, with Eyeglasses, Male, Serious",
        ]
    elif some_model.condition_type == "class":
        num_classes = 16
        prompts = range(num_classes)

    for p in prompts:

        with torch.autocast("cuda"):
            if some_model.condition_type == "text":
                input_ids = [tokenize_caption(some_model, p)]
                input_ids = torch.stack(input_ids)
                print(input_ids.shape)
                encoder_hidden_states = pipeline.text_encoder(input_ids.to(device))[0]
                images = pipeline(prompt_embeds=encoder_hidden_states, generator=None, **kwargs).images
            elif some_model.condition_type == "class":
                input_classes = torch.tensor([p])
                input_classes = F.one_hot(input_classes, num_classes=num_classes).to(device)
                images = pipeline(input_classes=input_classes, generator=None, **kwargs).images
            else:
                raise NotImplementedError

        images[0].save(f"{save_path}/{p}.jpg")

if __name__ == "__main__":
    main()