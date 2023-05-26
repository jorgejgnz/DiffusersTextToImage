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

from models import NanoStableDiffusion, MiniStableDiffusion, StableDiffusionLoRA

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

prompts = [
    "with Hair, without Eyeglasses, Female, Smiling",
    "with Hair, without Eyeglasses, Male, Serious",
    "without Hair, with Eyeglasses, Male, Smiling",
    "with Hair, with Eyeglasses, Female, Serious",
]


experiment = "StableDiffusionLoRA_batch512_lr-cosine-1e-04_epochs20_snrgamma5.0"
save_path = "samples"

args = dict2obj({
    'pretrained_model_name_or_path':"bguisard/stable-diffusion-nano-2-1",
    'revision':None,
    'non_ema_revision':None,
    'gradient_checkpointing':False
})

device = "cuda" if torch.cuda.is_available() else "cpu"

# get pipeline from pretrained
some_model = StableDiffusionLoRA()
some_model.setup_parts(args)
pipeline = some_model.get_pipeline(args, dtype=torch.float32)
pipeline.unet.load_attn_procs(f"generic_finetunings_4atts/{experiment}")
pipeline = pipeline.to(device)

# inference
kwargs = {
    'num_inference_steps':100,
    'guidance_scale':10.0,
    'height':64,
    'width':64
}

with torch.autocast("cuda"):
    input_ids = []
    for p in prompts: input_ids.append(tokenize_caption(some_model, p))
    input_ids = torch.stack(input_ids)
    print(input_ids.shape)
    encoder_hidden_states = pipeline.text_encoder(input_ids.to(device))[0]
    images = pipeline(prompt_embeds=encoder_hidden_states, generator=None, **kwargs).images

# save image
save_path = f"samples/{experiment}/inference-steps-{kwargs['num_inference_steps']}_guidance-scale{kwargs['guidance_scale']}"
if not os.path.exists(save_path):
    os.makedirs(save_path)

for idx, p in enumerate(prompts):
    images[idx].save(f"{save_path}/{prompts[idx]}.jpg")
