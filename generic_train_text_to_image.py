#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import sys
import time
import random
from pathlib import Path
import warnings
import shutil

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torch.utils.data import IterableDataset
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available

from PIL import Image
import albumentations as A
from matplotlib import pyplot as plt

import inspect
from functools import partial
from pprint import pprint

from datetime import datetime
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional.multimodal import clip_score

from mtcnn_cv2 import MTCNN

import torch.nn as nn
from torchvision import models

from models import BaseModel, StableDiffusion, NanoStableDiffusion, Nano21StableDiffusion, MiniStableDiffusion, StableDiffusionLoRA
from models import DeepFloydIF, NanoDeepFloydIF, MiniDeepFloydIF, DeepFloydIFLoRA
from models import VQDiffusion, MyDiffusion, DiT, DiTnoVAE, DDPM

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0.dev0")

class FaceAttEstimator(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        vgg = models.vgg16(weights=None)
        vgg.classifier[6] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=vgg.classifier[6].in_features, out_features=n_classes)
        )
        self.base_model = vgg
        self.sigm = nn.Sigmoid()
 
    def forward(self, x):
        return self.sigm(self.base_model(x))

class ActuallyIterableDataset(IterableDataset):
    def __init__(self, dataset, length, max=-1):
        self.src = dataset
        self.len = length
        self.max = max

    def __iter__(self):
        c = 0
        for example in self.src:
            c += 1
            if self.max >= 0 and c < self.max:
                yield example
            elif self.max < 0:
                yield example
            else:
                break

    def __len__(self):
        if self.max >= 0: return self.max
        else: return self.len
    
    def shuffle(self, buffer_size=10000, seed=42):
        self.src = self.src.shuffle(buffer_size=buffer_size,seed=seed)

def show_examples(args, dataset, transformations):
    examples = []
    for i in range(32):  # Mostrar solo las primeras 8 imágenes
        example = next(dataset)
        image = example[args.image_column].convert("RGB")
        transformed_image = transformations(image=np.array(image))['image']
        examples.append(transformed_image) 
    fig, axs = plt.subplots(4, 8, figsize=(12, 6))
    axs = axs.flatten()
    for i in range(len(examples)):
        axs[i].imshow(examples[i])
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/dataaugmentation.png")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        required=True,
        help="Type of available model to use.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the data (with subfolder train, val and test). Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--class_column",
        type=str,
        default="class",
        help="The column of the dataset containing a class or a list of classes.",
    )
    parser.add_argument(
        "--attributes_column",
        type=str,
        default="attributes",
        help="The column of the dataset containing a binary number (as string) that represents binary attributes.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--data_augmentation",
        default=False,
        action="store_true",
        help=(
            "Whether to use data augmentation to train dataset."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=100
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="For how many batches should gradients be accumulated.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--block_out_channels",
        type=str,
        default="256,512,768,768",
        help="Set max conv channels when possible",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--checkpointing_epochs",
        type=int,
        default=5,
        help=(
            "Save a checkpoint of the training state every X epochs. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--noise_offset", type=float, default=0, help="The scale of noise offset."
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--num_validation_samplings",
        type=int,
        default=5,
        help="How many samples generate on each validation.",
    )
    parser.add_argument(
        "--attestimator_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pth.tar file of face attribute estimator.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    ## new!
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help=(
            "Print parameters of trainable model"
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(t_images, prompts, to_uint8=True, to_255=True):
    t_images = t_images.clone().detach()
    if to_255: t_images *= 255
    if to_uint8: t_images = t_images.to(torch.uint8)
    clip_score = clip_score_fn(t_images, prompts).detach()
    return float(clip_score)

def binary_string_to_list(bin_str, as_tensor=False):
    result = []
    for bit in bin_str:
        if bit == '0':
            result.append(0.0)
        elif bit == '1':
            result.append(1.0)
    if as_tensor: result = torch.tensor(result)
    return result

def decode_image(raw_img, to_numpy=True, to_255=False):
    image = (raw_img / 2 + 0.5).clamp(0, 1)
    if to_numpy:
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    if to_255:
        image *= 255
        if to_numpy: image = np.round(image)
        else: image = torch.round(image)
    return image

def validation(some_model, device, dtype, args, val_dataloader, unique_name, metric_models, max_saves=25, loss=-1.0): # num_generations debe ser al menos 500 para ser significativo

        print("Performing validation...")

        kwargs = {
            'num_inference_steps':50, #25,
            'guidance_scale':7.5, #7.5,
            'height':64,
            'width':64
        }

        # crea una carpeta para este experimento y esta validación
        save_path = os.path.join(*[args.output_dir, 'validations', unique_name])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # carga la pipeline
        pipeline = some_model.get_pipeline(args, dtype=dtype)
        pipeline = pipeline.to(device)
        pipeline.set_progress_bar_config(disable=False)

        if args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=device).manual_seed(args.seed)

        sampled_images = 0

        avg_clipscore = 0.0
        avg_mtcnnscore = 0.0
        avg_attestimator = 0.0

        fid = metric_models['fid']
        mtcnn = metric_models['mtcnn']
        attestimator = metric_models['attestimator']

        total_samples = args.num_validation_samplings

        pbar = tqdm(total=total_samples-1)
        for idx, batch in enumerate(val_dataloader):
            if idx == 0:
                bs = batch['pixel_values'].shape[0]
                assert bs == 1, f"val_dataloader expected to have batch_size=1 instead of {bs}"

            with torch.autocast("cuda"):
                if some_model.condition_type == "text":
                    encoder_hidden_states = pipeline.text_encoder(batch['input_ids'].to(device))[0]
                    image = pipeline(prompt_embeds=encoder_hidden_states, generator=generator, **kwargs).images[0]
                elif some_model.condition_type == "class":
                    image = pipeline(input_classes=batch['input_classes'], generator=generator, **kwargs).images[0]
                else:
                    raise NotImplementedError

            sampled_images += 1
            prompt = [batch['input_texts'][0]]
            rgb_image = image.convert("RGB")

            t_image = transforms.PILToTensor()(rgb_image)
            t_image = torch.unsqueeze(t_image,0)
            t_fake = t_image.to("cpu", dtype=torch.float32)

            t_real = batch['pixel_values'].to("cpu", dtype=torch.float32)
            t_real = decode_image(t_real, to_numpy=False, to_255=True)

            # FID
            print("Updating FID...")
            fid.update(t_real, real=True)
            fid.update(t_fake, real=False)

            # CLIPScore
            print("Computing CLIPScore...")
            
            print(t_fake)
            assert False, "STOP!"
            avg_clipscore += calculate_clip_score(t_fake, prompt, to_uint8=True, to_255=True)

            # MTCNN
            print("Searching faces with MTCNN...")
            faces = mtcnn.detect_faces(np.array(rgb_image))
            if len(faces)>0:
                avg_mtcnnscore += faces[0]['confidence']
            else:
                avg_mtcnnscore += 0.0

            # Attribute estimator
            print("Estimating attributes...")
            attestimator.eval()
            with torch.no_grad():
                estimated_atts = attestimator(t_fake)[0]
            estimated_atts = (estimated_atts > 0.5).float() # to 1 or 0
            target_atts = batch['input_attributes'][0]
            # Calculate precision only for some attributes ######################################
            attributes = [
                (4,"Bald","with Hair"),
                (15,"Eyeglasses","without Eyeglasses"),
                (20,"Male","Female"),
                (31,"Smiling","Serious")
            ]
            # [40] -> [4]
            num_atts = len(attributes)
            target_atts = torch.tensor([target_atts[att[0]] for att in attributes])
            estimated_atts = torch.tensor([estimated_atts[att[0]] for att in attributes])
            attestimator_acc = (estimated_atts == target_atts).sum().item() / num_atts
            avg_attestimator += attestimator_acc

            # Save
            if sampled_images < min(args.num_validation_samplings, max_saves) + 1:
                img_file = f"{idx}.jpg"
                image.save(f"{save_path}/{img_file}")
                with open(f"{save_path}/prompts.txt", 'a') as file:
                    file.write(f"{img_file} '{prompt}'\n")

            pbar.update(idx)

            if idx+1 >= args.num_validation_samplings:
                break
        pbar.close()

        # actualiza calidad de imagen (FID)
        print("Computing metrics...")
        avg_fid = float(fid.compute())
        avg_clipscore /= sampled_images
        avg_mtcnnscore /= sampled_images
        avg_attestimator /= sampled_images

        # escribe en un fichero nueva linea con (id, fecha-hora, calidad de imagen, CLIPScore)
        with open(os.path.join(*[args.output_dir, 'validations', 'log.txt']), 'a') as file:
            datetime_str = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
            file.write(f"(fields:name&date&inference_steps&guidance_sacle&fid&clipscore&mtcnnscore&attestimator&loss)\t{unique_name}\t{datetime_str}\t{kwargs['num_inference_steps']}\t{kwargs['guidance_scale']}\t{avg_fid}\t{avg_clipscore}\t{avg_mtcnnscore}\t{avg_attestimator}\t{loss}\n")

        # libera memoria
        del pipeline
        torch.cuda.empty_cache()

        print(f"Validation completed! Check results at {save_path}")

def save_model(model, output_dir):
    model.save_pretrained(os.path.join(output_dir, model.subfolder))

def load_model(model, input_dir):
    # load diffusers style into model
    load_model = model.load_trainable_model(input_dir)
    model.register_to_config(**load_model.config)
    model.load_state_dict(load_model.state_dict())
    del load_model

def main():
    args = parse_args()

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.push_to_hub:
        repo_id = create_repo(
            repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        ).repo_id

    # generalization!

    if args.model_type == "StableDiffusion":
        some_model = StableDiffusion()
    elif args.model_type == "NanoStableDiffusion":
        some_model = NanoStableDiffusion()
    elif args.model_type == "Nano21StableDiffusion":
        some_model = Nano21StableDiffusion()
    elif args.model_type == "MiniStableDiffusion":
        some_model = MiniStableDiffusion()
    elif args.model_type == "StableDiffusionLoRA":
        some_model = StableDiffusionLoRA()
    elif args.model_type == "DeepFloydIF":
        some_model = DeepFloydIF()
    elif args.model_type == "NanoDeepFloydIF":
        some_model = NanoDeepFloydIF()
    elif args.model_type == "MiniDeepFloydIF":
        some_model = MiniDeepFloydIF()
    elif args.model_type == "DeepFloydIFLoRA":
        some_model = DeepFloydIFLoRA()
    elif args.model_type == "VQDiffusion":
        some_model = VQDiffusion()
    elif args.model_type == "MyDiffusion":
        args.block_out_channels = [int(c) for c in args.block_out_channels.split(',')]
        some_model = MyDiffusion(block_out_channels=args.block_out_channels)
    elif args.model_type == "DiT":
        some_model = DiT()
    elif args.model_type == "DiTnoVAE":
        some_model = DiTnoVAE()
    elif args.model_type == "DDPM":
        some_model = DDPM()
    else:
        raise NotImplementedError

    some_model.setup_parts(args)  

    print("Setting device and dtype!")
    main_dtype = torch.float32
    autocast_type = torch.float32
    if args.mixed_precision == "fp16":
        autocast_type = torch.float16
    elif args.mixed_precision == "bf16":
        autocast_type = torch.bfloat16

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    some_model.setup_noaccelerator(weight_dtype=main_dtype, device=device)                         

    # debug
    if args.debug:
        pprint(some_model.get_trainable(depth=1))

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        print("Allowing TF32!")
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    some_model.setup_optimizer(args)

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    num_classes = 16

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        print("Loading dataset!")
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            streaming=True
        )
        raise NotImplementedError
    else:
        print("Loading dataset from folder!")
        data_dir = os.path.abspath(args.data_dir)
        dataset = load_dataset(
            "imagefolder",
            data_files={
                'train': os.path.join(*[data_dir, "train", "**"]),
                'validation': os.path.join(*[data_dir, "validation", "**"])
            },
            cache_dir=args.cache_dir,
            streaming=True
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = list(next(iter(dataset["train"])).keys())

    # 6. Get the column names for input/target.
    image_column = args.image_column
    if image_column not in column_names:
        raise ValueError(
            f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
        )
    caption_column = args.caption_column
    if caption_column not in column_names:
        raise ValueError(
            f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
        )
    class_column = args.class_column
    if class_column not in column_names:
        raise ValueError(
            f"--class_column' value '{args.class_column}' needs to be one of: {', '.join(column_names)}"
        )
    attributes_column = args.attributes_column
    if attributes_column not in column_names:
        raise ValueError(
            f"--attributes_column' value '{args.attributes_column}' needs to be one of: {', '.join(column_names)}"
        )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_caption(caption):
        if hasattr(some_model,'tokenizer'):
            inputs = some_model.tokenizer(
                [caption],
                max_length=some_model.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return inputs.input_ids[0]
        else:
            #print(f"[WARNING!] {some_model.name} has not tokenizer. Batch[input_ids] will be zeroes")
            return torch.zeros((1))	

    # Preprocessing the datasets.
    albumentations_transforms = A.Compose(
        [
            A.ColorJitter(always_apply=False, p=1.0, brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.02, 0.01)),
            A.ShiftScaleRotate(always_apply=False, p=1.0, shift_limit_x=(-0.06, 0.06), shift_limit_y=(-0.06, 0.06), scale_limit=(-0.040000000000000036, 0.040000000000000036), rotate_limit=(-21, 19), interpolation=2, border_mode=4, value=(0, 0, 0), mask_value=None, rotate_method='largest_box')
        ]
    )
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution), #?
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(example):
        np_image=np.array(example[image_column].convert("RGB")) # PIL -> np
        if args.data_augmentation: np_image = albumentations_transforms(image=np_image)['image'] # np -> np
        example["pixel_values"] = train_transforms(Image.fromarray(np_image)) # np -> PIL
        example["input_ids"] = tokenize_caption(example[caption_column])
        example["input_texts"] = example[caption_column]
        example["input_classes"] = example[class_column]
        example["attributes"] = binary_string_to_list(example[attributes_column], as_tensor=True)
        return example
    
    def preprocess_val(example):
        example["pixel_values"] = val_transforms(example[image_column].convert("RGB"))
        example["input_ids"] = tokenize_caption(example[caption_column])
        example["input_texts"] = example[caption_column]
        example["input_classes"] = example[class_column]
        example["attributes"] = binary_string_to_list(example[attributes_column], as_tensor=True)
        return example

    print("Setting data transforms!")
    # Set the training transforms
    print(f"dataset size: {sys.getsizeof(dataset)}")
    train_dataset = dataset["train"].map(preprocess_train)
    val_dataset = dataset["validation"].map(preprocess_val)

    # num_samples = images - metadata.jsonl
    num_train_samples = len(os.listdir(os.path.join(data_dir,'train'))) - 1
    num_val_samples = len(os.listdir(os.path.join(data_dir,'validation'))) - 1

    train_dataset = ActuallyIterableDataset(train_dataset, num_train_samples)#, max=1024)
    val_dataset = ActuallyIterableDataset(val_dataset, num_val_samples)

    # Show data augmentation
    demo_dataset = ActuallyIterableDataset(dataset["train"], num_train_samples)
    show_examples(args, iter(demo_dataset), albumentations_transforms)
    del demo_dataset

    def collate_fn(batch):
        pixel_values = torch.stack([example["pixel_values"] for example in batch])
        pixel_values = pixel_values.to(device, memory_format=torch.contiguous_format)
        input_ids = torch.stack([example["input_ids"] for example in batch]).to(device)
        input_texts = [example["input_texts"] for example in batch]
        input_classes = torch.tensor([example["input_classes"] for example in batch])
        input_classes = F.one_hot(input_classes, num_classes=num_classes).to(device)
        input_attributes = torch.stack([example["attributes"] for example in batch])
        input_attributes = input_attributes.to(device)
        return {"pixel_values": pixel_values, "input_ids": input_ids, "input_texts": input_texts, "input_classes": input_classes, "input_attributes": input_attributes}

    print("Loading dataloaders!")

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False, # shuffled every epoch
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=0
    )

    train_dataloader.dataset.shuffle(buffer_size=10000, seed=42)

    args.max_train_steps = len(train_dataloader)
    print(f"[WARNING] args.max_train_steps is IGNORED, len(train_dataloader) will be used instead")

    print("Setting up scheduler!")
    some_model.setup_scheduler(args)

    print("Setting up metric models!")      
    fid = FrechetInceptionDistance(normalize=True)
    mtcnn = MTCNN(min_face_size=20, steps_threshold=[0.1, 0.1, 0.1])
    attestimator = FaceAttEstimator(40)
    attestimator.load_state_dict(torch.load(args.attestimator_checkpoint)['state_dict'])

    metric_models = {
        'fid':fid,
        'mtcnn':mtcnn,
        'attestimator':attestimator
    }

    # Train!
    total_batch_size = args.train_batch_size
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Steps/epoch = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            print(f"Resuming from checkpoint {path}")
            load_model(some_model, os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step
            first_epoch = global_step // args.max_train_steps
            resume_step = resume_global_step % args.max_train_steps

    trainable = some_model.get_trainable()

    scaler = torch.cuda.amp.GradScaler()

    #validation(some_model, device, main_dtype, args, val_dataloader, f"{some_model.name}_epochNone", metric_models)
    #some_model.save(args, dtype=main_dtype)

    for epoch in range(first_epoch, args.num_train_epochs):

        print(f"Starting epoch {epoch}!")

        trainable.train()
        epoch_loss = 0.0

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(0, args.max_train_steps))
        progress_bar.set_description("Steps")

        for step, batch in enumerate(train_dataloader):

            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                progress_bar.update(1)
                continue

            # Convert images to dtype
            batch["pixel_values"] = batch["pixel_values"].to(main_dtype)
            batch["input_classes"] = batch["input_classes"].to(main_dtype)

            if args.mixed_precision == "no":
                # Backprop step
                loss = some_model.get_loss(args, batch)
                loss = loss / args.gradient_accumulation_steps 
                loss.backward()
                
            else:
                # Backprop step
                with torch.autocast(device_type=device, dtype=autocast_type):
                    loss = some_model.get_loss(args, batch)
                    loss = loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(trainable.parameters(), args.max_grad_norm)

            # Weights update
            if ((step + 1) % args.gradient_accumulation_steps == 0) or (step + 1 == len(train_dataloader)):
                if args.mixed_precision == "no":
                    some_model.optimizer.step()
                    some_model.optimizer.zero_grad()
                else:
                    scaler.step(some_model.optimizer)
                    some_model.optimizer.zero_grad()
                    scaler.update()

            # Gather the losses across all processes for logging (if we use distributed training).
            step_loss = loss.detach().item()
            epoch_loss += step_loss / args.max_train_steps

            # Checks if the accelerator has performed an optimization step behind the scenes
            progress_bar.update(1)
            global_step += 1

            logs = {"step_loss": step_loss, "lr": some_model.lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
        
        some_model.lr_scheduler.step() # algo mejor (tmb cambiado en some_model.lr_scheduler)

        print(f"Epoch {epoch} completed (step {step+1}/{args.max_train_steps})")

        if epoch % args.validation_epochs == 0:
            validation(some_model, device, main_dtype, args, val_dataloader, f"{some_model.name}_epoch{epoch}", metric_models, loss=epoch_loss)

        if epoch % args.checkpointing_epochs == 0:
            #save_path = os.path.join(args.output_dir, f"checkpoint")
            #if not os.path.exists(save_path):
            #    os.makedirs(save_path)
            some_model.save(args, dtype=main_dtype) # model has not save_pretrained!

    # Create the pipeline using the trained modules and save it.
    some_model.save(args, dtype=main_dtype)

    if args.push_to_hub:
        upload_folder(
            repo_id=repo_id,
            folder_path=args.output_dir,
            commit_message="End of training",
            ignore_patterns=["step_*", "epoch_*"],
        )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
