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

from PIL import Image

from models import StableDiffusion, StableDiffusionLoRA, MyDiffusion

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

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
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
        "--results_dir",
        type=str,
        default=None,
        help="The directory that contains images and 'prompts.txt' or 'log.txt'. If None, ground truth will be evaluated.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    ##############################
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None
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
    ##############################
    parser.add_argument(
        "--model_type",
        type=str,
        default=None
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=None
    )
    parser.add_argument(
        "--generate_samples",
        type=int,
        default=None
    )
    parser.add_argument(
        "--measure_from_txt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--dont_plot",
        action="store_true"
    )
    parser.add_argument(
        "--dont_write",
        action="store_true"
    )
    ##############################
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
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--attestimator_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pth.tar file of face attribute estimator.",
    )

    args = parser.parse_args()

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

def plot(lst, x_values=None, y_lim=None, x_label="", y_label="", title="", save_as="plot.jpg"):
    fig, ax = plt.subplots()
    ticks = range(len(lst))
    ax.plot(ticks, lst)
    if x_values is not None: ax.set_xticks(ticks, x_values)
    if y_lim is not None: ax.set_ylim(y_lim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.savefig(save_as)

def tokenize_caption(some_model, caption):
    inputs = some_model.tokenizer(
        [caption], max_length=some_model.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids[0]

def generate_sample(some_model, pipeline, prompt, args, device):
    kwargs = {
        'num_inference_steps':50,
        'guidance_scale':7.5,
        'height':64,
        'width':64
    }
    with torch.autocast("cuda"):
        input_ids = [tokenize_caption(some_model, prompt)]
        input_ids = torch.stack(input_ids)
        print(input_ids.shape)
        encoder_hidden_states = pipeline.text_encoder(input_ids.to(device))[0]
        images = pipeline(prompt_embeds=encoder_hidden_states, generator=None, **kwargs).images
    return images[0]

def validation(eval_dir, args, val_dataloader, metric_models): # num_generations debe ser al menos 500 para ser significativo

        print(f"Performing validation for {eval_dir}...")

        if eval_dir is None:
            total_samples = len(val_dataloader)
        else:
            total_samples = len(os.listdir(eval_dir)) - 1
            
        avg_fid = 0.0
        avg_clipscore = 0.0
        avg_mtcnnscore = 0.0
        avg_attestimator = 0.0

        fid = metric_models['fid']
        mtcnn = metric_models['mtcnn']
        attestimator = metric_models['attestimator']

        evaluated_images = 0

        pbar = tqdm(total=total_samples)
        for idx, batch in enumerate(val_dataloader):
            if idx == 0:
                bs = batch['pixel_values'].shape[0]
                assert bs == 1, f"val_dataloader expected to have batch_size=1 instead of {bs}"

            prompt = [batch['input_texts'][0]]
            t_real = batch['pixel_values'].to("cpu", dtype=torch.float32)
            t_real = decode_image(t_real, to_numpy=False, to_255=False)

            if args.results_dir is None:
                t_fake = t_real
                rgb_fake = transforms.ToPILImage()(torch.squeeze(t_fake,0))
            else:
                # load fake image and covert to torch tensor [0.0, 255.0]
                rgb_fake = Image.open(f"{eval_dir}/{idx}.jpg").convert("RGB")
                t_fake = transforms.ToTensor()(rgb_fake)
                t_fake = torch.unsqueeze(t_fake,0)

            rgb_fake.save(f"{args.output_dir}/latest_rgb_fake.jpg")

            # FID
            fid.update(t_real, real=True)
            fid.update(t_fake, real=False)

            # 8,20
            step = 256
            if idx>1 and (idx+1 % step == 0 or idx+1 + step >= total_samples): avg_fid += float(fid.compute())

            # CLIPScore
            avg_clipscore += calculate_clip_score(t_fake, prompt, to_uint8=True, to_255=True)

            # MTCNN
            faces = mtcnn.detect_faces(np.array(rgb_fake))
            if len(faces)>0:
                avg_mtcnnscore += faces[0]['confidence']
            else:
                avg_mtcnnscore += 0.0

            # Attribute estimator
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

            evaluated_images += 1
            pbar.update(1)

            if idx >= total_samples - 1:
                break

        pbar.close()

        # actualiza calidad de imagen (FID)
        print("Computing metrics...")
        avg_fid /= evaluated_images
        avg_clipscore /= evaluated_images
        avg_mtcnnscore /= evaluated_images
        avg_attestimator /= evaluated_images

        results = {
            'num_samples':total_samples,
            'fid':avg_fid,
            'clipscore':avg_clipscore,
            'mtcnn':avg_mtcnnscore,
            'attestimator':avg_attestimator
        }

        return results


def main():
    args = parse_args()

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    print("Loading dataset from folder!")
    data_dir = os.path.abspath(args.data_dir)
    dataset = load_dataset(
        "imagefolder",
        data_files={
            'train': os.path.join(*[data_dir, "train", "**"]),
            'validation': os.path.join(*[data_dir, "validation", "**"])
        },
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
    val_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_val(example):
        example["pixel_values"] = val_transforms(example[image_column].convert("RGB"))
        example["input_texts"] = example[caption_column]
        example["input_classes"] = example[class_column]
        example["attributes"] = binary_string_to_list(example[attributes_column], as_tensor=True)
        return example

    print("Setting data transforms!")
    # Set the training transforms
    val_dataset = dataset["validation"].map(preprocess_val)

    # num_samples = images - metadata.jsonl
    num_val_samples = len(os.listdir(os.path.join(data_dir,'validation'))) - 1

    val_dataset = ActuallyIterableDataset(val_dataset, num_val_samples, max=1024)

    num_classes = 16
    def collate_fn(batch):
        pixel_values = torch.stack([example["pixel_values"] for example in batch])
        pixel_values = pixel_values.to(device, memory_format=torch.contiguous_format)
        input_texts = [example["input_texts"] for example in batch]
        input_classes = torch.tensor([example["input_classes"] for example in batch])
        input_classes = F.one_hot(input_classes, num_classes=num_classes).to(device)
        input_attributes = torch.stack([example["attributes"] for example in batch])
        input_attributes = input_attributes.to(device)
        return {"pixel_values": pixel_values, "input_texts": input_texts, "input_classes": input_classes, "input_attributes": input_attributes}

    print("Loading dataloaders!")

    # DataLoaders creation:
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=0
    )

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

    # Generate
    if args.pretrained_model_name_or_path is not None:

        # get pipeline from pretrained
        if args.model_type == "StableDiffusion":
            some_model = StableDiffusion()
        elif args.model_type == "StableDiffusionLoRA":
            some_model = StableDiffusionLoRA()
        elif args.model_type == "MyDiffusion":
            block_out_channels = [256,512,768,768]
            some_model = MyDiffusion(block_out_channels=block_out_channels)

        some_model.setup_parts(args)
    
        if args.model_type == "StableDiffusion":
            pipeline = some_model.get_pipeline(args, dtype=torch.float32, overwrite_current_weights=True)
        elif args.model_type == "StableDiffusionLoRA":
            pipeline = some_model.get_pipeline(args, dtype=torch.float32, overwrite_current_weights=True)
            if args.lora_weights is not None: pipeline.unet.load_attn_procs(args.lora_weights)
        elif args.model_type == "MyDiffusion":
            pipeline = some_model.get_pipeline(args, dtype=torch.float32, overwrite_current_weights=True)

        pipeline = pipeline.to(device)

        save_path = f"{args.output_dir}/samples"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Generate and save
        print(f"Generating {args.generate_samples} samples...")
        pbar = tqdm(total=args.generate_samples)
        generated_samples = 0
        for idx, batch in enumerate(val_dataloader):
            assert len(batch["input_texts"]) == 1, "Expected batch of 1"
            prompt = batch["input_texts"][0]
            img = generate_sample(some_model, pipeline, prompt, args, device)
            img_file = f"{idx}.jpg"
            img.save(f"{save_path}/{img_file}")
            with open(f"{save_path}/prompts.txt", 'a') as file:
                file.write(f"{img_file} '{prompt}'\n")
            generated_samples += 1
            pbar.update(1)
            if generated_samples >= args.generate_samples:
                break
        pbar.close()      

    # Eval!

    print("(!) Assuming folder content inside experiment results: abc_epoch0, abc_epoch1, abc_epoch2, abc_epochNone, abc_epochAbc, ..., log.txt")
    print("(!) Assuming folder content inside epoch results: 0.jpg, 1.jpg, 2.jpg, prompts.txt")
    
    if args.results_dir is None:

        gt = "ground_truth"
        eval_results = [{
            'name': gt,
            'result': validation(None, args, val_dataloader, metric_models)
        }]

    else:

        eval_results = []

        if os.path.isfile(f"{args.results_dir}/log.txt"):
            # Experiment results
            if args.measure_from_txt:
                with open(args.measure_from_txt, "r") as f:
                    lines = f.readlines()  # Lee todas las líneas del archivo y las guarda en una lista
                    for l in lines:
                        l = l.strip()
                        if len(l) == 0: continue
                        _, name, str_date, str_fid, str_clipscore, str_mtcnnscore, str_attestimator = l.split('\t')
                        eval_results.append({
                            'name':name,
                            'result':{
                                'num_samples':None,
                                'fid':float(str_fid),
                                'clipscore':float(str_clipscore),
                                'mtcnn':float(str_mtcnnscore),
                                'attestimator':float(str_attestimator)
                            }
                        })
            else:
                dir_content = []
                for f in os.listdir(args.results_dir):
                    split = f.split('epoch')
                    if len(split) > 1 and split[1].isnumeric():
                        e = split[1]
                        dir_content.append(next(d for d in os.listdir(args.results_dir) if d.endswith(f"epoch{e}")))
                dir_content = sorted(dir_content, key=lambda x: int(x.split('epoch')[1]))
                for subdir in dir_content:
                    if os.path.isdir(os.path.join(args.results_dir,subdir)):
                        eval_results.append({
                            'name':subdir,
                            'result': validation(os.path.join(args.results_dir,subdir), args, val_dataloader, metric_models)
                        })
        elif os.path.isfile(f"{args.results_dir}/prompts.txt"):
            # Epoch results
            eval_results.append({
                    'name':os.path.basename(os.path.normpath(args.results_dir)), # .../.../abc/ -> abc
                    'result': validation(args.results_dir, args, val_dataloader, metric_models)
                })
        else:
            print(f"Invalid args.resuts_dir:{args.results_dir}")
            print(os.listdir(args.results_dir))
            raise NotImplementedError
        

    # crea una carpeta para este experimento y esta validación
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    out_filename = 're-evaluation.txt'

    # plot
    if len(eval_results) > 1 and not args.dont_plot:
        plot_path = f"{args.output_dir}/plots"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        lst_clipscore = [r['result']['clipscore'] for r in eval_results]
        lst_attestimator = [r['result']['attestimator'] for r in eval_results]
        lst_fid = [r['result']['fid'] for r in eval_results]
        lst_mtcnn = [r['result']['mtcnn'] for r in eval_results]

        plot(lst_clipscore, x_values=range(len(eval_results)), y_lim=(20,30), x_label="Epochs", y_label="CLIPScore", title="", save_as=f"{plot_path}/clipscore.jpg")
        plot(lst_attestimator, x_values=range(len(eval_results)), y_lim=(0.5,1.0), x_label="Epochs", y_label="Accuracy of Attributes Estimator", title="", save_as=f"{plot_path}/attestimator.jpg")
        plot(lst_fid, x_values=range(len(eval_results)), y_lim=(80.0,230.0), x_label="Epochs", y_label="FID", title="", save_as=f"{plot_path}/fid.jpg")
        plot(lst_mtcnn, x_values=range(len(eval_results)), y_lim=(0.5,1.0), x_label="Epochs", y_label="MTCNN Confidence", title="", save_as=f"{plot_path}/mtcnn.jpg")
        
    if not args.dont_write:
        for res_dict in eval_results:
            name = res_dict['name']
            r = res_dict['result']

            # write file
            with open(os.path.join(*[args.output_dir, out_filename]), 'a') as file:
                datetime_str = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
                file.write(f"(fields:name&date&fid&clipscore&mtcnnscore&attestimator)\t{name}\t{datetime_str}\t{r['fid']}\t{r['clipscore']}\t{r['mtcnn']}\t{r['attestimator']}\n")
            
    torch.cuda.empty_cache()

    print(f"Done! Saved at {args.output_dir}/{out_filename}")


if __name__ == "__main__":
    main()
