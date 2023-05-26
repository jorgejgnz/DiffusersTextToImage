
import argparse
import os
import sys
import time
import random
import numpy as np
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.checkpoint
from torch.utils.data import IterableDataset
from torchvision.transforms import transforms
from torchvision import models

from datasets import load_dataset
from tqdm.auto import tqdm

import inspect
from pprint import pprint

from datetime import datetime

import albumentations as A
from albumentations.pytorch import ToTensorV2


#######


class ResNet50_Sigm(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()
 
    def forward(self, x):
        return self.sigm(self.base_model(x))
    

class VGG16_Sigm(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        vgg.classifier[6] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=vgg.classifier[6].in_features, out_features=n_classes)
        )
        self.base_model = vgg
        self.sigm = nn.Sigmoid()
 
    def forward(self, x):
        return self.sigm(self.base_model(x))
    

class DenseNet_Sigm(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        vgg = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        vgg.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=vgg.classifier.in_features, out_features=n_classes)
        )
        self.base_model = vgg
        self.sigm = nn.Sigmoid()
 
    def forward(self, x):
        return self.sigm(self.base_model(x))
    

#######


class ActuallyIterableDataset(IterableDataset):
    def __init__(self, dataset, length):
        self.src = dataset
        self.len = length

    def __iter__(self):
        for example in self.src:
            yield example

    def __len__(self):
        return self.len
    
    def shuffle(self, buffer_size=10000, seed=42):
        self.src = self.src.shuffle(buffer_size=buffer_size,seed=seed)

def binary_string_to_list(bin_str, as_tensor=False):
    result = []
    for bit in bin_str:
        if bit == '0':
            result.append(0.0)
        elif bit == '1':
            result.append(1.0)
    if as_tensor: result = torch.tensor(result)
    return result

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
    plt.savefig('dataaugmentation.png')

def criterion(loss_func, predicted, targets, att_probability):
    loss = 0
    batch_size, num_atts = targets.shape
    #loss_weights = att_probability.repeat(batch_size, 1)
    for att_idx in range(num_atts):
        att_loss = loss_func(predicted[:,att_idx], targets[:,att_idx]) #weight=loss_weights[:,att_idx])
        loss += att_loss / num_atts
    return loss

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)

    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    torch.save(state, filepath)

def measure_accuracy(model, dataloader, dtype, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch["pixel_values"] = batch["pixel_values"].to(device, dtype=dtype)             # [b, 3, 64, 64]
            batch["input_attributes"] = batch["input_attributes"].to(device, dtype=dtype)     # [b, 40]
            outputs = model(batch["pixel_values"] )
            predicted = (outputs > 0.5).float()   # convertir las salidas en etiquetas binarias
            total += torch.numel(batch["input_attributes"])
            correct += (predicted == batch["input_attributes"]).sum().item()
    acc = correct / total
    return acc

def measure_att_accuracy(model, dataloader, att_idx, dtype, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_size = batch["input_attributes"].shape[0]
            valid_images = []
            valid_attributes = []

            for idx in range(batch_size):
                if batch["input_attributes"][idx][att_idx] > 0.5:
                    valid_images.append(batch["pixel_values"][idx])
                    valid_attributes.append(batch["input_attributes"][idx])

            valid_images = torch.stack(valid_images).to(device, dtype=dtype)
            valid_attributes = torch.stack(valid_attributes).to(device, dtype=dtype)

            outputs = model(valid_images)
            predicted = (outputs > 0.5).float()   # convertir las salidas en etiquetas binarias
            total += len(valid_images)
            correct += (predicted[:,att_idx] == valid_attributes[:,att_idx]).sum().item()
    acc = correct / total
    return acc

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

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
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--attributes_column",
        type=str,
        default="attributes",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
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
        default=True,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1024,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=1024,
        help="Batch size (per device) for the validation dataloader."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=20
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="face_attribute_estimator_output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--checkpoint_epochs",
        type=int,
        default=5,
        help="Save checkpoint every X epochs.",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.output_dir is None or args.data_dir is None:
        raise ValueError("Need dataset folder and output folder!")

    return args


#######


def main():

    args = parse_args()

    print("Setting device and dtype!")
    main_dtype = torch.float32

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    data_dir = os.path.abspath(args.data_dir)
    dataset = load_dataset(
        "imagefolder",
        data_files={
            'train': os.path.join(*[data_dir, "train", "**"]),
            'validation': os.path.join(*[data_dir, "validation", "**"])
        },
        streaming=True
    )

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = list(next(iter(dataset["train"])).keys())

    # 6. Get the column names for input/target.
    image_column = args.image_column
    if image_column not in column_names:
        raise ValueError(
            f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
        )
    attributes_column = args.attributes_column
    if attributes_column not in column_names:
        raise ValueError(
            f"--attributes_column' value '{args.attributes_column}' needs to be one of: {', '.join(column_names)}"
        )

    # Preprocessing the datasets.
    albumentations_transforms = A.Compose(
        [
            A.ColorJitter(always_apply=False, p=0.25, brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1)),
            A.ToGray(always_apply=False, p=0.1)
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
        example["pixel_values"] = albumentations_transforms(image=np.array(example[image_column].convert("RGB")))['image'] # image -> numpy
        example["pixel_values"] = train_transforms(Image.fromarray(example["pixel_values"])) # numpy -> image
        example["attributes"] = binary_string_to_list(example[attributes_column], as_tensor=True)
        return example
    
    def preprocess_val(example):
        example["pixel_values"] = val_transforms(example[image_column].convert("RGB"))
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

    train_dataset = ActuallyIterableDataset(train_dataset, num_train_samples)
    val_dataset = ActuallyIterableDataset(val_dataset, num_val_samples)

    def collate_fn(batch):
        pixel_values = torch.stack([example["pixel_values"] for example in batch])
        pixel_values = pixel_values.to(device, memory_format=torch.contiguous_format)
        input_attributes = torch.stack([example["attributes"] for example in batch])
        input_attributes = input_attributes.to(device)
        return {"pixel_values": pixel_values, "input_attributes": input_attributes}

    print("Loading dataloaders!")

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=0,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.val_batch_size,
        num_workers=0
    )

    train_dataloader.dataset.shuffle(buffer_size=10000, seed=42)

    # Show data augmentation
    demo_dataset = ActuallyIterableDataset(dataset["train"], num_train_samples)
    show_examples(args, iter(demo_dataset), albumentations_transforms)
    del demo_dataset

    num_labels=40
    validation_attribute = ('Eyeglasses',15)
    target_range = "[0,1]" #[-1,1]
    steps_per_epoch = len(train_dataloader)

    loss_func = nn.functional.binary_cross_entropy
    #model = ResNet50_Sigm(num_labels).to(device)
    model = VGG16_Sigm(num_labels).to(device)
    #model = DenseNet_Sigm(num_labels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)   

    # Train!
    total_batch_size = args.train_batch_size
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = Disabled")
    print(f"  Steps/epoch = {steps_per_epoch}")
    global_step = 0
    first_epoch = 0

    logs = []

    #print('Performing validation!')
    #val_acc = measure_accuracy(model, val_dataloader, main_dtype, device)
    #print(f"Validation accuracy: {val_acc}")
    #att_val_acc = measure_att_accuracy(model, val_dataloader, validation_attribute[1], main_dtype, device)
    #print(f"Validation accuracy for {validation_attribute[0]}: {att_val_acc}")

    att_probability = torch.tensor([0.11113579040370387, 0.26698058726844653, 0.51250499755675, 0.20457159215988233, 0.022443348683853327, 0.15157527924619568, 0.24079585782753124, 0.23453225336748948, 0.23925093411122464, 0.14799184596172735, 0.05089857304330229, 0.2051935103332198, 0.14216753290983666, 0.05756691790186526, 0.046688285726977925, 0.06511878143524893, 0.06276437692189991, 0.041949861549168556, 0.38692194926924617, 0.45503186096673726, 0.41675427815537097, 0.48342785502396357, 0.04154512115064734, 0.11514864337928618, 0.8349399552811219, 0.28414256733744986, 0.04294690496991594, 0.27744460732777554, 0.07977828123534667, 0.06572095617451221, 0.05651064417889526, 0.48208036564839907, 0.20840181837027824, 0.3195672239250934, 0.1889249206560743, 0.04846025893513788, 0.47243569810314956, 0.12296704327267163, 0.07271506769529958, 0.773616849046639])
    att_probability = att_probability.unsqueeze(0).to(device)

    for epoch in range(first_epoch, args.num_train_epochs):

        print(f"Starting epoch {epoch}!")

        #trainable.train()
        epoch_loss = 0.0

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(0, steps_per_epoch))
        progress_bar.set_description("Steps")

        for step, batch in enumerate(train_dataloader):

            batch_size = batch["input_attributes"].shape[0]

            # Convert images to dtype
            batch["pixel_values"] = batch["pixel_values"].to(main_dtype)            # [b, 3, 64, 64]
            batch["input_attributes"] = batch["input_attributes"].to(main_dtype)    # [b, 40]

            outputs = model(batch["pixel_values"])
            loss = criterion(loss_func, outputs, batch["input_attributes"], att_probability)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step_loss = loss.item()
            epoch_loss += step_loss / steps_per_epoch

            progress_bar.update(1)
            global_step += 1

            pbar_logs = {"step_loss": step_loss, "lr": args.learning_rate}
            progress_bar.set_postfix(**pbar_logs)

        if epoch % args.validation_epochs == 0:
            print('Performing validation!')
            val_acc = measure_accuracy(model, val_dataloader, main_dtype, device)
            print(f"Validation accuracy: {val_acc}")
            att_val_acc = measure_att_accuracy(model, val_dataloader, validation_attribute[1], main_dtype, device)
            print(f"Validation accuracy for {validation_attribute[0]}: {att_val_acc}")

            # Checkpoint
            print("Saving checkpoint!")
            logs.append({'loss':epoch_loss,'val_acc':val_acc})
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'log_per_epoch': logs
            }, checkpoint='checkpoint')

    # Evaluacion con test (TO DO)

if __name__ == "__main__":
    main()

