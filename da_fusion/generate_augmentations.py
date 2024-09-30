from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.spurge import SpurgeDataset
from semantic_aug.datasets.imagenet import ImageNetDataset
from semantic_aug.datasets.pascal import PASCALDataset

from mvip_dataset import MVIPDatasetDAFusion, crop_to_object

from semantic_aug.augmentations.compose import ComposeParallel
from semantic_aug.augmentations.compose import ComposeSequential
from semantic_aug.augmentations.real_guidance import RealGuidance
from semantic_aug.augmentations.textual_inversion import TextualInversion

from diffusers import StableDiffusionPipeline
from itertools import product
from torch import autocast
from PIL import Image

from tqdm import tqdm
import os
import torch
import argparse
import numpy as np
import random


DATASETS = {
    "spurge": SpurgeDataset, 
    "coco": COCODataset, 
    "pascal": PASCALDataset,
    "imagenet": ImageNetDataset,
    "mvip": MVIPDatasetDAFusion,
}

COMPOSE = {
    "parallel": ComposeParallel,
    "sequential": ComposeSequential
}

AUGMENT = {
    "real-guidance": RealGuidance,
    "textual-inversion": TextualInversion
}

PROMPTS = [
    "a photo of a used {name}",
    "a photo of a dirty {name}",
    "a photo of a rusty {name}",
    "a photo of a broken {name}",
    "a photo of a old {name}",
    "a photo of a clean {name}",
    "a photo of a brand new {name}",
]


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Inference script for generating augmentations")
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="mvip", 
        choices=DATASETS.keys(),
    )
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--aug_name",
        type=str,
        default="augs",
        help="Subfolder name for augmentations output. Will default to 'augs'."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument(
        "--embed_path",
        type=str,
        default=None,
        help="Will default to 'output/{output_name}/fine-tuned-merged/{dataset}-0-16.pt'"
    )
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples_per_class", type=int, default=1)
    parser.add_argument("--crop_object", type=bool, default=False, help="Before augmenting, create a crop using the object mask bounding box.")
    parser.add_argument("--num_synthetic", type=int, default=10)

    parser.add_argument("--aug", nargs="+", type=str, default=["real-guidance"], choices=["real-guidance", "textual-inversion"])

    parser.add_argument("--prompt", type=str, default="a photo of a {name}")
    parser.add_argument("--guidance_scale", nargs="+", type=float, default=[7.5])
    parser.add_argument("--strength", nargs="+", type=float, default=[0.5])

    
    parser.add_argument("--mask", nargs="+", type=int, default=[0], choices=[0, 1], help="Only augment masked regions.")
    parser.add_argument("--inverted", nargs="+", type=int, default=[0], choices=[0, 1], help="Invert mask.")
    
    parser.add_argument("--probs", nargs="+", type=float, default=None)
    
    parser.add_argument("--compose", type=str, default="parallel", choices=["parallel", "sequential"])

    parser.add_argument("--class_name", type=str, default=None)
    
    parser.add_argument("--erasure_ckpt_path", type=str, default=None)

    args = parser.parse_args()

    if args.embed_path == None:
        args.embed_path = os.path.abspath(f"output/{args.output_name}/fine-tuned-merged/seed=0_ex=32.pt")

    args.output_dir = os.path.abspath(f"output/{args.output_name}/{args.aug_name}/")

    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    aug = COMPOSE[args.compose]([
        
        AUGMENT[aug](
            embed_path=args.embed_path, 
            model_path=args.model_path, 
            prompt=args.prompt, 
            strength=strength, 
            guidance_scale=guidance_scale,
            mask=mask, 
            inverted=inverted,
            erasure_ckpt_path=args.erasure_ckpt_path
        )

        for (aug, guidance_scale, 
             strength, mask, inverted) in zip(
            args.aug, args.guidance_scale, 
            args.strength, args.mask, args.inverted
        )

    ], probs=args.probs)

    train_dataset = DATASETS[
        args.dataset](split="train", seed=args.seed, 
                      examples_per_class=args.examples_per_class)

    options = product(range(len(train_dataset)), range(args.num_synthetic))

    for idx, num in tqdm(list(
            options), desc="Generating Augmentations"):

        image = train_dataset.get_image_by_idx(idx)
        label = train_dataset.get_label_by_idx(idx)

        metadata = train_dataset.get_metadata_by_idx(idx)

        # Crop the image using object mask
        if args.crop_object:
            image, metadata["mask"] = crop_to_object(image, metadata["mask"])

        if args.class_name is not None: 
            if metadata["name"] != args.class_name: continue

        if args.dataset == "mvip" or "mvip-mini":
            prompt = random.choice(PROMPTS)
        else:
            prompt = None

        image, label = aug(
            image, label, metadata, prompt)

        name = metadata['name'].replace(" ", "_")

        pil_image, image = image, os.path.join(
            args.output_dir, f"{name}-{idx}-{num}.png")

        pil_image.save(image)