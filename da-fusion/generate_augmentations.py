from semantic_aug.datasets.mvip import MVIPDataset
from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.spurge import SpurgeDataset
from semantic_aug.datasets.imagenet import ImageNetDataset
from semantic_aug.datasets.pascal import PASCALDataset
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
    "mvip": MVIPDataset
}

COMPOSE = {
    "parallel": ComposeParallel,
    "sequential": ComposeSequential
}

AUGMENT = {
    "real-guidance": RealGuidance,
    "textual-inversion": TextualInversion
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Inference script")
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="mvip", 
        choices=DATASETS.keys(),
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Will default to '_experiments/{dataset}-{experiment_name}/aug/'."
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
        help="Will default to '_experiments/{dataset}-{experiment_name}/fine-tuned-merged/{dataset}-0-16.pt'"
    )
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples_per_class", type=int, default=1)
    parser.add_argument("--num_synthetic", type=int, default=10)

    parser.add_argument("--prompt", type=str, default="a photo of a {name}")
    
    parser.add_argument("--aug", nargs="+", type=str, default=["real-guidance"], 
                        choices=["real-guidance", "textual-inversion"])

    parser.add_argument("--guidance_scale", nargs="+", type=float, default=[7.5])
    parser.add_argument("--strength", nargs="+", type=float, default=[0.5])

    parser.add_argument("--mask", nargs="+", type=int, default=[0], choices=[0, 1])
    parser.add_argument("--inverted", nargs="+", type=int, default=[0], choices=[0, 1])
    
    parser.add_argument("--probs", nargs="+", type=float, default=None)
    
    parser.add_argument("--compose", type=str, default="parallel", 
                        choices=["parallel", "sequential"])

    parser.add_argument("--class_name", type=str, default=None)
    
    parser.add_argument("--erasure_ckpt_path", type=str, default=None)

    args = parser.parse_args()

    if args.embed_path == None:
        args.embed_path = os.path.abspath(
            os.path.join(os.path.dirname( __file__ ), '..', '_experiments', f"{args.dataset}-{args.experiment_name}/fine-tuned-merged/seed=0_ex=16.pt")
        )
    if args.out == None:
        args.out = os.path.abspath(
            os.path.join(os.path.dirname( __file__ ), '..', '_experiments', f"{args.dataset}-{args.experiment_name}", 'aug')
        )

    os.makedirs(args.out, exist_ok=True)

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

        if args.class_name is not None: 
            if metadata["name"] != args.class_name: continue

        image, label = aug(
            image, label, metadata)

        name = metadata['name'].replace(" ", "_")

        pil_image, image = image, os.path.join(
            args.out, f"{name}-{idx}-{num}.png")

        pil_image.save(image)