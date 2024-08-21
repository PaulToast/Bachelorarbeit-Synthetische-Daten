from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Dict

import numpy as np
import torchvision.transforms as transforms
import torch
import os
import json

from PIL import Image
from collections import defaultdict

MVIP_DIR = "/home/hofmpaul/Documents/Repositories/Bachelorarbeit-Synthetische-Daten/_datasets/mvip-mini"


class MVIPMiniDataset(FewShotDataset):

    class_names = ["motor_1", "motor_2"]

    num_classes: int = len(class_names)

    def __init__(self, *args, split: str = "train", seed: int = 0,
                 image_dir: str = MVIP_DIR,
                 examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 use_randaugment: bool = False,
                 image_size: Tuple[int] = (256, 256), **kwargs):

        super(MVIPMiniDataset, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability, 
            generative_aug=generative_aug, **kwargs)

        # Add every image & mask path to a class_to_images dict
        class_to_images = defaultdict(list)
        class_to_masks = defaultdict(list)

        for class_name in self.class_names:
            root = os.path.join(image_dir, class_name, split)

            for example in [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]:
                for file in os.listdir(os.path.join(root, example)):
                    if file.endswith("mask.jpg"):
                        class_to_masks[class_name].append(os.path.join(root, example, file))
                    else:
                        class_to_images[class_name].append(os.path.join(root, example, file))

        # Generate a sequence of ids for each class, in random order,
        # effectively shuffling the order of images
        rng = np.random.default_rng(seed)

        class_to_ids = {key: rng.permutation(
            len(class_to_images[key])) for key in self.class_names}
        
        if examples_per_class is not None:
            class_to_ids = {key: ids[:examples_per_class] 
                            for key, ids in class_to_ids.items()}

        ##############################################################################

        # Replicate class_to_images & class_to_masks, but in the random order from class_to_ids
        self.class_to_images = {
            key: [class_to_images[key][i] for i in ids] 
            for key, ids in class_to_ids.items()}
        self.class_to_masks = {
            key: [class_to_masks[key][i] for i in ids] 
            for key, ids in class_to_ids.items()}

        """self.class_to_annotations = {
            key: [class_to_annotations[key][i] for i in ids] 
            for key, ids in class_to_ids.items()}"""

        self.all_images = sum([
            self.class_to_images[key] 
            for key in self.class_names], [])
        self.all_masks = sum([
            self.class_to_masks[key] 
            for key in self.class_names], [])

        """self.all_annotations = sum([
            self.class_to_annotations[key] 
            for key in self.class_names], [])"""

        self.all_labels = [i for i, key in enumerate(
            self.class_names) for _ in self.class_to_images[key]]

        if use_randaugment: train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        else: train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15.0),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        self.transform = {"train": train_transform, "val": val_transform}[split]

    def __len__(self):
        
        return len(self.all_images)

    def get_image_by_idx(self, idx: int) -> torch.Tensor:

        return Image.open(self.all_images[idx]).convert('RGB')

    def get_label_by_idx(self, idx: int) -> torch.Tensor:

        return self.all_labels[idx]
    
    def get_metadata_by_idx(self, idx: int) -> Dict:

        return dict(
            name=self.class_names[self.all_labels[idx]],
            mask=np.array(Image.open(self.all_masks[idx]).convert('L'))
        )