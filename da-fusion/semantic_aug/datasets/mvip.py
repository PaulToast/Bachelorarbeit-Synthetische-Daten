from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Dict

import numpy as np
import torchvision.transforms as transforms
import torch
import os
import json

from PIL import Image
from scipy.ndimage import maximum_filter
from collections import defaultdict

MVIP_DIR = "/mnt/HDD/MVIP/sets"
SUPER_CLASS = "CarComponent"
NUM_CLASSES = 20


class MVIPDataset(FewShotDataset):

    # Go through all classes & check the metadata
    # If the class is of the selected super_class, add it to class_names
    class_names = []

    for class_name in [f for f in os.listdir(MVIP_DIR) if os.path.isdir(os.path.join(MVIP_DIR, f))]:
        meta_file = open(os.path.join(MVIP_DIR, class_name, "meta.json"))
        meta_data = json.load(meta_file)

        if SUPER_CLASS in meta_data['super_class']:
            class_names.append(class_name)

        meta_file.close()

        # Select only a certain number of the classes
        del class_names[NUM_CLASSES:]

    num_classes: int = len(class_names)

    def __init__(self, *args, split: str = "train", seed: int = 0,
                 image_dir: str = MVIP_DIR,
                 examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 use_randaugment: bool = False,
                 image_size: Tuple[int] = (256, 256), **kwargs):

        super(MVIPDataset, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability, 
            generative_aug=generative_aug, **kwargs)
        
        split_dir = {
            "train": "train_data",
            "test": "test_data",
            "val": "valid_data"
        } 

        # Add every image & mask path to a class_to_images dict
        class_to_images = defaultdict(list)
        class_to_masks = defaultdict(list)

        for class_name in self.class_names:
            root = os.path.join(image_dir, class_name, split_dir[split])

            # Go through every set + orientation + cam
            # Select only the rgb images
            for set in [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]:
                for orientation in [f for f in os.listdir(os.path.join(root, set)) if os.path.isdir(os.path.join(root, set, f))]:
                    for cam in [f for f in os.listdir(os.path.join(root, set, orientation)) if os.path.isdir(os.path.join(root, set, orientation, f))]:
                        for file in os.listdir(os.path.join(root, set, orientation, cam)):
                            if file.endswith("rgb.png"):
                                class_to_images[class_name].append(os.path.join(root, set, orientation, cam, file))
                            if file.endswith("rgb_mask_gen.png"):
                                class_to_masks[class_name].append(os.path.join(root, set, orientation, cam, file))

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

        self.all_images = sum([
            self.class_to_images[key] 
            for key in self.class_names], [])
        self.all_masks = sum([
            self.class_to_masks[key] 
            for key in self.class_names], [])

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
            #transforms.Resize(image_size),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15.0),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.4213, 0.4252, 0.4242],
                                  std=[0.1955, 0.1923, 0.1912])
        ])

        val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.4213, 0.4252, 0.4242],
                                  std=[0.1955, 0.1923, 0.1912])
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

def crop_object(image, mask):
    mask = np.where(mask, 255, 0).astype(np.uint8)
    
    # Dilate mask with maximum filter
    mask = Image.fromarray(maximum_filter(mask, size=32))
    
    # Get bounding box of object mask
    mask_box = mask.getbbox()

    # Make mask_box square without offsetting the center
    mask_box_width = mask_box[2] - mask_box[0]
    mask_box_height = mask_box[3] - mask_box[1]
    mask_box_size = max(mask_box_width, mask_box_height)
    mask_box_center_x = (mask_box[2] + mask_box[0]) // 2
    mask_box_center_y = (mask_box[3] + mask_box[1]) // 2
    mask_box = (
        mask_box_center_x - mask_box_size // 2,
        mask_box_center_y - mask_box_size // 2,
        mask_box_center_x + mask_box_size // 2,
        mask_box_center_y + mask_box_size // 2
    )
    
    return image.crop(mask_box), mask.crop(mask_box)