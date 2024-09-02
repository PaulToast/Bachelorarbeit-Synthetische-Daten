import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy.ndimage import maximum_filter


class MVIPDataset(Dataset):
    def __init__(
        self,
        split="train",
        aug_mode=None, # None, "positive", "both", "positive_only", "negative_only"
        aug_dir_positive=None,
        aug_dir_negative=None,
        aug_ex_positive=-1, # -1 for all
        aug_ex_negative=-1, # -1 for all
        image_size=512,
        transform=None,
    ):
        self.data_root = "/mnt/HDD/MVIP/sets"
        self.split = split
        
        self.aug_mode = aug_mode
        self.aug_dir_positive = aug_dir_positive
        self.aug_dir_negative = aug_dir_negative
        self.aug_ex_positive = aug_ex_positive
        self.aug_ex_negative = aug_ex_negative

        # Define classes & collect dataset
        self._initialize_classes(num_classes=20, super_class="CarComponent")
        self._collect_dataset()

        # Set image transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.)), # ratio=(1.0, 1.0)?
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ])

        # Shuffle dataset
        np.random.seed(0)
        shuffle_idx = np.random.permutation(self._length)
        self.all_images = [self.all_images[i] for i in shuffle_idx]
        self.all_masks = [self.all_masks[i] for i in shuffle_idx]
        self.all_labels = [self.all_labels[i] for i in shuffle_idx]
    
    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # Get label
        label = self.all_labels[idx % self._length]

        # Get image
        image = Image.open(self.all_images[idx % self._length])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        # Crop object using mask
        if self.all_masks[idx % self._length] is not None:
            mask = Image.open(self.all_masks[idx % self._length]).convert("L")
            image, mask = self.crop_object(image, mask)

        return self.transform(image), label
        
    def _initialize_classes(self, num_classes, super_class):
        """Define class names list & set-up class to label id mapping.
        Limit dataset to specified num_classes from specified super_class."""
        
        self.class_names = []

        for class_name in [f for f in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, f))]:
            meta_file = open(os.path.join(self.data_root, class_name, "meta.json"))
            meta_data = json.load(meta_file)

            if super_class in meta_data["super_class"]:
                self.class_names.append(class_name)

            meta_file.close()

            del self.class_names[num_classes:]
        
        self.class_to_label_id = {self.class_names[i]: i for i in range(len(self.class_names))}

    def _collect_dataset(self):
        """Collect all images, labels, and masks from the dataset (including augmentations)."""

        self.all_images = []
        self.all_labels = []
        self.all_masks = []

        # Collect all real images
        if self.aug_mode not in ["positive_only", "negative_only"]:
            self.all_images, self.all_labels, self.all_masks = self.parse_dataset()

        # Collect all augmentations & OOD images
        if self.aug_mode in ["positive", "both", "positive_only"]:
            positive_augs, positive_labels, positive_masks = self.parse_augs("positive")
            self.all_images += positive_augs
            self.all_labels += positive_labels
            self.all_masks += positive_masks
        if self.aug_mode in ["both", "negative_only"]:
            negative_augs, negative_labels, negative_masks = self.parse_augs("negative")
            self.all_images += negative_augs
            self.all_labels += negative_labels
            self.all_masks += negative_masks
        
        self._length = len(self.all_images)

    def parse_dataset(self):
        """
        Parse the dataset directory and return all real images, labels, and masks.
        
        train:      /mnt/HDD/MVIP/sets/{class_name}/{split}/{set}/{orientation}/{cam}/{file}
        val & test: /mnt/HDD/MVIP/sets/{class_name}/{split}/{set}/{cam}/{file}
        """

        images = []
        labels = []
        masks = []

        split_dir = {
            "train": "train_data",
            "val": "valid_data",
            "test": "test_data",
        }
        
        for class_name in self.class_names:
            root = os.path.join(self.data_root, class_name, split_dir[self.split])
            if self.split == "train":
                for set in [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]:
                    for orientation in [f for f in os.listdir(os.path.join(root, set)) if os.path.isdir(os.path.join(root, set, f))]:
                        for cam in [f for f in os.listdir(os.path.join(root, set, orientation)) if os.path.isdir(os.path.join(root, set, orientation, f))]:
                            for file in os.listdir(os.path.join(root, set, orientation, cam)):
                                if file.endswith("rgb.png"):
                                    images.append(os.path.join(root, set, orientation, cam, file))
                                    labels.append(self.class_to_label_id[class_name])
                                elif file.endswith("rgb_mask_gen.png"):
                                    masks.append(os.path.join(root, set, orientation, cam, file))
            else: # val & test split have no orientation subfolder
                for set in [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]:
                    for cam in [f for f in os.listdir(os.path.join(root, set)) if os.path.isdir(os.path.join(root, set, f))]:
                        for file in os.listdir(os.path.join(root, set, cam)):
                            if file.endswith("rgb.png"):
                                images.append(os.path.join(root, set, cam, file))
                                labels.append(self.class_to_label_id[class_name])
                            elif file.endswith("rgb_mask_gen.png"):
                                masks.append(os.path.join(root, set, cam, file))
        
        return images, labels, masks
    
    def parse_augs(self, aug_type):
        """Parse the augmentation directory and return all augmented images and labels."""

        if aug_type == "positive":
            aug_dir = self.aug_dir_positive
            examples_per_class = self.aug_ex_positive
            label_sign = 1
        else:
            aug_dir = self.aug_dir_negative
            examples_per_class = self.aug_ex_negative
            label_sign = -1 # OOD augmentations receive negative labels
        
        augs = os.listdir(aug_dir)

        # Shuffle before potentially limiting num of examples per class
        shuffle_idx = np.random.permutation(len(augs))
        augs = [os.path.join(aug_dir, augs[i]) for i in shuffle_idx]

        if examples_per_class > 0:
            del augs[examples_per_class*len(self.class_names)*4:] # num_synthetic=4
        print(f"{aug_type}: {len(augs)}")

        # Get labels from file names
        labels = []
        for class_name in self.class_names:
            for file in augs:
                if class_name in file:
                    labels.append(self.class_to_label_id[class_name] * label_sign)

        # Return masks as None
        masks = [None for _ in range(len(augs))]
        
        return augs, labels, masks
    
    def get_mean_std(self):
        mean = torch.zeros(3)
        std = torch.zeros(3)

        for image in self.all_images:
            image = Image.open(image)
            image = self.transform(image)
            mean += torch.mean(image, dim=(1, 2))
            std += torch.std(image, dim=(1, 2))

        mean /= len(self.all_images)
        std /= len(self.all_images)

        return mean, std
    
    def crop_object(self, image: Image, mask: Image):
        """Use object mask to create a square crop around object."""

        # Apply maximum filter to dilate mask
        mask = Image.fromarray(maximum_filter(np.array(mask), size=32))
        
        # Get bounding box of mask
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

        # Crop image with mask_box
        return image.crop(mask_box), mask.crop(mask_box)