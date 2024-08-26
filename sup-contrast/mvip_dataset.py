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
        aug_dir_positive=None,
        aug_dir_negative=None,
        size=224,
        transform=None,
        repeats=1, #da-fusion: 100
    ):
        self.data_root = '/mnt/HDD/MVIP/sets'
        self.aug_dir_positive = aug_dir_positive
        self.aug_dir_negative = aug_dir_negative

        self.split = split

        self.size = size

        # Define class names list; Limit dataset to 20 classes from the "CarComponent" super class
        self.class_names = []

        for class_name in [f for f in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, f))]:
            meta_file = open(os.path.join(self.data_root, class_name, "meta.json"))
            meta_data = json.load(meta_file)

            if "CarComponent" in meta_data['super_class']:
                self.class_names.append(class_name)

            meta_file.close()

            del self.class_names[20:]
        
        self.class_to_label_id = {self.class_names[i]: i for i in range(len(self.class_names))}

        # Collect all real images
        self.all_images, self.all_masks, self.all_labels = self.parse_dataset(self.class_names, self.split)

        # Collect all augmentations
        if self.split == "train":
            if self.aug_dir_positive is not None:
                self.all_augs_positive, self.all_augs_labels_positive = self.parse_augs(self.class_names, self.aug_dir_positive)
                for i in range(len(self.all_augs_positive)):
                    self.all_images.append(self.all_augs_positive[i])
                    self.all_masks.append(None)
                    self.all_labels.append(self.all_augs_labels_positive[i])
            """if self.aug_dir_negative is not None:
                self.all_augs_negative, self.all_augs_labels_negative = self.parse_augs(self.class_names, self.aug_dir_negative)
                for i in range(len(self.all_augs_negative)):
                    self.all_images.append(self.all_augs_negative[i])
                    self.all_masks.append(None)
                    self.all_labels.append(-1)"""

        self._length = len(self.all_images)

        # Shuffle dataset
        np.random.seed(0)
        shuffle_idx = np.random.permutation(self._length)
        self.all_images = [self.all_images[i] for i in shuffle_idx]
        self.all_masks = [self.all_masks[i] for i in shuffle_idx]
        self.all_labels = [self.all_labels[i] for i in shuffle_idx]

        # Set transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(self.size, scale=(0.8, 1.)),# ratio=(1.0, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # Get label
        label = self.all_labels[idx % self._length]

        # Get image
        image = Image.open(self.all_images[idx % self._length])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        # Get object mask & use maximum filter to dilate it
        if self.all_masks[idx % self._length] is not None:
            mask = np.array(Image.open(self.all_masks[idx % self._length]).convert('L'))
            mask = Image.fromarray(maximum_filter(mask, size=32))

            # Use mask to crop image
            image = self.mask_crop(image, mask)

        return self.transform(image), label
    
    def parse_dataset(self, class_names, split):
        images = []
        masks = []
        labels = []

        split_dir = {
            "train": "train_data",
            "val": "valid_data",
            "test": "test_data",
        }

        for class_name in class_names:
            root = os.path.join(self.data_root, class_name, split_dir[split])

            if split == "train_data":
                for set in [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]:
                    for orientation in [f for f in os.listdir(os.path.join(root, set)) if os.path.isdir(os.path.join(root, set, f))]:
                        for cam in [f for f in os.listdir(os.path.join(root, set, orientation)) if os.path.isdir(os.path.join(root, set, orientation, f))]:
                            for file in os.listdir(os.path.join(root, set, orientation, cam)):
                                if file.endswith("rgb.png"):
                                    images.append(os.path.join(root, set, orientation, cam, file))
                                    labels.append(self.class_to_label_id[class_name])
                                elif file.endswith("rgb_mask_gen.png"):
                                    masks.append(os.path.join(root, set, orientation, cam, file))
                
            else:
                for set in [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]:
                    for cam in [f for f in os.listdir(os.path.join(root, set)) if os.path.isdir(os.path.join(root, set, f))]:
                        for file in os.listdir(os.path.join(root, set, cam)):
                            if file.endswith("rgb.png"):
                                images.append(os.path.join(root, set, cam, file))
                                labels.append(self.class_to_label_id[class_name])
                            elif file.endswith("rgb_mask_gen.png"):
                                masks.append(os.path.join(root, set, cam, file))
        
        return images, masks, labels
    
    def parse_augs(self, class_names, aug_dir):
        augs = []
        labels = []

        for class_name in class_names:
            for file in os.listdir(aug_dir):
                if class_name in file:
                    augs.append(os.path.join(aug_dir, file))
                    labels.append(self.class_to_label_id[class_name])
        
        return augs, labels
    
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
    
    def mask_crop(self, image: Image, mask: Image):
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
        return image.crop(mask_box)