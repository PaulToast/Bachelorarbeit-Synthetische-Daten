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
        size=224,
        transform=None,
        repeats=1, # da-fusion: 100
        synt=None
    ):
        self.data_root = '/mnt/HDD/MVIP/sets'

        split_dir = {
            "train": "train_data",
            "val": "valid_data",
            "test": "test_data",
        }
        self.split = split_dir[split]

        self.size = size

        # Limit dataset to 20 classes from the "CarComponent" super class
        self.class_names = []

        for class_name in [f for f in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, f))]:
            meta_file = open(os.path.join(self.data_root, class_name, "meta.json"))
            meta_data = json.load(meta_file)

            if "CarComponent" in meta_data['super_class']:
                self.class_names.append(class_name)

            meta_file.close()

            del self.class_names[20:]

        self.all_images, self.all_masks = self.get_all_image_paths(self.class_names, self.split)
        self.num_images = len(self.all_images)

        self.class_to_label_id = {self.class_names[i]: i for i in range(len(self.class_names))}
        self.all_labels = [self.class_to_label_id[self.all_images[i].split("/")[-6 if split == "train" else -5]] for i in range(self.num_images)]
        # Example: "/mnt/HDD/MVIP/sets/ >CLASS_NAME< /train_data/0/0/cam0/0_rgb.png"
        # Example: "/mnt/HDD/MVIP/sets/ >CLASS_NAME< /valid_data/0/cam0/0_rgb.png"

        # Shuffle dataset
        np.random.seed(0)
        shuffle_idx = np.random.permutation(self.num_images)
        self.all_images = [self.all_images[i] for i in shuffle_idx]
        self.all_masks = [self.all_masks[i] for i in shuffle_idx]
        self.all_labels = [self.all_labels[i] for i in shuffle_idx]

        if split == "train":
            self._length = self.num_images * repeats
        else:
            self._length = self.num_images

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
        label = self.all_labels[idx % self.num_images]

        # Get image
        image = Image.open(self.all_images[idx % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        # Get object mask & use maximum filter to dilate it
        mask = np.array(Image.open(self.all_masks[idx % self.num_images]).convert('L'))
        mask = Image.fromarray(maximum_filter(mask, size=32))

        # Use mask to crop image
        image = self.mask_crop(image, mask)

        return self.transform(image), label
    
    def get_all_image_paths(self, class_names, split):
        images = []
        masks = []

        for class_name in class_names:
            root = os.path.join(self.data_root, class_name, split)

            for set in [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]:
                if split == "train_data":
                    for orientation in [f for f in os.listdir(os.path.join(root, set)) if os.path.isdir(os.path.join(root, set, f))]:
                        for cam in [f for f in os.listdir(os.path.join(root, set, orientation)) if os.path.isdir(os.path.join(root, set, orientation, f))]:
                            for file in os.listdir(os.path.join(root, set, orientation, cam)):
                                if file.endswith("rgb.png"):
                                    images.append(os.path.join(root, set, orientation, cam, file))
                                elif file.endswith("rgb_mask_gen.png"):
                                    masks.append(os.path.join(root, set, orientation, cam, file))
                else:
                    for cam in [f for f in os.listdir(os.path.join(root, set)) if os.path.isdir(os.path.join(root, set, f))]:
                        for file in os.listdir(os.path.join(root, set, cam)):
                            if file.endswith("rgb.png"):
                                images.append(os.path.join(root, set, cam, file))
                            elif file.endswith("rgb_mask_gen.png"):
                                masks.append(os.path.join(root, set, cam, file))
        
        return images, masks
    
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