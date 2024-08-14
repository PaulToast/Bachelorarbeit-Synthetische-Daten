import os
from collections import defaultdict

MVIP_DIR = "/mnt/HDD/MVIP/sets"
split = "train"

class_names = [f for f in os.listdir(MVIP_DIR) if os.path.isdir(os.path.join(MVIP_DIR, f))]

split_dir = {
    "train": "train_data",
    "test": "test_data",
    "val": "valid_data"
} 

class_to_images = defaultdict(list)

for class_name in class_names:
    root = os.path.join(MVIP_DIR, class_name, split_dir[split])

    # Go through every set, orientation, cam and select only the rgb images
    for set in [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]:
        for orientation in [f for f in os.listdir(os.path.join(root, set)) if os.path.isdir(os.path.join(root, set, f))]:
            for cam in [f for f in os.listdir(os.path.join(root, set, orientation)) if os.path.isdir(os.path.join(root, set, orientation, f))]:
                for file in os.listdir(os.path.join(root, set, orientation, cam)):
                    if file.endswith("rgb.png"):
                        class_to_images[class_name].append(os.path.join(root, set, orientation, cam, file))

print(class_to_images[class_names[0]])