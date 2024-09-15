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