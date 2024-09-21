def parse_augs(self, aug_type):
    #Parse the augmentation directory and return all augmented images and labels.

    if aug_type == "id":
        aug_dir = self.aug_dir_id
        examples_per_class = self.aug_ex_id
        label_sign = 1
    else: # OODs
        aug_dir = self.aug_dir_ood
        examples_per_class = self.aug_ex_ood
        label_sign = -1 # OOD augmentations receive negative labels

    augs = os.listdir(aug_dir)

    # Shuffle before potentially limiting num of examples per class
    shuffle_idx = np.random.permutation(len(augs))
    augs = [os.path.join(aug_dir, augs[i]) for i in shuffle_idx]

    if examples_per_class > 0:
        del augs[examples_per_class*len(self.class_names)*4:] # num_synthetic=4

    # Get labels from file names
    labels = []
    for class_name in self.class_names:
        for file in augs:
            if class_name in file:
                labels.append(self.class_to_label_id[class_name] * label_sign)

    # Return masks as None
    masks = [None for _ in range(len(augs))]

    return augs, labels, masks