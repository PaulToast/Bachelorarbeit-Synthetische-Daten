# Select subset of classes from MVIP dataset
class_names = []
class_descriptions = []

for class_name in [
        f for f in os.listdir(MVIP_DIR) if os.path.isdir(os.path.join(MVIP_DIR, f))
    ]:
    meta_file = open(os.path.join(MVIP_DIR, class_name, "meta.json"))
    meta_data = json.load(meta_file)

    if SUPER_CLASS in meta_data['super_class']:
        class_names.append(class_name)
        class_descriptions.append(meta_data['description'])

    meta_file.close()

    # Limit the number of classes
    del class_names[NUM_CLASSES:]

    num_classes: int = len(class_names)