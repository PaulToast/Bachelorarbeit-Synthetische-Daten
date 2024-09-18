import argparse
import os

from semantic_aug.datasets.mvip import MVIPDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Export MVIPDataset.")
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/mvip",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--examples_per_class",
        type=int,
        default=None, # -1 for full dataset
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--crop_object",
        type=bool,
        default=True,
        help="Crop object from image before resizing to resolution (for MVIP dataset).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    # Prepare output directory
    os.makedirs(os.path.abspath(args.output_dir), exist_ok=True)

    return args

if __name__ == "__main__":
    args = parse_args()

    dataset = MVIPDataset(
        split=args.split,
        seed=args.seed,
        examples_per_class=args.examples_per_class,
        image_size=(args.resolution, args.resolution)
    )
    print(f"length: {len(dataset)}")

    for idx, description in enumerate(dataset.class_descriptions):
        print(f"{dataset.class_names[idx]}: {description}")
        print()

    for idx in range(len(dataset)):
        image = dataset.get_image_by_idx(idx)
        label = dataset.get_label_by_idx(idx)
        class_name = dataset.class_names[label]

        os.makedirs(os.path.abspath(os.path.join(args.output_dir, class_name)), exist_ok=True)
        image.save(f"{args.output_dir}/{class_name}/{idx}.png")
    
    for class_name in [f for f in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, f))]:
        print(f"{class_name}: ")