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
        default=-1, # -1 for full dataset
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
        image_size=(args.resolution, args.resolution),
    )
    print("length: ", len(dataset))

    for i, (image, label) in enumerate(dataset):
        class_name = dataset.class_names[label]
        os.makedirs(os.path.abspath(os.path.join(args.output_dir, class_name)), exist_ok=True)
        image.save(f"{args.output_dir}/{class_name}/{i}.png")