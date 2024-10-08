import torch
import os
import glob
import argparse
from itertools import product
from tqdm import trange


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Merge token files")

    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--examples_per_class",
        nargs='+',
        type=int,
        default=[1, 2, 4, 8, 16],
    )

    args = parser.parse_args()

    input_path = os.path.abspath(
        os.path.join("output", f"{args.output_name}", "fine-tuned")
    )
    output_path = os.path.abspath(
        os.path.join("output", f"{args.output_name}", "fine-tuned-merged/seed={seed}_ex={examples_per_class}.pt")
    )

    for seed, examples_per_class in product(
            range(args.num_trials), args.examples_per_class):

        path = os.path.join(input_path, (
            f"seed={seed}-ex={examples_per_class}/*/learned_embeds.bin"))

        merged_dict = dict()
        for file in glob.glob(path):
            merged_dict.update(torch.load(file))

        target_path = output_path.format(
            seed=seed, 
            examples_per_class=examples_per_class)

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        torch.save(merged_dict, target_path)