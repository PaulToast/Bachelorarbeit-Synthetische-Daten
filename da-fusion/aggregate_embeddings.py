import torch
import os
import glob
import argparse
from itertools import product
from tqdm import trange


DEFAULT_EMBED_PATH = "sd-fine-tuned-merged/{dataset}-{seed}-{examples_per_class}.pt"


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Merge token files")

    parser.add_argument("--num_trials", type=int, default=8)
    parser.add_argument("--examples_per_class", nargs='+', type=int, default=[1, 2, 4, 8, 16])
    
    parser.add_argument("--embed_path", type=str, default=DEFAULT_EMBED_PATH)
    parser.add_argument("--input_path", type=str, default=None) ##
    
    parser.add_argument("--dataset", type=str, default="mvip", 
                        choices=["spurge", "imagenet", "coco", "pascal", "mvip"])

    args = parser.parse_args()

    if args.embed_path == DEFAULT_EMBED_PATH:
        args.embed_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '_data', args.dataset, DEFAULT_EMBED_PATH))
    if args.input_path == None:
        args.input_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '_data', args.dataset, "sd-fine-tuned"))

    for seed, examples_per_class in product(
            range(args.num_trials), args.examples_per_class):

        path = os.path.join(args.input_path, (
            f"{args.dataset}-{seed}-{examples_per_class}/*/learned_embeds.bin"))

        merged_dict = dict()
        for file in glob.glob(path):
            merged_dict.update(torch.load(file))

        target_path = args.embed_path.format(
            dataset=args.dataset,
            seed=seed, 
            examples_per_class=examples_per_class)

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        torch.save(merged_dict, target_path)