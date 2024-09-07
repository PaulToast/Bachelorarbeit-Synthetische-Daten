import argparse
import os

from datasets import MVIPDataset

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
        "--aug_mode",
        type=str,
        default=None, # None, "with_id", "with_both", "id_only", "ood_only"
    )
    parser.add_argument(
        '--aug_output_name',
        type=str,
        default="mvip-v9-final",
        help="DA-Fusion output name for the augmentations."
    )
    parser.add_argument(
        '--aug_name_id',
        type=str,
        default="aug=0.2_ex=16_num=4_g=15",
        help="Name of the subfolder containing the in-distribution augmentations."
    )
    parser.add_argument(
        '--aug_name_ood',
        type=str,
        default="aug=0.5_ex=16_num=4_g=15",
        help="Name of the subfolder containing the out-of-distribution augmentations."
    )
    parser.add_argument(
        "--aug_ex_id",
        type=int,
        default=-1, # -1 for all
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
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

    # Set aug directories
    if args.aug_mode is not None:
        assert args.aug_output_name is not None

        if args.aug_mode == 'with_id':
            assert args.aug_name_id is not None

            args.aug_dir_id = os.path.abspath(os.path.join(
                os.path.dirname( __file__ ), '..', f'da-fusion/output/{args.aug_output_name}/{args.aug_name_id}'
            ))
            args.aug_dir_ood = None
        
        elif args.aug_mode == 'with_both':
            assert args.aug_name_id is not None \
                and args.aug_name_ood is not None
            
            args.aug_dir_id = os.path.abspath(os.path.join(
                os.path.dirname( __file__ ), '..', f'da-fusion/output/{args.aug_output_name}/{args.aug_name_id}'
            ))
            args.aug_dir_ood = os.path.abspath(os.path.join(
                os.path.dirname( __file__ ), '..', f'da-fusion/output/{args.aug_output_name}/{args.aug_name_ood}'
            ))
    else:
        args.aug_dir_id = None
        args.aug_dir_ood = None
    
    # Prepare output directory
    os.makedirs(os.path.abspath(args.output_dir), exist_ok=True)

    return args

if __name__ == "__main__":
    args = parse_args()

    dataset = MVIPDataset(
        split=args.split,
        seed=args.seed,
        aug_mode=args.aug_mode,
        aug_dir_id=args.aug_dir_id,
        aug_dir_ood=args.aug_dir_ood,
        aug_ex_id=args.aug_ex_id,
        aug_ex_ood=-args.augs_ex_ood,
        image_size=args.resolution,
    )

    for i, (image, label) in enumerate(dataset):
        if label < 0:
            dist = "ood"
        else:
            dist = "id"
        class_name = dataset.classes[abs(label)]
        image.save(f"{args.output_dir}/{i}_{class_name}_{dist}.png")