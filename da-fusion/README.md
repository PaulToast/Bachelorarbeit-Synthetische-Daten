# Effective Data Augmentation With Diffusion Models

This is a modified version of the [original DA-Fusion repository](https://github.com/brandontrabucco/da-fusion).

It is embedded into the main repository for this thesis work, but is used in isolation.

[DA-Fusion Website](btrabuc.co/da-fusion)     |     [Paper](https://openreview.net/forum?id=ZWzUA9zeAg)

## Installation

First set the following channel configuration the `/home/username/.condarc` file:

```
channel_priority: strict
channels:
  - nvidia
  - pytorch
  - conda-forge
  - defaults
```

Now create the `conda` environment.

```bash
conda create -n da-fusion python=3.7 pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6
conda activate da-fusion
pip install diffusers["torch"] transformers pycocotools pandas matplotlib seaborn scipy
```

Then install the source code of this repository.

```bash
pip install -e da-fusion
```

## Setting up the Dataset

First we benchmark DA-Fusion on a classification task derived from COCO (2017).

Custom datasets can be evaluated by implementing subclasses of `semantic_aug/few_shot_dataset.py`.

To setup COCO, first download the [2017 Training Images](http://images.cocodataset.org/zips/train2017.zip), the [2017 Validation Images](http://images.cocodataset.org/zips/val2017.zip), and the [2017 Train/Val Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip). These files should be unzipped into the following directory structure.

```
coco2017/
    train2017/
    val2017/
    annotations/
```

`COCO_DIR` located at `semantic_aug/datasets/coco.py` (Line 15) should be updated to point to the location of `coco2017` on your system.

## Fine-Tuning Tokens

```bash
python fine_tune.py
  --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5"
  --dataset coco
  --train_batch_size 64
```

We perform [Textual Inversion](https://arxiv.org/abs/2208.01618) to adapt Stable Diffusion to the classes present in our few-shot datasets. The implementation in `fine_tune.py` is adapted from the [Diffusers](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py) example.

We wrap this script for distributing experiments on a slurm cluster in a set of `sbatch` scripts located at `scripts/fine_tuning`. These scripts will perform multiple runs of Textual Inversion in parallel, subject to the number of available nodes on your slurm cluster.

If `sbatch` is not available in your system, you can run these scripts with `bash` and manually set `SLURM_ARRAY_TASK_ID` and `SLURM_ARRAY_TASK_COUNT` for each parallel job (these are normally set automatically by slurm to control the job index, and the number of jobs respectively, and can be set to 0, 1).

## Few-Shot Classification

Code for training image classification models using augmented images from DA-Fusion is located in `train_classifier.py`. This script accepts a number of arguments that control how the classifier is trained:

```bash
python train_classifier.py --logdir pascal-baselines/textual-inversion-0.5 \
--synthetic-dir "aug/textual-inversion-0.5/{dataset}-{seed}-{examples_per_class}" \
--dataset pascal --prompt "a photo of a {name}" \
--aug textual-inversion --guidance-scale 7.5 \
--strength 0.5 --mask 0 --inverted 0 \
--num-synthetic 10 --synthetic-probability 0.5 \
--num-trials 1 --examples-per-class 4
```

This example will train a classifier on the PASCAL VOC task, with 4 images per class, using the prompt `"a photo of a ClassX"` where the special token `ClassX` is fine-tuned (from scratch) with textual inversion. Slurm scripts that reproduce the paper are located in `scripts/textual_inversion`. Results are logged to `.csv` files based on the script argument `--logdir`. 

We used a [custom plotting script](https://github.com/brandontrabucco/da-fusion/blob/main/plot.py) to generate the figures in the main paper.