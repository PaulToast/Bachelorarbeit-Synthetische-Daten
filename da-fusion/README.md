# Effective Data Augmentation With Diffusion Models

Modified version of the [original DA-Fusion repository](https://github.com/brandontrabucco/da-fusion) ( [Website](btrabuc.co/da-fusion) | [Paper](https://openreview.net/forum?id=ZWzUA9zeAg) )

## Installation

First create the `conda` environment.

```bash
conda create -n da-fusion python=3.7 pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6 -c nvidia -c pytorch -c conda-forge
conda activate da-fusion
pip install diffusers["torch"] transformers pycocotools pandas matplotlib seaborn scipy
```

Run `conda list` to check if the correct GPU/CUDA-versions of the packages `pytorch`, `torchvision`, etc. were installed. If not, set the conda channel configuration manually in the `/home/username/.condarc` file, declaring `channel_priority: strict`:

```
channel_priority: strict
channels:
  - nvidia
  - pytorch
  - conda-forge
  - defaults
```

Then install the source code of this repository.

```bash
pip install -e da-fusion
```

## Setting up benchmark & custom datasets

Datasets should be placed into the directory `da-fusion/_data/{dataset}`. This is also where the trained weights & image augmentations will be saved for each dataset.

To benchmark DA-Fusion, we use a classification task derived from COCO (2017).

Download the [2017 Training Images](http://images.cocodataset.org/zips/train2017.zip), the [2017 Validation Images](http://images.cocodataset.org/zips/val2017.zip), and the [2017 Train/Val Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).

These files should be unzipped into the top-level folder `_data/` with the following directory structure:

```
coco2017/
    train2017/
    val2017/
    annotations/
```

To set up custom datasets, add them to the same `_data` directory and implement a subclass of `semantic_aug/few_shot_dataset.py` (since every dataset may have a different structure and form - you can take `semantic_aug/datasets/coco.py` as an example.)

## Fine-Tuning Tokens

We perform [Textual Inversion](https://arxiv.org/abs/2208.01618) to adapt Stable Diffusion to the classes present in our few-shot datasets. The implementation in `fine_tune.py` is adapted from the [Diffusers](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py) example.

Here is an example for executing the `fine_tune.py` script on the COCO dataset:

```bash
python fine_tune.py --dataset=coco \
--pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
--resolution=512 --train_batch_size=4 --lr_warmup_steps=0 \
--gradient_accumulation_steps=1 --max_train_steps=1000 \
--learning_rate=5.0e-04 --scale_lr --lr_scheduler="constant" \
--mixed_precision=fp16 --revision=fp16 --gradient_checkpointing \
--only_save_embeds --num-trials 8 --examples-per-class 1 2 4 8 16 
```

## Aggregate Embeddings

After the previous step, we should be able to find the learned class tokens in the directory `_data/{dataset}/sd-fine-tuned/`.

In order to use them, we will call the script `aggregate_embeddings.py`, which merges all of them together into a single directory, creating a class-agnostic template to use for the next steps. Make sure to use the parameter `--dataset` to point to the correct dataset, as in the previous steps.

## Generate Augmentations

In order to generate augmentations, a connection to the HuggingFace API has to be set up.

Run the following in your terminal:

```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

You will be prompted for your HuggingFace account credentials, and for an access token that you may create for your account by following [this guide](https://huggingface.co/docs/huggingface_hub/quick-start#login).

Afterwards, we can call the `generate_augmentations.py` script with the according parameters, for example:

```bash
python generate_augmentations.py --dataset=coco \
--examples-per-class=1 --num-synthetic=4 --strength=0.5
```