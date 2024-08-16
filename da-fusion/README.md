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

To benchmark DA-Fusion, we use a classification task derived from COCO (2017).

Download the [2017 Training Images](http://images.cocodataset.org/zips/train2017.zip), the [2017 Validation Images](http://images.cocodataset.org/zips/val2017.zip), and the [2017 Train/Val Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).

These files should be unzipped into the top-level folder `_datasets/` with the following directory structure:

```
coco2017/
    train2017/
    val2017/
    annotations/
```

To set up custom datasets, add them to the same `_datasets` directory and implement a subclass of `semantic_aug/few_shot_dataset.py` (since every dataset may have a different structure and form - you can take `semantic_aug/datasets/coco.py` as an example.)

## Fine-Tuning Tokens

We perform [Textual Inversion](https://arxiv.org/abs/2208.01618) to adapt Stable Diffusion to the classes present in our few-shot datasets. The implementation in `fine_tune.py` is adapted from the [Diffusers](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py) example.

The `fine_tune_upstream.py` script seems to be an updated version of the script with a couple of modifications and more parameters. Here is an example for executing the script on the MVIP dataset, as it was used for the thesis:

```bash
python fine_tune_upstream.py --dataset=mvip --experiment_name="test-01" \
--pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
--initializer_token="component" --validation_prompt="a photo of a {name}" \
--num_vectors=4 --resolution=512 --train_batch_size=8 --lr_warmup_steps=0 \
--gradient_accumulation_steps=1 --max_train_steps=1000 \
--learning_rate=5.0e-04 --scale_lr --lr_scheduler="constant" \
--mixed_precision=fp16 --revision=fp16 --gradient_checkpointing \
--num_trials=1 --examples_per_class=16
```

The trained token weights will be saved under `_experiments/{dataset}-{experiment_name}/fine-tuned`.

## Aggregate Embeddings

Before generating augmentations, we call the script `aggregate_embeddings.py`, which merges all of the learned tokens together into a single directory, creating a class-agnostic template to use for the next steps:

```bash
python aggregate_embeddings.py --dataset=mvip --experiment_name="test-01" \
--num_trials=1 --examples_per_class=16
```

The results will be saved under `_experiments/{dataset}-{experiment_name}/fine-tuned-merged`.

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
python generate_augmentations.py --dataset=mvip -experiment_name="test-01" \
--model_path="CompVis/stable-diffusion-v1-4" \
--examples-per-class=1 --num-synthetic=4 --strength=0.5
```

The images will be saved under `_experiments/{dataset}-{experiment_name}/aug`

There are several ways to configure the augmentation:

- `--prompt`: A custom prompt can be specified; Will default to "a photo of a {name}".
- `--aug`: The method for generating the augmentations, either "real-guidance" or "textual-inversion"
- `--guidance_scale`: Represents how much importance is given to your prompt when generating images. Lower values will give less attention to the prompt, but generate more freely. Will default to 7.5
- ...