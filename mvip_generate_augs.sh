conda activate synt-contrast

# Fine-tune stable diffusion by learning new tokens
python da_fusion/fine_tune.py --dataset=mvip --output_name="mvip-test-dafusion" \
--pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
--initializer_token="motor" --num_vectors=16 --validation_prompt="a photo of a {}" \
--resolution=512 --crop_object --train_batch_size=16 \
--lr=5.0e-04 --lr_warmup_steps=150 --lr_scheduler="constant_with_warmup" \
--gradient_accumulation_steps=1 --max_train_steps=1000 \
--mixed_precision=fp16 --revision=fp16 --gradient_checkpointing \
--num_trials=1 --examples_per_class=32

# Merge learned tokens
python da_fusion/aggregate_embeddings.py --output_name="mvip-test-dafusion" \
--num_trials=1 --examples_per_class=32

# Generate in-distribution augmentations (strenth=0.2)
python da_fusion/generate_augmentations.py --dataset=mvip --output_name="mvip-test-dafusion" \
--aug_name="augs-id" --model_path="CompVis/stable-diffusion-v1-4" \
--examples_per_class=16 --num_synthetic=4 --strength=0.2 --guidance_scale=15

# Generate out-of-distribution augmentations (strength=0.5)
python da_fusion/generate_augmentations.py --dataset=mvip --output_name="mvip-test-dafusion" \
--aug_name="augs-ood" --model_path="CompVis/stable-diffusion-v1-4" \
--examples_per_class=16 --num_synthetic=4 --strength=0.5 --guidance_scale=15