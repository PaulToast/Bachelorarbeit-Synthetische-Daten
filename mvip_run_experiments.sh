conda activate synt-contrast

# Contrastive pre-training stage

# Run 1: With real images only
python sup_contrast/main_supcon.py --dataset=mvip --output_name="mvip-test-supcon-pre" \
--trial=0 --model="resnet-50" --method="SupCon" --image_size=224 --epochs=110 --batch_size=16 \
--num_workers=4 --lr=0.001 --lr_cosine
# Run 2: With real images & in-distribution augs
python sup_contrast/main_supcon.py --dataset=mvip --output_name="mvip-test-supcon-pre" \
--trial=1 --aug_mode="with_id" --aug_output_name="mvip-test-dafusion" --aug_name_id="augs-id" \
--model="resnet-50" --method="SupCon" --image_size=224 --epochs=110 --batch_size=16 \
--num_workers=4 --lr=0.001 --lr_cosine
# Run 3: With real images, in-distribution & out-of-distribution augs
python sup_contrast/main_supcon.py --dataset=mvip --output_name="mvip-test-supcon-pre" \
--trial=2 --aug_mode="with_both" --aug_output_name="mvip-test-dafusion" --aug_name_id="augs-id" \
--aug_name_ood="augs-ood" --model="resnet-50" --method="SupCon" --image_size=224 \
--epochs=110 --batch_size=16 --num_workers=4 --lr=0.001 --lr_cosine

# Linear classification stage

# Run 1: With real images only
python sup_contrast/main_linear.py --dataset=mvip --output_name="mvip-test-supcon-lin" \
--trial=0 --model="resnet-50" --ckpt="output/mvip-test-supcon-pre/trial=0/last.ckpt" \
--image_size=224 --epochs=50 --batch_size=16 --num_workers=4 --lr=0.001 --lr_cosine
# Run 2: With real images & in-distribution augs for both pre-training & linear stage
python sup_contrast/main_linear.py --dataset=mvip --output_name="mvip-test-supcon-lin" \
--trial=1 --aug_mode="with_id" --aug_output_name="mvip-test-dafusion" --aug_name_id="augs-id" \
--model="resnet-50" --ckpt="output/mvip-test-supcon-pre/trial=1/last.ckpt" \
--image_size=224 --epochs=50 --batch_size=16 --num_workers=4 --lr=0.001 --lr_cosine
# Run 3: Pre-training with real images, in-distribution & out-of-distribution augs,
#        linear stage with real images & in-distribution augs
python sup_contrast/main_linear.py --dataset=mvip --output_name="mvip-test-supcon-lin" \
--trial=2 --aug_mode="with_both" --aug_output_name="mvip-test-dafusion" --aug_name_id="augs-id" \
--aug_name_ood="augs-ood" --model="resnet-50" --ckpt="output/mvip-test-supcon-pre/trial=2/last.ckpt" \
--image_size=224 --epochs=50 --batch_size=16 --num_workers=4 --lr=0.001 --lr_cosine