# 预实验运行
cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate

python -m anymani.distill.ssl.pretrain \
  --phase calibrate_objectives \
  --num_minibatches 128 \
  --assets_per_minibatch 64 \
  --q_per_asset_per_minibatch 8 \
  --mini_epochs 1 \
  --gradient_accumulation_steps 4 \
  --max_resident_assets 64 \
  --seed 20260813 \
  --device cuda:0 \
  --output_dir logs/ssl \
  --experiment_name canonical_multi_anchor_gaussian_preexperiment
