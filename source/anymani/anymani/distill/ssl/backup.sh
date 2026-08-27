#!/usr/bin/env bash
set -euo pipefail

# Geometry SSL Multitask Representation v0.7.3
# 实验语义唯一来源：source/anymani/anymani/distill/ssl/experiments/geometry_ssl_multitask_representation_v0_7_3.py
#
# 一条命令完成 source cache 复用/补建和正式训练：
#
#   ./source/anymani/anymani/distill/ssl/backup.sh

cd /home/hac/isaac/AnyMani
source /home/hac/isaac/env_isaaclab/bin/activate

/home/hac/isaac/IsaacLab/isaaclab.sh -p -m anymani.distill.ssl.pretrain \
  --config geometry_ssl_multitask_representation_v0_7_3 \
  --device cuda:0 \
  --seed 20260813
