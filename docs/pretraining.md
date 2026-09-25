# Geometry pretraining

The default model download includes the frozen geometry encoder used by the teachers and shared student. Pretraining learns two objectives: a continuous surface-proximity field and the joint-driven motion of fixed material points. FairGrad balances their gradients in the shared encoder. The encoder is frozen during policy learning.

## Prepare a geometry dataset

Follow [asset generation](assets.md) to build a typed hand dataset and its SSL manifest. Each manifest resolves the URDF, `hand.yaml`, meshes, and train/validation partitions. Geometry pretraining samples joint configurations and geometric targets directly; it does not require Isaac Sim or pregrasp caches.

The paper's encoder used 8,192 generated training assets. The default 384-hand control bundle supports teacher collection and student evaluation. For a new pretraining run, generate the larger geometry dataset and pass its manifest explicitly.

## Run the paper configuration

Run from the AnyMani repository root in the installed Python environment:

```bash
python -m anymani.distill.ssl.pretrain \
  --config geometry_ssl_density_material_jacobian_se3_v0_8_1 \
  --dataset-manifest /absolute/path/to/ssl.yaml \
  --output_dir outputs/geometry-pretraining \
  --device cuda:0
```

The command records the selected manifest's SHA-256. To verify a known manifest before training, also pass `--dataset-sha256 <expected-sha256>`. When changing datasets under the same output directory, add `--new_run` to start a separate run.

| Setting | Paper configuration |
| --- | --- |
| Random seed | 20260830 |
| Budget | 512 epochs, 4 updates per epoch |
| Fresh asset-state pairs per update | 64 hands × 8 configurations |
| Encoder | 4 transformer layers, width 128, 4 attention heads |
| Optimizer | AdamW, learning rate 0.0003 |
| Precision | FP32 parameters and losses, BF16 model autocast, TF32 disabled |
| Checkpoints | Every 32 epochs |

Each run writes to `outputs/geometry-pretraining/<experiment-name>/<UTC timestamp>/`, containing the resolved configuration, dataset identity, learning records, checkpoints, and exported encoder. The full schedule uses 2,048 updates and 1,048,576 fresh asset-state pairs. Newly generated datasets create a new training run; the supplied frozen encoder remains the reference for the paper's control results.

An interrupted run can continue from its recovery checkpoint:

```bash
python -m anymani.distill.ssl.pretrain \
  --config geometry_ssl_density_material_jacobian_se3_v0_8_1 \
  --dataset-manifest /absolute/path/to/ssl.yaml \
  --resume_checkpoint /absolute/path/to/run/checkpoints/recovery.pt
```

The training snapshot is in [`geometry_ssl_density_material_jacobian_se3_v0_8_1.py`](../source/anymani/anymani/distill/ssl/experiments/geometry_ssl_density_material_jacobian_se3_v0_8_1.py). It defines the geometry sampling, two prediction heads, FairGrad settings, and optimizer schedule.
