# Geometry pretraining

The default model package includes the pretrained geometry encoder used by both the family teachers and the shared student. Pretraining uses two self-supervised objectives: a continuous surface-proximity field and joint-angle derivatives of relations between fixed surface points and palm anchors. FairGrad balances gradient updates between both tasks, and the trained encoder is frozen during policy learning.

## Prepare a geometry dataset

Follow the [asset generation guide](assets.md) to generate a typed hand dataset and its SSL manifest. The manifest specifies the URDF, `hand.yaml` metadata, mesh paths, and train/validation splits. Pretraining computes geometric targets from sampled joint configurations and hand surfaces. It does not require Isaac Sim or pregrasp catalogs.

The paper's encoder was pretrained on 8,192 generated hand assets. The default 384-hand asset download is designed for teacher demonstration collection and policy evaluation. To follow the paper's data recipe, generate the full geometry dataset and provide its manifest path explicitly.

## Run the paper configuration

Run from the AnyMani repository root in the installed Python environment:

```bash
python -m anymani.distill.ssl.pretrain \
  --config geometry_ssl_density_material_jacobian_se3_v0_8_1 \
  --dataset-manifest /absolute/path/to/ssl.yaml \
  --output_dir outputs/geometry-pretraining \
  --device cuda:0
```

The pretraining script logs the selected manifest's SHA-256 hash. To enforce checksum verification before training begins, pass `--dataset-sha256 <expected-sha256>`. When launching a new dataset run under an existing output directory, add `--new_run` to create a separate experiment directory.

| Setting | Paper configuration |
| --- | --- |
| Random seed | 20260830 |
| Budget | 512 epochs, 4 updates per epoch |
| Fresh asset-state pairs per update | 64 hands × 8 configurations |
| Encoder | 4 transformer layers, width 128, 4 attention heads |
| Optimizer | AdamW, learning rate 0.0003 |
| Precision | FP32 parameters and losses, BF16 model autocast, TF32 disabled |
| Checkpoints | Every 32 epochs |

Each run outputs to `outputs/geometry-pretraining/<experiment-name>/<UTC timestamp>/`, storing the resolved configuration, dataset checksum, training metrics, periodic checkpoints, and the exported encoder. The complete training schedule executes 2,048 updates across 1,048,576 dynamically sampled asset-state pairs. Newly generated datasets create a new run directory; the supplied frozen encoder remains the reference for the paper's control results.

To resume an interrupted run from its recovery checkpoint:

```bash
python -m anymani.distill.ssl.pretrain \
  --config geometry_ssl_density_material_jacobian_se3_v0_8_1 \
  --dataset-manifest /absolute/path/to/ssl.yaml \
  --resume_checkpoint /absolute/path/to/run/checkpoints/recovery.pt
```

The complete experiment configuration is implemented in [`geometry_ssl_density_material_jacobian_se3_v0_8_1.py`](../source/anymani/anymani/distill/ssl/experiments/geometry_ssl_density_material_jacobian_se3_v0_8_1.py), including surface point sampling, dual prediction heads, FairGrad hyperparameters, and the optimizer schedule.
