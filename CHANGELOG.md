# Changelog

## 1.0.1

- Align sampling weights with compact training views and retain the complete requested batch.
- Exclude validation replicas from sampling weights, correct per-hand sample counts, and version the sampling protocol for safe resume checks.
- Validate a full seed-42 retraining and evaluation: 74/128 successful held-out hands versus 71/128 for the frozen-recipe reproduction, with slightly lower safe completion and mean net turns.
- Keep v1.0.0, the original reference models, and the paper's formal results available unchanged.

## 1.0.0

- Release the paper's asset generation, geometry pretraining, family teachers, and shared BC student implementation.
- Provide versioned model and 384-hand asset downloads, with portable paths and checksum verification.
- Add unified replay, teacher collection, student training, and fixed evaluation commands.
- Preserve the frozen paper sampler and record deterministic BC execution settings.
- Document the quick replay and shared-student reproduction workflows, with optional asset generation and pretraining guides.
