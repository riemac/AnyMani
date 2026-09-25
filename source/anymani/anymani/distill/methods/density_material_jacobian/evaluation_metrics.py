"""Physical metrics for density and material Jacobian evaluation."""


from __future__ import annotations

from typing import Any

import torch


class DensityPhysicalMetricAccumulator:


    tolerance_m = (0.001, 0.002, 0.004, 0.008)  # 1/2/4/8 mm tolerance curve
    posed_success_fractions = (0.80, 0.90)
    groups = ("all", "workspace", "shell", "adjacent", "palm", "joint", "tip")

    def __init__(self) -> None:


        self.group_statistics: dict[str, dict[str, Any]] = {
            group: {
                "count": 0,
                "absolute_sum_m": 0.0,
                "square_sum_m2": 0.0,
                "tolerance_counts": [0, 0, 0, 0],
                "histogram": torch.zeros(500, dtype=torch.float64),
                "histogram_overflow": 0,
            }
            for group in self.groups
        }
        self.posed_success_count = {
            fraction: [0, 0, 0, 0]
            for fraction in self.posed_success_fractions
        }
        self.posed_success_total = 0
        self.contact_counts = {"tp": 0, "fp": 0, "fn": 0}

    @staticmethod
    def infer_distance(
        predicted_density: torch.Tensor,
        teacher_distance: torch.Tensor,
        bandwidths: torch.Tensor,
    ) -> torch.Tensor:


        probability = predicted_density.clamp(min=1.0e-8, max=1.0 - 1.0e-7)
        sigma = bandwidths[:, None, None, :]
        inferred_per_band = sigma * torch.sqrt((-2.0 * torch.log(probability)).clamp_min(0.0))
        informative_band = torch.argmin(
            torch.abs(teacher_distance.unsqueeze(-1) - sigma),
            dim=-1,
            keepdim=True,
        )
        return torch.gather(inferred_per_band, -1, informative_band).squeeze(-1)

    def update(
        self,
        predicted_density: torch.Tensor,
        teacher_distance: torch.Tensor,
        bandwidths: torch.Tensor,
        valid_mask: torch.Tensor,
        query_stratum: torch.Tensor,
        owner_role: torch.Tensor,
    ) -> None:


        inferred_distance = self.infer_distance(predicted_density, teacher_distance, bandwidths)
        distance_error = torch.abs(inferred_distance - teacher_distance)
        masks = {
            "all": valid_mask,
            "workspace": valid_mask & (query_stratum == 0),
            "shell": valid_mask & (query_stratum == 1),
            "adjacent": valid_mask & (query_stratum == 2),
            "palm": valid_mask & (owner_role[:, :, None] == 0),
            "joint": valid_mask & (owner_role[:, :, None] == 1),
            "tip": valid_mask & (owner_role[:, :, None] == 2),
        }
        for group, mask in masks.items():
            values = distance_error[mask].detach().float().cpu()
            stats = self.group_statistics[group]
            stats["count"] += int(values.numel())
            stats["absolute_sum_m"] += float(values.double().sum())
            stats["square_sum_m2"] += float(values.double().square().sum())
            for tolerance_index, tolerance in enumerate(self.tolerance_m):
                stats["tolerance_counts"][tolerance_index] += int((values <= tolerance).sum())
            stats["histogram"] += torch.histc(values, bins=500, min=0.0, max=0.05).double()
            stats["histogram_overflow"] += int((values > 0.05).sum())


        critical = valid_mask & (query_stratum != 0)
        critical_count = critical.flatten(start_dim=1).sum(dim=1)
        valid_rows = critical_count > 0
        self.posed_success_total += int(valid_rows.sum())
        for tolerance_index, tolerance in enumerate(self.tolerance_m):
            hit = ((distance_error <= tolerance) & critical).flatten(start_dim=1).sum(dim=1)
            ratio = hit.to(torch.float64) / critical_count.clamp_min(1).to(torch.float64)
            for fraction in self.posed_success_fractions:
                self.posed_success_count[fraction][tolerance_index] += int(
                    ((ratio >= fraction) & valid_rows).sum()
                )


        teacher_contact = teacher_distance <= 0.004
        predicted_contact = inferred_distance <= 0.004
        self.contact_counts["tp"] += int((valid_mask & teacher_contact & predicted_contact).sum())
        self.contact_counts["fp"] += int((valid_mask & ~teacher_contact & predicted_contact).sum())
        self.contact_counts["fn"] += int((valid_mask & teacher_contact & ~predicted_contact).sum())

    def finalize(self) -> dict[str, object]:


        report: dict[str, object] = {}
        for group, stats in self.group_statistics.items():
            count = max(1, int(stats["count"]))
            cumulative = torch.cumsum(stats["histogram"], dim=0)
            if int(stats["histogram_overflow"]) > 0.05 * count:
                p95_m: float | str = ">0.05"
            else:
                rank = 0.95 * count
                index = int(torch.searchsorted(cumulative, torch.tensor(rank, dtype=torch.float64)))
                p95_m = min(0.05, (index + 1) * 0.0001)
            report[group] = {
                "mae_m": float(stats["absolute_sum_m"]) / count,
                "rmse_m": (float(stats["square_sum_m2"]) / count) ** 0.5,
                "p95_m": p95_m,
                "query_success": {
                    f"{int(tolerance * 1000)}mm": int(stats["tolerance_counts"][index]) / count
                    for index, tolerance in enumerate(self.tolerance_m)
                },
                "count": int(stats["count"]),
            }
        report["posed_geometry_success"] = {
            f"PGS@{int(tolerance * 1000)}mm,{int(fraction * 100)}%": (
                self.posed_success_count[fraction][index] / max(1, self.posed_success_total)
            )
            for fraction in self.posed_success_fractions
            for index, tolerance in enumerate(self.tolerance_m)
        }
        report["headline"] = {
            "metric": "PGS@4mm,80%",
            "value": self.posed_success_count[0.80][2] / max(1, self.posed_success_total),
            "physical_basis": "4 mm equals the narrowest canonical Gaussian bandwidth",
        }
        precision = self.contact_counts["tp"] / max(1, self.contact_counts["tp"] + self.contact_counts["fp"])
        recall = self.contact_counts["tp"] / max(1, self.contact_counts["tp"] + self.contact_counts["fn"])
        report["contact_4mm"] = {
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(1.0e-30, precision + recall),
            **self.contact_counts,
        }
        return report


__all__ = ["DensityPhysicalMetricAccumulator"]
