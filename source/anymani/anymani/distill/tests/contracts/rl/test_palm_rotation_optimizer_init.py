r"""名称驱动Adam warm-start的CPU合同测试。

这些测试只构造小型``torch.optim.Adam``，不启动Isaac Sim；它们证伪整数参数ID错配、未授权新增参数、
状态孤儿和Adam语义漂移等会污染真实PPO续接的边界条件。
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from anymani.distill.rl.runtime.palm_rotation_optimizer_init import (
    load_named_optimizer_state,
    load_optimizer_parameter_names,
    optimizer_parameter_names,
)


def _build_source() -> tuple[torch.optim.Adam, dict[str, torch.nn.Parameter], dict[str, Any]]:
    r"""构造一次真实CPU Adam更新后的source与其名称组。"""

    alpha = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    beta = torch.nn.Parameter(torch.tensor([0.5, 1.5, -0.25]))
    source_parameters = {"alpha": alpha, "beta": beta}
    optimizer = torch.optim.Adam(
        [{"params": [alpha], "name": "base", "lr": 0.1}, {"params": [beta], "name": "context", "lr": 0.2}],
        eps=1.0e-8,
    )
    optimizer.zero_grad()
    (alpha.square().sum() + 2.0 * beta.square().sum()).backward()
    optimizer.step()
    return optimizer, source_parameters, optimizer.state_dict()


def test_optimizer_parameter_names_exports_relative_order_and_group_names() -> None:
    r"""名称表保留optimizer真实组序与组内顺序，并剥离Actor路径前缀。"""

    optimizer, parameters, _state = _build_source()
    assert optimizer_parameter_names(
        optimizer,
        {"actor.alpha": parameters["alpha"], "actor.beta": parameters["beta"]},
    ) == [
        {"name": "base", "parameters": ["alpha"]},
        {"name": "context", "parameters": ["beta"]},
    ]


def test_named_optimizer_state_maps_reordered_names_and_preserves_target_lr() -> None:
    r"""source组/参数重排仍按名称恢复矩；target学习率保持新配置且新增参数从零开始。"""

    source_optimizer, source_parameters, source_state = _build_source()
    source_groups = optimizer_parameter_names(source_optimizer, source_parameters)
    target_alpha = torch.nn.Parameter(torch.tensor([4.0, 5.0]))
    target_beta = torch.nn.Parameter(torch.tensor([6.0, 7.0, 8.0]))
    target_gamma = torch.nn.Parameter(torch.tensor([9.0]))
    target_optimizer = torch.optim.Adam(
        [
            {"params": [target_beta], "name": "context", "lr": 0.3},
            {"params": [target_gamma, target_alpha], "name": "base", "lr": 0.4},
        ],
        eps=1.0e-8,
    )

    report = load_named_optimizer_state(
        target_optimizer,
        source_state,
        source_groups,
        {"beta": target_beta, "gamma": target_gamma, "alpha": target_alpha},
        allowed_new_names=("gamma",),
    )
    assert report["loaded_state_count"] == 2
    assert report["initialized_state_count"] == 1
    assert report["initialized_parameter_names"] == ["gamma"]
    assert [group["lr"] for group in target_optimizer.param_groups] == [0.3, 0.4]
    assert target_optimizer.state[target_alpha]["step"].item() == 1.0
    assert target_optimizer.state[target_beta]["step"].item() == 1.0
    assert target_optimizer.state[target_gamma]["step"].item() == 0.0
    assert torch.count_nonzero(target_optimizer.state[target_gamma]["exp_avg"]) == 0
    assert torch.count_nonzero(target_optimizer.state[target_gamma]["exp_avg_sq"]) == 0
    torch.testing.assert_close(
        target_optimizer.state[target_alpha]["exp_avg"], source_optimizer.state[source_parameters["alpha"]]["exp_avg"]
    )
    torch.testing.assert_close(
        target_optimizer.state[target_beta]["exp_avg"], source_optimizer.state[source_parameters["beta"]]["exp_avg"]
    )

    target_optimizer.zero_grad()
    (target_alpha.square().sum() + target_beta.square().sum() + target_gamma.square().sum()).backward()
    target_optimizer.step()
    assert target_optimizer.state[target_alpha]["step"].item() == 2.0
    assert target_optimizer.state[target_beta]["step"].item() == 2.0
    assert target_optimizer.state[target_gamma]["step"].item() == 1.0


def test_named_optimizer_state_rejects_same_shape_different_name() -> None:
    r"""相同shape不能替代名称身份，避免整数ID或形状猜测导致静默错配。"""

    _source_optimizer, _source_parameters, source_state = _build_source()
    source_groups = [
        {"name": "base", "parameters": ["source_name"]},
        {"name": "context", "parameters": ["beta"]},
    ]
    alpha = torch.nn.Parameter(torch.zeros(2))
    beta = torch.nn.Parameter(torch.zeros(3))
    target = torch.optim.Adam(
        [{"params": [alpha], "name": "base"}, {"params": [beta], "name": "context"}], eps=1.0e-8
    )
    with pytest.raises(ValueError, match="unknown in target"):
        load_named_optimizer_state(target, source_state, source_groups, {"alpha": alpha, "beta": beta})


def test_named_optimizer_state_rejects_unknown_new_missing_duplicate_or_orphan() -> None:
    r"""未授权新增、源状态缺失、重复名称和孤儿state都在target写入前失败。"""

    source_optimizer, source_parameters, source_state = _build_source()
    source_groups = optimizer_parameter_names(source_optimizer, source_parameters)
    alpha = torch.nn.Parameter(torch.zeros(2))
    beta = torch.nn.Parameter(torch.zeros(3))
    gamma = torch.nn.Parameter(torch.zeros(1))
    target = torch.optim.Adam(
        [{"params": [alpha], "name": "base"}, {"params": [beta, gamma], "name": "context"}], eps=1.0e-8
    )
    with pytest.raises(ValueError, match="non-allowlisted new"):
        load_named_optimizer_state(target, source_state, source_groups, {"alpha": alpha, "beta": beta, "gamma": gamma})

    duplicate_groups = copy.deepcopy(source_groups)
    duplicate_groups[1]["parameters"] = ["alpha"]
    with pytest.raises(ValueError, match="duplicate optimizer parameter name"):
        load_named_optimizer_state(target, source_state, duplicate_groups, {"alpha": alpha, "beta": beta})

    missing_state = copy.deepcopy(source_state)
    missing_state["state"].pop(1)
    with pytest.raises(ValueError, match="missing parameter ids"):
        load_named_optimizer_state(target, missing_state, source_groups, {"alpha": alpha, "beta": beta, "gamma": gamma}, allowed_new_names=("gamma",))

    orphan_state = copy.deepcopy(source_state)
    orphan_state["state"][999] = copy.deepcopy(next(iter(orphan_state["state"].values())))
    with pytest.raises(ValueError, match="orphan"):
        load_named_optimizer_state(target, orphan_state, source_groups, {"alpha": alpha, "beta": beta, "gamma": gamma}, allowed_new_names=("gamma",))


def test_named_optimizer_state_rejects_adam_semantic_mismatch() -> None:
    r"""betas等Adam语义必须相同；只有source学习率可以与target配置不同。"""

    source_optimizer, source_parameters, source_state = _build_source()
    source_groups = optimizer_parameter_names(source_optimizer, source_parameters)
    alpha = torch.nn.Parameter(torch.zeros(2))
    beta = torch.nn.Parameter(torch.zeros(3))
    target = torch.optim.Adam(
        [{"params": [alpha], "name": "base"}, {"params": [beta], "name": "context"}],
        betas=(0.8, 0.999),
        eps=1.0e-8,
    )
    with pytest.raises(ValueError, match="semantic mismatch"):
        load_named_optimizer_state(target, source_state, source_groups, {"alpha": alpha, "beta": beta})


def test_optimizer_name_ledger_enforces_both_hashes_and_schema(tmp_path: Path) -> None:
    r"""名称ledger同时绑定source checkpoint字节和ledger自身字节，未知schema字段立即拒绝。"""

    checkpoint_sha = "a" * 64
    ledger = {
        "schema_version": "1.0.0",
        "checkpoint_sha256": checkpoint_sha,
        "optimizers": {
            "optimizer": [{"name": "base", "parameters": ["alpha"]}],
            "anymani_critic_optimizer": [{"name": None, "parameters": ["value"]}],
        },
    }
    path = tmp_path / "optimizer-names.json"
    path.write_text(json.dumps(ledger, separators=(",", ":")) + "\n", encoding="utf-8")
    ledger_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_optimizer_parameter_names(
        path,
        expected_checkpoint_sha256=checkpoint_sha,
        expected_ledger_sha256=ledger_sha,
    )
    assert loaded == ledger["optimizers"]
    with pytest.raises(ValueError, match="ledger SHA-256 mismatch"):
        load_optimizer_parameter_names(path, expected_checkpoint_sha256=checkpoint_sha, expected_ledger_sha256="b" * 64)
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        load_optimizer_parameter_names(path, expected_checkpoint_sha256="c" * 64, expected_ledger_sha256=ledger_sha)

    invalid = dict(ledger)
    invalid["extra"] = True
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
    invalid_sha = hashlib.sha256(invalid_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="exactly schema_version"):
        load_optimizer_parameter_names(
            invalid_path,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_ledger_sha256=invalid_sha,
        )
