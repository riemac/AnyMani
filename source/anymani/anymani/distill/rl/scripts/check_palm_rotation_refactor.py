r"""对照已封存Git源码，验证掌旋PPO重构的数值与状态语义。

本脚本不启动Isaac。它在相同合成具名输入和随机流上比较四种动作读出、两种History30编码，以及可选
真实checkpoint的前向、采样、概率、梯度和独立Adam单步。合成输入只证伪软件语义漂移，不证明物理能力。
旧代码从本地已审查的版本读取；比较结果记录两个源码摘要与checkpoint摘要，不改写任何旧实验产物。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import torch
from anymani.assets.bank.path_utils import resolve_anymani_root
from anymani.distill.rl.rl_games_backend import prefer_local_rl_games


def _model(module: ModuleType, arm: str, history: str) -> torch.nn.Module:
    r"""由指定代码版本构造同一具名ABI的网络与value normalizer。"""

    from anymani.distill.rl.runtime.palm_rotation_vecenv import (
        PALM_ROTATION_BOOL_SHAPES,
        PALM_ROTATION_FLOAT_SHAPES,
        PALM_ROTATION_INT16_SHAPES,
    )

    builder = module.PalmRotationRlGamesBuilder()  # 两版本使用各自真实builder类
    builder.load(
        {
            "palm_rotation": {"arm": arm, "history_encoder": history, "compile_mode": None},
            "anymani_identity": {"identity_digest": "refactor-contract-only"},
        }
    )
    return module.PalmRotationMaskedContinuousModel(builder).build(
        {
            "actions_num": 16,
            "input_shape": {**PALM_ROTATION_FLOAT_SHAPES, **PALM_ROTATION_BOOL_SHAPES, **PALM_ROTATION_INT16_SHAPES},
            "value_size": 1,
            "normalize_input": False,
            "normalize_value": True,
        }
    )


def _observations() -> dict[str, torch.Tensor]:
    r"""构造两种有效关节集合，覆盖ghost masking和完整privileged输入。"""

    from anymani.distill.rl.runtime.palm_rotation_vecenv import PALM_ROTATION_FLOAT_SHAPES

    generator = torch.Generator().manual_seed(20260905)  # 独立fixture随机流，不干扰动作采样
    observation = {
        name: torch.randn((2, *shape), generator=generator) for name, shape in PALM_ROTATION_FLOAT_SHAPES.items()
    }
    joint = torch.tensor(
        ([True] * 12 + [False] * 4, [True, False, True, True] * 2 + [True, False, False, True] + [False] * 4)
    )
    tip = joint.reshape(2, 4, 4).any(dim=1)  # depth-major关节轴派生真实TIP有效集合
    observation.update(
        {
            "jnt_valid": joint,
            "tip_valid": tip,
            "owner_valid": torch.cat((torch.ones(2, 1, dtype=torch.bool), joint, tip), dim=-1),
            "actor_jnt_limits": torch.stack((-torch.ones(2, 16), torch.ones(2, 16)), dim=-1),
            "actor_owner_contact": torch.randint(0, 2, (2, 21, 1), generator=generator).float(),
            "prototype_index": torch.arange(2, dtype=torch.int16).unsqueeze(-1),
            **{
                name: torch.zeros(2, 21, 21, dtype=torch.int16)
                for name in ("shortest_path", "parent_direction", "child_direction")
            },
        }
    )
    return observation


def _equal_state(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    r"""逐键、逐值比较；不允许宽松容差掩盖纯迁移引入的变化。"""

    assert list(left) == list(right), "state_dict key order changed"
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0.0, atol=0.0, msg=key)


def _compare(reference: ModuleType, current: ModuleType, arm: str, history: str, state: Any = None) -> dict[str, Any]:
    r"""比较完整网络概率路径、梯度和独立actor/critic Adam更新。"""

    torch.manual_seed(42)
    old = _model(reference, arm, history)  # 固定初始化RNG，检验构造顺序也保持一致
    torch.manual_seed(42)
    new = _model(current, arm, history)
    _equal_state(old.state_dict(), new.state_dict())
    if state is not None:
        old.load_state_dict(state, strict=True)
        new.load_state_dict(state, strict=True)  # 真实checkpoint必须无缺失/多余键
    observation = _observations()
    outputs = []
    for model in (old, new):
        model.eval()
        torch.manual_seed(701)
        with torch.no_grad():
            outputs.append(model({"is_train": False, "obs": dict(observation)}))
    for key in outputs[0]:
        if isinstance(outputs[0][key], torch.Tensor):
            assert torch.isfinite(outputs[0][key]).all(), key
            torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=0.0, rtol=0.0, msg=key)

    group_names = []
    for model in (old, new):
        model.train()
        torch.manual_seed(702)  # Monte Carlo entropy也使用同一采样流
        result = model({"is_train": True, "prev_actions": outputs[0]["actions"], "obs": dict(observation)})
        loss = result["prev_neglogp"].mean() + 0.5 * result["values"].square().mean() - 0.002 * result["entropy"].mean()
        loss.backward()  # 联合检验actor/critic参数坐标；它不是新的生产PPO目标
        network = cast(Any, model.a2c_network)  # 两份源码中的适配器类不是同一个Python类型
        base, context = network.actor_parameter_groups()
        names = {id(parameter): name for name, parameter in network.package.actor.named_parameters()}
        group_names.append([[names[id(parameter)] for parameter in group] for group in (base, context)])
    assert group_names[0] == group_names[1], "actor optimizer partition changed"
    for (name, left), (right_name, right) in zip(old.named_parameters(), new.named_parameters(), strict=True):
        assert name == right_name and (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=0.0, rtol=0.0, msg=name)
    for model in (old, new):
        network = cast(Any, model.a2c_network)  # 跨版本比较相同参数接口，不依赖类身份
        base, context = network.actor_parameter_groups()
        actor_optimizer = torch.optim.Adam([{"params": base, "lr": 3e-4}, {"params": context, "lr": 1e-4}], eps=1e-8)
        critic_optimizer = torch.optim.Adam(network.package.critic.parameters(), lr=5e-4, eps=1e-8)
        torch.nn.utils.clip_grad_norm_(network.package.actor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(network.package.critic.parameters(), 1.0)
        actor_optimizer.step()
        critic_optimizer.step()
        network.package.actor.project_exploration_parameters()
    _equal_state(old.state_dict(), new.state_dict())
    return {"arm": arm, "history": history, "checkpoint_loaded": state is not None, "exact_forward_gradient_adam": True}


def main() -> None:
    r"""加载本地版本对照并发布独立JSON；默认v0.8.3是工程前置封存锚。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_revision", default="v0.8.3")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new evidence file")
    root = resolve_anymani_root()
    path = "source/anymani/anymani/distill/rl/palm_rotation_ppo.py"
    source = subprocess.check_output(["git", "show", f"{args.reference_revision}:{path}"], cwd=root).decode()
    reference = ModuleType("anymani.distill.rl._refactor_reference")
    reference.__package__ = "anymani.distill.rl"
    reference.__file__ = str(root / path)
    sys.modules[reference.__name__] = reference
    prefer_local_rl_games(strict=True)  # 两版本先固定同一个后端，再导入任何rl_games对象
    exec(compile(source, reference.__file__, "exec"), reference.__dict__)
    from anymani.distill.rl import palm_rotation_ppo as current
    from anymani.distill.rl.runtime.palm_rotation_identity import palm_rotation_implementation_files

    torch.set_num_threads(1)
    old_agent = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "PalmRotationPpoAgent"
    )
    current_source = (root / path).read_text()
    new_agent = next(
        node
        for node in ast.parse(current_source).body
        if isinstance(node, ast.ClassDef) and node.name == "PalmRotationPpoAgent"
    )
    old_methods = {node.name: node for node in old_agent.body if isinstance(node, ast.FunctionDef)}
    new_methods = {node.name: node for node in new_agent.body if isinstance(node, ast.FunctionDef)}
    unchanged = (
        "__init__",
        "init_tensors",
        "update_lr",
        "prepare_dataset",
        "calc_gradients",
        "_reset_optimization_metrics",
        "get_full_state_weights",
        "set_full_state_weights",
    )
    for name in unchanged:
        assert ast.dump(old_methods[name]) == ast.dump(new_methods[name]), (
            f"training mathematics/state flow changed: {name}"
        )
    labels = torch.arange(16).repeat_interleave(16)
    advantages = torch.linspace(-2.0, 3.0, labels.numel())
    old_batch = reference.normalize_advantages_per_asset(advantages, labels, asset_count=16)
    new_batch = current.normalize_advantages_per_asset(advantages, labels, asset_count=16)
    for left, right in zip(old_batch, new_batch, strict=True):
        torch.testing.assert_close(left, right, atol=0.0, rtol=0.0)
    permutations = [
        module.stratified_asset_permutation(
            labels, asset_count=16, minibatch_count=4, generator=torch.Generator().manual_seed(703)
        )
        for module in (reference, current)
    ]
    assert torch.equal(*permutations), "stratified sample ordering changed"
    cases = [
        _compare(reference, current, arm, history)
        for arm in ("base", "residual", "direct", "direct_token")
        for history in ("tcn", "raw_stack")
    ]
    checkpoint_sha = None
    if args.checkpoint is not None:
        checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        identity = checkpoint["anymani_identity"]
        cases.append(
            _compare(
                reference,
                current,
                identity["policy"]["arm"],
                identity["training"]["history_encoder"],
                checkpoint["model"],
            )
        )
        assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == checkpoint_sha
    identity_path = "source/anymani/anymani/distill/rl/runtime/palm_rotation_identity.py"
    identity_source = subprocess.check_output(
        ["git", "show", f"{args.reference_revision}:{identity_path}"], cwd=root
    ).decode()
    reference_paths = next(
        ast.literal_eval(node.value)
        for node in ast.parse(identity_source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_IMPLEMENTATION_PATHS" for target in node.targets)
    )
    reference_files = {
        name: hashlib.sha256(
            subprocess.check_output(["git", "show", f"{args.reference_revision}:{name}"], cwd=root)
        ).hexdigest()
        for name in reference_paths
    }
    current_files = palm_rotation_implementation_files()
    relocated = {path, identity_path, "source/anymani/anymani/distill/rl/train_palm_rotation_mvp.py"}
    for name, expected in reference_files.items():
        if name not in relocated:
            assert current_files[name] == expected, f"non-refactored implementation changed: {name}"
    result = {
        "artifact_type": "anymani.palm_rotation.refactor_equivalence",
        "schema_version": "1.0.0",
        "reference_revision": args.reference_revision,
        "reference_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "current_source_sha256": hashlib.sha256(current_source.encode()).hexdigest(),
        "reference_implementation_files": reference_files,
        "current_implementation_files": current_files,
        "checkpoint_sha256": checkpoint_sha,
        "unchanged_training_methods": unchanged,
        "cases": cases,
        "passed": True,
        "boundary": "CPU synthetic interface/gradient/update parity; not simulator or learning acceptance",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases), "passed": True}))


if __name__ == "__main__":
    main()
