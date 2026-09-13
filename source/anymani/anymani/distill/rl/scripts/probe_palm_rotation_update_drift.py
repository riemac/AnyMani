r"""用保存的完整采样批次测量两个冻结策略端点之间的分布与价值变化。

本探针消费更新前审计包的同一组状态、缓存N040、动作、旧概率和GAE目标，
比较其更新前参数与同更新终点checkpoint，或两个事先指定的跨更新checkpoint。
它不重放Adam、不采集物理状态；端点差来自实际保存参数。

设潜Normal中心为 m=atanh(mu)，逐状态活动关节数为 d，统计定义为：
    K = D_KL(pi_after || pi_before)/d = K_mean + K_scale，
    K_mean = mean_active[0.5*((m_after-m_before)/sigma_before)^2]，
    K_scale = mean_active[0.5*expm1(2u)-u]，u=log(sigma_after/sigma_before)。
KL沿共同可逆tanh变换保持不变。这里的参考固定为before端点，
与训练循环中逐小批次刷新的KL参考具有不同时间语义。

PPO概率比始终另对原采样概率计算；Critic只与该批已经形成的GAE目标比较。
跨策略的旧目标拟合、训练状态上的动作变化均是描述性证据，物理能力仍由固定评价给出。
确定性动作中心差除以24，只是同状态下请求target增量的差，未包含关节限位投影与动力学。
逐资产/家族标签仅用于下游统计，不作为Actor或Critic特征。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from anymani.distill.rl.rl_games_backend import prefer_local_rl_games

prefer_local_rl_games(strict=True)  # 只固定概率适配器依赖；不构造Runner、环境或仿真器。

from anymani.distill.models.palm_rotation_policy import (  # noqa: E402
    PalmRotationActorCritic,
    PalmRotationActorObservation,
    PalmRotationCriticObservation,
    PalmRotationGeometry,
)
from anymani.distill.rl.algorithms.policy_statistics import mean_preserving_squashed_kl  # noqa: E402
from anymani.distill.rl.runtime.palm_rotation_network import PalmRotationMaskedContinuousModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[6]  # 源码根，保持Research单向依赖。
PREFIX = "a2c_network.package."  # 保存模型的稳定参数命名空间。


def load(path: str | Path) -> dict[str, Any]:
    r"""按CPU只读mmap加载本机训练证据；NumPy RNG对象要求完整pickle读取。"""
    return torch.load(ROOT / path, map_location="cpu", weights_only=False, mmap=True)  # 不把完整批次搬入GPU。


def optimizer_step(optimizer: dict[str, Any]) -> int:
    r"""提取所有Adam参数一致的实际step计数，检查端点时间差。"""
    steps = {int(state["step"]) for state in optimizer["state"].values()}  # 所有参与更新参数的时钟。
    assert len(steps) == 1, steps  # 存在不同步参数时不能给出单一预算解释。
    return steps.pop()  # 不依据文件名推断optimizer步数。


def package_state(model: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""提取完全分参的Actor/Critic；PopArt统计单独保留在外层。"""
    return {key[len(PREFIX):]: value for key, value in model.items() if key.startswith(PREFIX)}


def verify_sources(identity: dict[str, Any]) -> None:
    r"""重放前要求原运行声明的每个实现文件逐字节相同。"""
    for name, expected in identity["implementation"]["files"].items():  # 新诊断文件不改原训练身份。
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


@torch.no_grad()
def predict(model: dict[str, torch.Tensor], packet: dict[str, Any], device: torch.device, chunk: int) -> dict:
    r"""在同一完整批次上重算Actor与Critic，分块只改变激活内存。

    前向采用原FP32/TF32与bound-forward编译设置，输出与归约转回CPU；N040使用逐状态保存的FP32缓存。
    返回有界mean、潜sigma、整手negative logp，以及归一化/物理value，均保持样本排列。
    """
    identity = packet["identity"]  # checkpoint结构来自原合同。
    cfg = identity["training"]  # H与History30分别保留。
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg["allow_tf32"])  # 还原原矩阵运算精度。
    torch.backends.cudnn.allow_tf32 = bool(cfg["allow_tf32"])  # TCN卷积使用相同设置。
    package = PalmRotationActorCritic(
        arm=identity["policy"]["arm"],  # 单一共享Actor的实际读出模式。
        initial_log_std=cfg["initial_log_std"], max_log_std=cfg["max_log_std"],
        base_action_limit=cfg["base_action_limit"], history_encoder=cfg["history_encoder"],
        sigma_mode=cfg.get("sigma_mode", "global"),
    )  # 构造后必须严格载入保存参数，初始sigma不代表实际探索量。
    state = package_state(model)  # Actor/Critic全部参数，保持原Tensor dtype。
    package.load_state_dict(state, strict=True)  # 缺失/多余参数立即停止。
    package.to(device).eval()  # dropout为0；此处不更新任何学习统计。
    actor_forward, critic_forward = package.actor.forward, package.critic.forward  # 与生产相同的bound函数边界。
    compile_mode = cfg["torch_compile"]  # 原运行明确记录None/default/reduce-overhead。
    if compile_mode is not None:  # 编译算子融合会影响浮点输出，忠实重放保持同一执行方式。
        actor_forward = torch.compile(actor_forward, mode=compile_mode)
        critic_forward = torch.compile(critic_forward, mode=compile_mode)
    data = packet["dataset"]  # dataset已经分层排列，直接使用其样本对应。
    output: dict[str, list[torch.Tensor]] = {key: [] for key in ("mean", "sigma", "neglogp", "value_norm")}
    minimum_free = torch.cuda.mem_get_info(device)[0]  # driver余量包括外部占用。
    for start in range(0, data["actions"].shape[0], chunk):  # 完整B，最后一块允许较短。
        obs = {k: v[start:start + chunk].to(device) for k, v in data["obs"].items()}  # 仅当前激活块驻留GPU。
        geometry = PalmRotationGeometry(
            tokens=obs["geometry_tokens"].float(), owner_valid=obs["owner_valid"].bool(),
            shortest_path=obs["shortest_path"].long(), parent_direction=obs["parent_direction"].long(),
            child_direction=obs["child_direction"].long(),
        )  # graph整型转换与生产适配器一致。
        actor_obs = PalmRotationActorObservation(
            jnt_current=obs["actor_jnt_current"].float(), jnt_history=obs["actor_jnt_history"].float(),
            jnt_limits=obs["actor_jnt_limits"].float(), owner_contact=obs["actor_owner_contact"].float(),
            jnt_valid=obs["jnt_valid"].bool(), tip_valid=obs["tip_valid"].bool(), owner_valid=obs["owner_valid"].bool(),
        )  # 不向Actor提供prototype_index、object、task或privileged contact。
        critic_obs = PalmRotationCriticObservation(
            jnt_state=obs["critic_jnt_state"].float(), owner_contact=obs["critic_owner_contact"].float(),
            obj=obs["critic_obj"].float(), task=obs["critic_task"].float(),
            reward_release=obs["critic_reward_release"].float(), jnt_valid=obs["jnt_valid"].bool(),
            tip_valid=obs["tip_valid"].bool(), owner_valid=obs["owner_valid"].bool(),
        )  # Critic保持独立参数和原允许特权信息。
        prediction = actor_forward(actor_obs, geometry)  # 有界确定性动作中心[B_chunk,16]。
        logstd = prediction.log_std.expand_as(prediction.mean)  # 标准差属于tanh之前的潜空间。
        sigma = logstd.exp()  # [B_chunk,16]，保持FP32概率实现。
        actions = data["actions"][start:start + chunk].to(device)  # 使用已经采集的动作，不再采样。
        joint_neglogp = PalmRotationMaskedContinuousModel.Network._squashed_per_joint_neglogp(
            actions, prediction.mean, sigma, logstd
        )  # 同生产atanh边界及Jacobian项。
        output["mean"].append(prediction.mean.cpu())  # 原样保留逐关节诊断数据。
        output["sigma"].append(sigma.cpu())  # 不将ghost sigma计入后续分布归约。
        output["neglogp"].append((joint_neglogp * actor_obs.jnt_valid).sum(-1).cpu())  # 整手联合概率。
        output["value_norm"].append(critic_forward(critic_obs, geometry).cpu())  # [B_chunk]，归一化value。
        minimum_free = min(minimum_free, torch.cuda.mem_get_info(device)[0])  # 每个块检查资源。
        assert minimum_free >= 2 * 1024**3, minimum_free  # 本机至少保留2GiB驱动显存。
    result: dict[str, Any] = {key: torch.cat(parts) for key, parts in output.items()}  # 沿原样本轴串接。
    mean = model["value_mean_std.running_mean"].float()  # 当前模型对应的PopArt坐标。
    scale = (model["value_mean_std.running_var"] + 1e-5).sqrt().float()  # 原生产epsilon。
    result["value_phys"] = result["value_norm"] * scale + mean  # 不裁剪反归一化预测。
    assert all(bool(torch.isfinite(value).all()) for value in result.values())  # 非有限端点不参与比较。
    for name, value in package.state_dict().items():  # 模型/统计buffer在本次只读前向中逐值保持。
        assert torch.equal(value.cpu(), state[name]), name
    result["minimum_driver_free_bytes"] = minimum_free  # 模型生命周期外仍可审计资源。
    return result  # package离开作用域，数据在CPU；没有optimizer对象被构造。


def distribution_shift(before: dict, after: dict, active: torch.Tensor) -> dict[str, torch.Tensor]:
    r"""计算固定端点的KL、均值/尺度分解和同状态动作差；统计归约使用FP64。"""
    old_mean, new_mean = before["mean"].double(), after["mean"].double()  # 有界动作中心。
    old_sigma, new_sigma = before["sigma"].double(), after["sigma"].double()  # 潜空间尺度。
    old_latent = torch.atanh(old_mean.clamp(-1 + 1e-6, 1 - 1e-6))  # 生产mean-preserving坐标。
    new_latent = torch.atanh(new_mean.clamp(-1 + 1e-6, 1 - 1e-6))  # tanh公共变换的KL不变性。
    count = active.sum(-1)  # 各状态真实活动自由度，ghost不进入分母。
    delta = new_latent - old_latent  # 未按探索尺度归一的潜中心差。
    scaled = delta / old_sigma  # 相对before探索支持的中心变化。
    u = new_sigma.log() - old_sigma.log()  # 对数标准差比，无量纲。
    mean_kl = (0.5 * scaled.square() * active).sum(-1) / count  # 均值变化贡献。
    scale_kl = ((0.5 * torch.expm1(2 * u) - u) * active).sum(-1) / count  # 尺度变化贡献。
    total_kl = mean_preserving_squashed_kl(new_mean, new_sigma, old_mean, old_sigma, active)
    assert torch.allclose(total_kl, mean_kl + scale_kl, atol=1e-12, rtol=1e-9)  # 对照生产KL实现。
    return {
        "kl_per_active_dof": total_kl, "kl_joint": total_kl * count,  # 两种单位独立保存。
        "kl_mean_term": mean_kl, "kl_scale_term": scale_kl,  # 不把sigma变化与中心变化混称。
        "action_delta_sq": (new_mean - old_mean).square(),  # 同状态下动作中心差的二阶量。
        "latent_delta_sq": delta.square(), "scaled_latent_delta_sq": scaled.square(),
    }  # 每状态KL与每坐标二阶量由下游按各自分母归约。


def value_fit(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    r"""在同一冻结GAE目标上分别报告带偏差MSE分数和去偏差方差分数。"""
    y, v = target.double(), prediction.double()  # FP64仅用于归约，不改变网络前向。
    residual = v - y  # 预测减目标，正偏差表示高估该批目标。
    variance = float(y.var(unbiased=False))  # 明确总体方差分母n。
    mse = float(residual.square().mean())  # 保留系统性偏差。
    return {
        "target_mean": float(y.mean()), "target_variance": variance,
        "bias": float(residual.mean()), "mse": mse,
        "mse_score": 1 - mse / variance if variance > 1e-12 else None,  # 近常数目标不给虚假分数。
        "centered_variance_score": 1 - float(residual.var(unbiased=False)) / variance if variance > 1e-12 else None,
    }  # 两种分数不能按相同名称互换。


def summarize(before: dict, after: dict, data: dict, shift: dict, selected: torch.Tensor, epsilon: float) -> dict:
    r"""归约一个预先指定的样本总体；逐关节RMS使用真实活动坐标分母。"""
    active = data["obs"]["jnt_valid"][selected].bool()  # [B_selected,16]。
    sample_count = int(selected.sum())  # 状态等权，家族内每资产样本数相同。
    row: dict[str, Any] = {"samples": sample_count, "active_coordinate_samples": int(active.sum())}
    for key in ("kl_per_active_dof", "kl_joint", "kl_mean_term", "kl_scale_term"):
        values = shift[key][selected]  # 每状态统计量，避免把ghost计入自由度。
        row[key] = {"mean": float(values.mean()), "q50": float(torch.quantile(values, 0.5)),
                    "q90": float(torch.quantile(values, 0.9)), "max": float(values.max())}
    for source, name in (("action_delta_sq", "action_center_change_rms"),
                         ("latent_delta_sq", "latent_mean_change_rms"),
                         ("scaled_latent_delta_sq", "mean_shift_over_before_sigma_rms")):
        row[name] = float(((shift[source][selected] * active).sum() / active.sum()).sqrt())  # 坐标加权RMS。
    row["requested_target_increment_change_rms_rad"] = row["action_center_change_rms"] / 24  # 未含限位投影。
    advantage = data["advantages"][selected].double()  # 当前配置实际用于Actor的冻结优势。
    target = data["raw_returns"].reshape(-1)[selected]  # 原GAE目标，非独立Monte Carlo真值。
    for name, prediction in (("before", before), ("after", after)):
        log_ratio = data["old_logp_actions"][selected].double() - prediction["neglogp"][selected].double()
        ratio = log_ratio.exp()  # 参考始终是packet采样策略，不跟随端点变化。
        assert bool(torch.isfinite(ratio).all()), name  # 不静默裁剪概率比。
        objective = torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)
        row[name] = {
            "sigma_active_mean": float(prediction["sigma"][selected][active].double().mean()),
            "rollout_ratio_mean": float(ratio.mean()),  # 有限样本量，不能预设精确为1。
            "rollout_clip_fraction": float(((ratio - 1).abs() > epsilon).double().mean()),
            "rollout_clipped_surrogate": float(objective.mean()),  # 正号表示要最大化的surrogate。
            "value_fit_to_frozen_gae_target": value_fit(prediction["value_phys"][selected], target),
        }  # surrogate未包含entropy/bounds；也不等同物理净圈收益。
    row["clipped_surrogate_change"] = row["after"]["rollout_clipped_surrogate"] - row["before"]["rollout_clipped_surrogate"]
    return row  # 全体/家族/资产始终使用相同公式，分母各自明确。


def compare(case: dict, output_dir: Path, device: torch.device, chunk: int) -> dict:
    r"""对一个指定端点对执行重放门、实际步数核对及固定状态分布比较。"""
    started = time.perf_counter()  # 与原训练时长分开。
    packet = load(case["packet"])  # 完整原始采样批次。
    assert packet["capture_phase"] == "post-dataset-prepare-before-optimizer"
    identity, data = packet["identity"], packet["dataset"]  # 输入与方法身份绑定。
    verify_sources(identity)  # 当前生产实现必须能忠实解释保存参数。
    before_checkpoint = load(case["before_checkpoint"]) if case.get("before_checkpoint") else None
    after_checkpoint = load(case["after_checkpoint"])  # 实际运行保存的参数，未合成影子更新。
    for checkpoint in (before_checkpoint, after_checkpoint):
        if checkpoint is not None:  # 同方法跨时点，不放宽身份比较。
            assert checkpoint["anymani_identity"]["identity_digest"] == identity["identity_digest"]
    before_model = before_checkpoint["model"] if before_checkpoint else packet["model"]  # PopArt快照随模型走。
    before_optim = before_checkpoint["optimizer"] if before_checkpoint else packet["actor_optimizer"]
    steps = [optimizer_step(before_optim), optimizer_step(after_checkpoint["optimizer"])]  # 真实端点计数。
    assert steps[1] - steps[0] == case["expected_optimizer_steps"], steps
    if before_checkpoint is None:  # 同一个PPO更新的完整前后边界。
        assert after_checkpoint["epoch"] == packet["update"]  # 不把下一次更新或最终点误配给本包。
        for key in ("running_mean", "running_var", "count"):
            assert torch.equal(packet["model"]["value_mean_std." + key], after_checkpoint["model"]["value_mean_std." + key])
    replay = predict(packet["model"], packet, device, chunk)  # 每个批次先重建其真实采样策略。
    replay_errors = {
        "mean_max_abs": float((replay["mean"] - data["rollout_mu"]).abs().max()),
        "neglogp_max_abs": float((replay["neglogp"] - data["old_logp_actions"]).abs().max()),
        "value_normalized_max_abs": float((replay["value_norm"] - data["old_values"].reshape(-1)).abs().max()),
        "value_physical_max_abs": float((replay["value_phys"] - data["raw_values"].reshape(-1)).abs().max()),
    }  # 数值重放误差与模型端点变化独立报告。
    (output_dir / f"{case['name']}-replay.json").write_text(json.dumps(replay_errors, indent=2) + "\n")
    assert replay_errors["mean_max_abs"] < 5e-4 and replay_errors["neglogp_max_abs"] < 0.02, replay_errors
    assert replay_errors["value_normalized_max_abs"] < 0.005, replay_errors  # 预设为value clip半宽0.2的2.5%。
    before = predict(before_model, packet, device, chunk) if before_checkpoint else replay  # 同一状态总体。
    after = predict(after_checkpoint["model"], packet, device, chunk)  # 不生成新采样或修改原checkpoint。
    active = data["obs"]["jnt_valid"].bool()  # ghost坐标单独排除。
    shift = distribution_shift(before, after, active)  # 固定before端点参考。
    labels = data["obs"]["prototype_index"].reshape(-1).long()  # 仅做统计与cohort连接。
    manifest = ROOT / identity["manifest"]["path"]  # 原冻结成员顺序。
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == identity["manifest"]["sha256"]
    members = json.loads(manifest.read_text())["members"]  # 本入口消费canonical JSON内容的lock。
    assert identity["manifest"]["selected_rows"] == list(range(len(members)))  # 明确完整成员范围。
    counts = torch.bincount(labels, minlength=len(members))  # 每个资产的实际样本数。
    assert bool((counts == counts[0]).all()) and int(counts[0]) > 0  # 不静默遗漏某个资产。
    epsilon = float(packet["loss_configuration"]["e_clip"])  # 实際PPO裁剪半宽。
    whole = summarize(before, after, data, shift, torch.ones_like(labels, dtype=torch.bool), epsilon)
    assets, families = [], {}  # 保留全部资产，焦点集合在研究记录中事后解释。
    for index, member in enumerate(members):
        assert member["cohort_index"] == index  # 固定成员身份与观察标签逐位对应。
        row = summarize(before, after, data, shift, labels == index, epsilon)  # 每资产相同120样本。
        row.update(asset_index=index, member=f"{member['source_alias']}#{member['source_row']}",
                   group=member["provenance"]["group_name"])  # ID仅写诊断结果。
        assets.append(row)  # 不按策略成绩改变成员集合。
    for group in sorted({row["group"] for row in assets}):
        indices = torch.tensor([row["asset_index"] for row in assets if row["group"] == group])
        families[group] = summarize(before, after, data, shift, torch.isin(labels, indices), epsilon)
    parameter_rows = []  # 真实Actor参数差另存，不能用梯度范数替代。
    for key, old in before_model.items():
        if key.startswith(PREFIX + "actor."):
            delta = after_checkpoint["model"][key].double() - old.double()  # 固定参数坐标的实际差。
            parameter_rows.append({"name": key, "count": old.numel(), "delta_l2": float(delta.norm()),
                                   "before_l2": float(old.double().norm())})  # 保存每层尺度，避免仅给总范数。
    dense = {f"{side}_{key}": prediction[key].numpy() for side, prediction in (("before", before), ("after", after))
             for key in ("mean", "sigma", "neglogp", "value_norm", "value_phys")}  # 原样端点输出供复算。
    np.savez_compressed(output_dir / f"{case['name']}-outputs.npz", **dense,
                        asset_index=labels.numpy(), active=active.numpy())  # 不重复复制760MB输入包。
    result = {
        "comparison": case, "method_identity": identity["identity_digest"], "packet_update": packet["update"],
        "actor_optimizer_steps": steps, "replay_errors": replay_errors, "whole": whole,
        "families": families, "assets": assets, "actor_parameter_deltas": parameter_rows,
        "actor_delta_l2": sum(row["delta_l2"] ** 2 for row in parameter_rows) ** 0.5,
        "minimum_driver_free_bytes": min(p["minimum_driver_free_bytes"] for p in (replay, before, after)),
        "wall_seconds": time.perf_counter() - started, "models_unchanged_by_probe": True,
        "probe_optimizer_steps": 0, "new_environment_steps": 0,
    }  # 保存统计与执行见证，工程通过不代表物理能力通过。
    (output_dir / f"{case['name']}-summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result  # 外层仅打印压缩摘要。


def main() -> None:
    r"""执行明确列举的保存状态对照，结果写入全新目录。"""
    parser = argparse.ArgumentParser(description=__doc__)  # 不暴露仿真、训练或策略修改选项。
    parser.add_argument("--plan", type=Path, required=True)  # pairs列表冻结输入与预算预期。
    parser.add_argument("--output_dir", type=Path, required=True)  # 新证据，保留所有旧结果。
    parser.add_argument("--device", default="cuda:0")  # 主代理分配唯一GPU。
    parser.add_argument("--chunk_size", type=int, default=512)  # 只改变前向激活峰值。
    args = parser.parse_args()
    assert args.chunk_size > 0 and not args.output_dir.exists()  # 本次输出独立。
    device = torch.device(args.device)  # CPU归约与GPU前向边界显式。
    assert device.type == "cuda"  # 忠实于原GPU/TF32执行语义。
    torch.set_num_threads(1)  # 限制CPU读取和分位数计算争用。
    plan = json.loads(args.plan.read_text())  # 所有输入路径均由研究者/主代理在运行前指定。
    args.output_dir.mkdir(parents=True)  # 调用方已核对case父目录。
    provenance = {"plan": plan, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "prediction_backend": "recorded bound-forward compile mode", "chunk_size": args.chunk_size}
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")  # 失败也保留执行源码身份。
    for case in plan["comparisons"]:  # GPU对照串行，内存不同时容纳多个完整批次。
        result = compare(case, args.output_dir, device, args.chunk_size)
        print(json.dumps({"name": case["name"], "steps": result["actor_optimizer_steps"],
                          "kl_fixed_reference": result["whole"]["kl_per_active_dof"],
                          "actor_delta_l2": result["actor_delta_l2"],
                          "after_clip_fraction": result["whole"]["after"]["rollout_clip_fraction"]}), flush=True)
    metadata = {"status": "completed", "plan": plan, "chunk_size": args.chunk_size,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope": "Frozen endpoint differences on saved training states/GAE targets; zero new interaction or optimizer steps."}
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()  # 保存批次的只读审计入口。
