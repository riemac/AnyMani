r"""FlashSAC 原子检查点的 CPU 合同：身份闸门、故障保留与真实算法恢复。

使用已有 learner 合同的完整 History30/21-owner 输入、小型真实网络及其 Adam/RNG
状态。CompactReplay 和 StaticHandBank 同样来自生产实现；完整回放不以计数占位。
本测试没有环境交互或固定评价回放，精确续接只验证 CPU 算法状态和序列存储。
"""

from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from test_flash_sac_identity import identity_inputs as identity_inputs
from test_flash_sac_identity import identity_module as identity_module
from test_flash_sac_learner import _assert_tree_equal
from test_flash_sac_learner import api as api
from test_flash_sac_learner import batch as batch
from test_flash_sac_learner import restore_cpu_rng as restore_cpu_rng
from test_flash_sac_learner import small_config as small_config


@pytest.fixture(scope="module")
def checkpoint_module(identity_module: ModuleType) -> ModuleType:
    r"""复用已有 CPU 导入生命周期，加载生产 checkpoint I/O。"""
    return importlib.import_module("anymani.distill.rl.flash_sac.checkpoint")  # 没有环境或logger构造。


@pytest.fixture
def state_case(
    api: SimpleNamespace,  # 真实learner/network/replay API。
    small_config: Any,  # A=N=B=4、History30、n=3。
    batch: dict[str, Any],  # 完整具名观察与四种终点折扣。
    identity_module: ModuleType,  # 原生方法身份函数。
    identity_inputs: dict[str, Any],  # 同一四资产支持域与初态键。
) -> SimpleNamespace:
    r"""建立非空模型、静态表及 n=3 回放：全部状态都可由真实 owner 恢复。"""
    learner = api.learner.FlashSACLearner(small_config, "cpu")  # 真实小MLP、双Q及三个Adam。
    bank = api.observations.StaticHandBank.from_live(batch["obs"], torch.arange(4))  # 四资产真实静态字段。
    replay_module = importlib.import_module("anymani.distill.rl.flash_sac.replay")  # 同包真实CompactReplay。
    replay = replay_module.CompactReplay(small_config.replay_capacity, 4, torch.arange(4), device="cpu")  # T=64。
    done = torch.zeros(4, dtype=torch.bool)  # 三个合法未终止vector steps，成熟n-step窗口。
    for _ in range(3):  # 只形成 CPU 张量序列，不运行仿真。
        replay.append(
            api.observations.compact_observation(batch["obs"]),  # 当前动态帧，不把历史存入每条转移。
            batch["actions"],  # 当前动作[4,16]。
            batch["rewards"],  # 原始逐环境奖励[4]。
            done,  # 未发生物理失败。
            done,  # 未发生timeout。
            api.observations.compact_observation(batch["next_obs"]),  # 显式下一物理帧。
        )  # 完整显式物理终点字段。
        learner.observe_transition(batch["rewards"], done, done)  # reward trace 与真实新样本时钟同步。
    learner.act(batch["obs"], explore=True)  # 行为噪声的保持时钟和生成器都非初始状态。
    learner.update(batch)  # 真实反向/Adam，使 moments、BN、EMA 与更新相位都非空。
    identity = identity_module.build_method_identity(**identity_inputs)  # 与小网络config严格一致。
    arguments = {  # 同步调用写入，state_dict的共享tensor在调用结束前不再更新。
        "learner_state": learner.state_dict(),  # 完整算法状态。
        "identity": identity,  # 原生FlashSAC身份。
        "static_bank_state": bank.state_dict(),  # 每资产一份静态输入，不含learned activation。
        "replay_state": replay.state_dict(),  # 所有环槽、终点、段起点及资产路由。
        "experiment_state": {"collected_transitions": 12, "replay_rng": torch.Generator().manual_seed(91).get_state()},
    }
    return SimpleNamespace(learner=learner, bank=bank, replay=replay, arguments=arguments)  # 三个独立状态owner。


@pytest.mark.parametrize("completed_updates", [1, 2])
def test_real_learner_bank_and_full_replay_resume_exactly(
    checkpoint_module: ModuleType,  # 真实CPU序列化与身份闸门。
    api: SimpleNamespace,  # 真实算法状态owner。
    small_config: Any,  # 完整输入宽度、小隐层网络。
    batch: dict[str, Any],  # 下一次更新的共同参照样本。
    state_case: SimpleNamespace,  # 已有非空Adam、BN和行为噪声状态。
    tmp_path: Path,  # 测试独占保存位置。
    completed_updates: int,  # 奇/偶相位分别测试Actor更新与跳过。
) -> None:
    r"""分别在下一次跳过/执行 Actor 的相位保存；下一随机动作、更新结果及全部状态逐位恢复。"""
    learner = state_case.learner  # fixture已经完成一次真实优化。
    if completed_updates == 2:  # 第二种相位下一次将更新Actor和温度。
        learner.update(batch)  # 使用相同真实batch再执行一次完整Q更新。
    state_case.arguments["learner_state"] = learner.state_dict()  # 刷新标量计数与RNG快照。
    path = checkpoint_module.write_checkpoint(tmp_path / "resume.pt", **state_case.arguments)  # 真正torch.save。
    payload = checkpoint_module.read_checkpoint(  # 完整续接同时要求方法相同且存在回放。
        path,  # 实际已发布文件。
        expected_identity=state_case.arguments["identity"],  # 精确方法一致。
        require_replay=True,  # 不能静默建新池。
    )
    assert payload["schema_version"] == "flash-sac-checkpoint-1", "checkpoint schema 不符"  # 原生状态协议。
    assert payload["metadata"] == {"kind": "full_resume", "has_replay": True}, "完整恢复资格未明确记录"
    assert payload["learner"]["actor_optimizer"]["state"], "测试没有建立真实Adam moments"
    _assert_tree_equal(payload["experiment"], state_case.arguments["experiment_state"], "experiment")  # 包括抽样RNG。
    _assert_tree_equal(payload["replay"], state_case.replay.state_dict(), "full replay")  # 不能只保存cursor。
    restored_replay = type(state_case.replay)(  # 新池分配独立tensor，避免共享引用造成伪通过。
        small_config.replay_capacity,  # 同一个256-transition容量。
        4,  # 同步环境数N=4。
        torch.arange(4),  # 四个环境分别对应四种手型。
        device="cpu",  # 和保存状态同一CPU布局。
    )  # 相同布局。
    restored_replay.load_state_dict(payload["replay"])  # 生产load执行完整序列和路由核验。
    _assert_tree_equal(  # 同时比较全部已写和未写槽，不只检查cursor。
        restored_replay.state_dict(),  # 新实例的完整持久状态。
        state_case.replay.state_dict(),  # 原实例的独立数组参照。
        "restored replay",  # 精确零容差。
    )  # 整池逐位一致。
    restored_bank = api.observations.StaticHandBank(payload["static_bank"])  # 静态表的生产恢复入口。
    _assert_tree_equal(restored_bank.state_dict(), state_case.bank.state_dict(), "restored bank")  # 保持手型轴。

    # 先生成原实例的参考续接；新实例构造消耗的随机数必须被checkpoint完整撤销。
    reference_action = learner.act(batch["obs"], explore=True).actions.clone()  # 原实例未来行为样本。
    reference_info = learner.update(batch)  # 原实例下一完整更新。
    reference_state = deepcopy(learner.state_dict())  # 参数、moments、统计及全局/行为RNG。
    restored = api.learner.FlashSACLearner(small_config, "cpu")  # 网络形状保持完整，只有隐宽小。
    restored.load_state_dict(payload["learner"])  # 真实算法owner恢复其状态。
    _assert_tree_equal(restored.act(batch["obs"], explore=True).actions, reference_action, "next action")  # 逐位。
    _assert_tree_equal(restored.update(batch), reference_info, "next update diagnostics")  # 所有loss与计数。
    _assert_tree_equal(restored.state_dict(), reference_state, "next update state")  # 不止Actor权重。


def test_snapshot_declares_no_replay_and_all_loads_use_cpu(
    checkpoint_module: ModuleType, state_case: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""模型快照可省略回放；完整恢复不得把快照当作非空离策略数据池。"""
    arguments = {**state_case.arguments, "replay_state": None, "experiment_state": None}  # 合法模型快照。
    path = checkpoint_module.write_checkpoint(tmp_path / "snapshot.pt", **arguments)  # 写入同一原生schema。
    real_load = torch.load  # 包装保留真实反序列化。
    calls: list[dict[str, Any]] = []  # 审计CPU map_location而不是只观察输入本来在CPU。

    def checked_load(*args: Any, **kwargs: Any) -> Any:
        r"""无论保存时设备为何，I/O 边界明确映射到 CPU。"""
        calls.append(kwargs)  # 记录真实生产调用参数。
        return real_load(*args, **kwargs)  # 不返回伪造payload。

    monkeypatch.setattr(checkpoint_module.torch, "load", checked_load)  # 只观测实际加载行为。
    payload = checkpoint_module.read_checkpoint(path)  # 不指定expected仍需验证自身摘要/配置。
    assert payload["replay"] is None and payload["experiment"] == {}, "模型快照可选状态语义不符"
    assert payload["metadata"] == {"kind": "model_snapshot", "has_replay": False}, "快照被误标为完整resume"
    with pytest.raises(ValueError, match="replay"):  # 缺回放是明确的资格错误。
        checkpoint_module.read_checkpoint(path, require_replay=True)  # 不构造空池继续训练。
    assert calls and all(call["map_location"] == "cpu" and call["weights_only"] is True for call in calls)


@pytest.mark.parametrize("change", ["config", "task", "source", "provider"])
def test_resume_and_overwrite_reject_different_method(
    checkpoint_module: ModuleType,  # 读取与覆盖共用相同身份闸门。
    identity_module: ModuleType,  # 新身份仍由生产builder形成。
    identity_inputs: dict[str, Any],  # 可进行单因素干预的实验合同。
    state_case: SimpleNamespace,  # 原方法非空训练状态。
    tmp_path: Path,  # 本测试拥有的checkpoint路径。
    change: str,  # 分别改变Bellman参数、MDP、源码或冻结表示。
) -> None:
    r"""同形网络、同算法标签仍不足以续训；任何完整方法字段改变均拒绝读取与覆盖。"""
    path = checkpoint_module.write_checkpoint(tmp_path / "latest.pt", **state_case.arguments)  # 原方法目标。
    old_bytes = path.read_bytes()  # 拒绝覆盖后应逐字节保留。
    if change == "config":  # gamma变化保持网络shape兼容，但改变Bellman目标。
        identity_inputs["config"] = replace(identity_inputs["config"], gamma=0.98)  # 新配置身份。
    elif change == "task":  # MDP目标发生变化。
        identity_inputs["task_contract"]["rotation_progress_reward_weight"] = 5.0  # reward/rad。
    elif change == "provider":  # 相同维度的冻结N040权重改变。
        identity_inputs["provider_identity"]["identity_digest"] = "b" * 64  # 完整provider provenance。
    expected = identity_module.build_method_identity(**identity_inputs)  # 用生产builder生成自洽身份。
    if change == "source":  # 模拟另一个已审核源码版本，不改动任何实际生产文件。
        name = next(key for key in expected["implementation"]["files"] if key.endswith("flash_sac/math.py"))
        expected["implementation"]["files"][name] = "c" * 64  # 仅更改对照身份记录。
        body = {key: value for key, value in expected.items() if key != "identity_digest"}  # 摘要必须仍自洽。
        expected["identity_digest"] = identity_module._stable_digest(body)  # 不用旧摘要绕过全字段比较。
    with pytest.raises(ValueError, match="identity mismatch"):  # expected身份比较早于状态交付。
        checkpoint_module.read_checkpoint(path, expected_identity=expected, require_replay=True)
    other_state = {  # 移除配置自相矛盾这个混淆因素，只验证目标文件所有权。
        **state_case.arguments["learner_state"],  # 同形参数，排除shape错误。
        "config": expected["training"]["config"],  # 排除config自身矛盾，仅检验文件方法所有权。
    }
    with pytest.raises(ValueError, match="identity mismatch"):  # 文件所有权属于原完整方法。
        checkpoint_module.write_checkpoint(  # 同算法/同形参数也不能越过完整方法身份。
            path, **{**state_case.arguments, "identity": expected, "learner_state": other_state}
        )
    assert path.read_bytes() == old_bytes, "身份拒绝时损坏了既有checkpoint"


@pytest.mark.parametrize("fault", ["config", "missing_config", "digest", "schema", "metadata", "missing_replay"])
def test_invalid_saved_payload_fails_even_without_expected_identity(
    checkpoint_module: ModuleType, state_case: SimpleNamespace, tmp_path: Path, fault: str
) -> None:
    r"""自洽验证不依赖调用者提供 expected_identity；损坏/缺项不能静默当成模型快照。"""
    path = checkpoint_module.write_checkpoint(  # 使用真实I/O建立待注入故障的参照。
        tmp_path / "damaged.pt",  # 只有本测试可修改的故障文件。
        **state_case.arguments,  # 模型、静态表和完整回放均来自真实owner。
    )  # 先形成真实合法payload。
    payload = torch.load(path, map_location="cpu", weights_only=True)  # CPU真实加载，再注入一项故障。
    if fault == "config":  # learner与身份中的方法参数矛盾。
        payload["learner"]["config"]["gamma"] = 0.8  # 同形也要拒绝。
    elif fault == "missing_config":  # 固定键config不可省略。
        payload["learner"].pop("config")  # 不能只靠权重shape恢复。
    elif fault == "digest":  # 方法记录改变但仍冒用原摘要。
        payload["identity"]["task_contract"]["physics_hz"] = 60  # 控制物理合同漂移。
    elif fault == "schema":  # 外层版本决定状态解释规则。
        payload["schema_version"] = "unknown"  # 未知schema不能迁移猜测。
    elif fault == "metadata":  # 完整恢复标志必须与实际回放presence一致。
        payload["metadata"]["has_replay"] = False  # 自相矛盾元数据。
    else:  # 缺键与明确None语义不同。
        payload.pop("replay")  # 截断/不完整payload。
    torch.save(payload, path)  # 仅测试拥有的故障文件，不经过生产writer主动校验。
    with pytest.raises(ValueError):  # read边界拒绝，不等到learner部分写入后才失败。
        checkpoint_module.read_checkpoint(path)  # expected省略仍需自身合同成立。


def test_invalid_write_retains_target_and_validates_expected_digest(
    checkpoint_module: ModuleType, state_case: SimpleNamespace, tmp_path: Path
) -> None:
    r"""错误 config 在发布前拒绝；调用者携带被修改的 expected 身份也不能沿用旧摘要。"""
    path = checkpoint_module.write_checkpoint(tmp_path / "latest.pt", **state_case.arguments)  # 原始完整状态。
    original = path.read_bytes()  # 拒绝后无字节变化。
    learner_state = deepcopy(state_case.arguments["learner_state"])  # 保留原对象所有权。
    learner_state["config"]["target_sigma"] = 0.2  # 改变每有效关节目标熵。
    with pytest.raises(ValueError, match="config"):  # learner/identity双重记录必须一致。
        checkpoint_module.write_checkpoint(path, **{**state_case.arguments, "learner_state": learner_state})
    expected = deepcopy(state_case.arguments["identity"])  # caller的expected本身也需要核验。
    expected["task_contract"]["policy_hz"] = 10  # 不重算digest，故意制造过期标签。
    with pytest.raises(ValueError, match="digest"):  # 不能只比较identity_digest字符串。
        checkpoint_module.read_checkpoint(path, expected_identity=expected)
    assert path.read_bytes() == original, "错误配置写入没有保留旧目标"


def test_same_identity_latest_is_fsynced_then_atomically_replaced(
    checkpoint_module: ModuleType, state_case: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""发布前旧 latest 仍可读；同目录新文件 fsync 完成后才切换文件名。"""
    path = checkpoint_module.write_checkpoint(tmp_path / "latest.pt", **state_case.arguments)  # 第一次发布。
    events: list[str] = []  # 核对persist-before-publish的真实系统调用顺序。
    real_fsync, real_replace = checkpoint_module.os.fsync, checkpoint_module.os.replace  # 保留实际I/O。

    def fsync(fd: int) -> None:
        r"""真正同步新文件内容，并记录它先于rename发布。"""
        real_fsync(fd)  # 真正调用内核fsync。
        events.append("fsync")  # 仅在成功返回后记录。

    def replace_file(source: Any, target: Any) -> None:
        r"""在最终发布线之前，外部读者只能看到完整旧目标。"""
        assert Path(source).parent == path.parent and Path(target) == path, "临时文件没有与目标位于同一目录"
        assert events == ["fsync"], f"publish前内容尚未同步：{events}"  # flush/fsync先于replace。
        assert checkpoint_module.read_checkpoint(path)["experiment"]["collected_transitions"] == 12, "旧目标提前改变"
        assert checkpoint_module.read_checkpoint(Path(source))["experiment"]["collected_transitions"] == 16, (
            "临时文件不完整"
        )
        real_replace(source, target)  # 真正执行原子文件名切换。
        events.append("replace")  # 记录提交点。

    monkeypatch.setattr(checkpoint_module.os, "fsync", fsync)  # 仅包装真实系统调用。
    monkeypatch.setattr(checkpoint_module.os, "replace", replace_file)  # 同样保留真实发布。
    result = checkpoint_module.write_checkpoint(
        path, **{**state_case.arguments, "experiment_state": {"collected_transitions": 16}}
    )
    assert result == path and events == ["fsync", "replace"], "latest没有执行先同步后原子替换"
    assert checkpoint_module.read_checkpoint(path)["experiment"]["collected_transitions"] == 16, "新状态未发布"
    assert not list(tmp_path.glob(".latest.pt.*.tmp")), "成功发布留下临时checkpoint"


@pytest.mark.parametrize("stage", ["save", "fsync", "replace"])
@pytest.mark.parametrize("existing", [False, True])
def test_failed_publication_keeps_old_target_and_cleans_owned_temp(
    checkpoint_module: ModuleType,  # 所有故障都经过生产writer的清理路径。
    state_case: SimpleNamespace,  # 合法科学状态，排除身份错误混淆。
    tmp_path: Path,  # 本测试独占文件目录。
    monkeypatch: pytest.MonkeyPatch,  # 只控制实际I/O报错时点。
    stage: str,  # 序列化、持久同步、原子替换三个阶段。
    existing: bool,  # 首次保存与覆盖旧latest两种生命周期。
) -> None:
    r"""序列化、内容同步或发布故障时，已有目标不变；首次失败不能留下看似完整的产物。"""
    path = tmp_path / "latest.pt"  # checkpoint与临时文件的同目录边界。
    if existing:  # 同时覆盖首次保存和latest覆盖两种生命周期。
        checkpoint_module.write_checkpoint(path, **state_case.arguments)  # 先发布旧状态。
    before = path.read_bytes() if existing else None  # 原状态的字节参照。

    def fail(*args: Any, **kwargs: Any) -> None:
        r"""模拟真实I/O报错；save先写部分字节，以检验残片清理。"""
        if stage == "save":  # 失败发生在流已被打开且已有内容之后。
            args[1].write(b"partial-checkpoint")  # 真正产生一个不完整临时文件。
        raise OSError(f"injected-{stage}-failure")  # 原始失败必须传给caller。

    owner = checkpoint_module.torch if stage == "save" else checkpoint_module.os  # 故障发生的真实I/O层。
    monkeypatch.setattr(owner, stage, fail)  # 不修改环境、学习器或回放算法。
    with pytest.raises(OSError, match=f"injected-{stage}"):  # 异常不能转为成功路径返回。
        checkpoint_module.write_checkpoint(path, **state_case.arguments)  # 原身份合法，隔离I/O故障命题。
    assert (path.read_bytes() if path.exists() else None) == before, f"{stage}故障损坏了旧目标"
    assert not list(tmp_path.glob(".latest.pt.*.tmp")), f"{stage}故障遗留了owned临时残片"


def test_competing_methods_cannot_both_claim_unpublished_latest(
    checkpoint_module: ModuleType,  # 生产目录锁与原子发布实现。
    identity_module: ModuleType,  # 两个自洽但不同的实验身份。
    identity_inputs: dict[str, Any],  # 仅修改第二方法的入口条件。
    state_case: SimpleNamespace,  # 共享只读训练状态，线程内没有优化器更新。
    tmp_path: Path,  # 两线程竞争同一个测试路径。
    monkeypatch: pytest.MonkeyPatch,  # 事件钩子保留真实内核锁与序列化。
) -> None:
    r"""首个writer尚在写临时文件时，第二方法必须等待并在发布后拒绝，而非覆盖第一方法。

    两条CPU线程使用各自生产writer打开的目录描述符，真实flock决定所有权。
    Event仅控制实验先后，等待有5秒上限；不依赖sleep或重复轮询制造竞态。
    """
    path = tmp_path / "latest.pt"  # 两个方法竞争同一个尚不存在的checkpoint路径。
    entered, release = Event(), Event()  # 固定首个writer已持锁、尚未发布的窗口。
    real_save, real_flock = checkpoint_module.torch.save, checkpoint_module.fcntl.flock  # 保留真实I/O。
    attempted = Event()  # 第二个writer确实进入锁竞争，而非主线程过早放行。
    identity_inputs["run_contract"]["experiment_condition"] = "second-method"  # 合法但不同的实验身份。
    other = {**state_case.arguments, "identity": identity_module.build_method_identity(**identity_inputs)}  # 同形网络。

    def held_save(payload: Any, stream: Any) -> None:
        r"""在真实首个序列化之前固定持锁窗口，两个线程都仍调用生产writer。"""
        if payload["identity"]["identity_digest"] == state_case.arguments["identity"]["identity_digest"]:
            entered.set()  # 首个writer已经取得目录锁并通过目标所有权检查。
            assert release.wait(5), "测试未释放首个writer"  # 有界同步，失败也能结束线程。
        real_save(payload, stream)  # 仍然保存真实learner/static/replay状态。

    def observed_flock(fd: int, operation: int) -> None:
        r"""第二writer开始竞争时发出事件，锁本身仍由内核flock实现。"""
        if entered.is_set():  # 首个writer进入save后，后续调用必属第二writer。
            attempted.set()  # 第二线程已经到达实际锁调用。
        real_flock(fd, operation)  # 阻塞式互斥语义没有被测试替代。

    monkeypatch.setattr(checkpoint_module.torch, "save", held_save)  # 不伪造checkpoint内容。
    monkeypatch.setattr(checkpoint_module.fcntl, "flock", observed_flock)  # 保留真实目录锁。
    with ThreadPoolExecutor(max_workers=2) as workers:  # 只并发文件发布，不执行任何模型前向。
        first = workers.submit(checkpoint_module.write_checkpoint, path, **state_case.arguments)  # 首个方法。
        try:  # 任何assert失败也须释放持锁线程，防止测试阻塞退出。
            assert entered.wait(5), "首个writer没有到达持锁序列化点"  # 确定竞争窗口已建立。
            second = workers.submit(checkpoint_module.write_checkpoint, path, **other)  # 不同方法。
            assert attempted.wait(5), "第二writer没有竞争发布锁"  # 竞争真实发生。
        finally:
            release.set()  # 首个方法完成发布，第二方法此后必须看见原方法身份。
        assert first.result(timeout=5) == path, "首个方法没有完成发布"  # 验证真实成功结果。
        with pytest.raises(ValueError, match="identity mismatch"):  # 第二个方法不得覆盖先到的有效目标。
            second.result(timeout=5)  # 生产writer的身份错误穿过Future原样返回。
    saved = checkpoint_module.read_checkpoint(path, expected_identity=state_case.arguments["identity"])  # 原方法保留。
    assert saved["identity"] == state_case.arguments["identity"], "并发争用篡改了文件的方法所有权"
    assert not list(tmp_path.glob(".latest.pt.*.tmp")), "并发拒绝后留下临时checkpoint"
