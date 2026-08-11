r"""Task-free multi-anchor geometry SSL pretraining entry.

运行入口：

``python -m anymani.distill.ssl.pretrain assets.train_paths='[/abs/hand_bundle]'``

进程只使用 assets/robots 的静态资产、PyTorch 与 Warp，不启动 Isaac Sim。train/validation generated
资产按内容哈希隔离；official 资产只写入独立 evaluation manifest，不参与权重更新、损失校准或
checkpoint 选择。
"""

from __future__ import annotations

import subprocess  # Git revision 只读查询，不通过 shell 执行
from dataclasses import asdict  # manifest 冻结结构写入 checkpoint metadata
from datetime import UTC, datetime  # run directory 使用 UTC 绝对时间
from importlib.metadata import PackageNotFoundError, version  # installed/editable 版本证据
from pathlib import Path  # 资产与运行目录路径
from typing import Literal  # generated/official 解析边界

import hydra  # 唯一 geometry SSL CLI 入口
import torch  # 模型、optimizer、CUDA 与 autograd
from hydra.core.config_store import ConfigStore  # 注册默认实验 mapping
from omegaconf import DictConfig, OmegaConf  # CLI override resolution

from anymani.assets.asset_schema_geometry import SEMANTICS_SCHEMA_VERSION  # checkpoint 资产 schema 证据
from anymani.assets.bank.hand_bank import HandBank, HandBankCfg  # 资产集合唯一入口
from anymani.assets.bank.hand_container import HandContainer, HandContainerCfg  # 显式 bundle 选择
from anymani.distill.diagnostics.recording.geometry_ssl import GeometrySSLRunLogger  # TB/JSONL/NPZ
from anymani.distill.models.geometry_ssl import GeometrySSLForward, GeometrySSLModel  # 网络与预测类型
from anymani.distill.objectives.representations.field_reconstruction import (  # 五项联合目标
    GeometrySSLObjective,
    GeometrySSLTerms,
)
from anymani.distill.ssl.checkpoint import (  # 完整 resume + retained transfer artifact
    GeometrySSLCheckpointMetadata,
    save_geometry_ssl_checkpoint,
)
from anymani.distill.ssl.config import (  # resolved experiment/split 合同
    GeometrySSLAssetManifest,  # 内容哈希 split
    GeometrySSLExperimentCfg,  # 冻结根配置
    experiment_config_from_dict,  # Hydra mapping -> dataclasses
    resolved_config_dict,  # metadata/YAML mapping
    write_resolved_experiment_files,  # run 前置 artifacts
)
from anymani.distill.ssl.dataset import (  # CPU cache、GPU state、online teacher
    OnlineGeometryBatcher,  # 轮转资产与 Sobol q
    PaddedOnlineGeometryBatch,  # `[B,20]/[B,26]` 稠密容器
    materialize_geometry_asset_runtime,  # CPU Manifold/home/anchor/workspace
    move_geometry_asset_to_device,  # spec/evidence/Warp BVH 上传
)


def _torch_dtype(name: str) -> torch.dtype:
    r"""把 resolved 字符串限制为显式训练 dtype。

    Args:
        name (str): ``float32`` 或数值 reference 用 ``float64``。

    Returns:
        torch.dtype: 对应 PyTorch dtype。

    Raises:
        ValueError: 其他 dtype 名称均拒绝，避免隐式 AMP 改变 Warp/teacher 合同。
    """

    if name == "float32":  # 正式 Warp/训练主路径
        return torch.float32  # CUDA float32
    if name == "float64":  # 纯 tensor 数值诊断
        return torch.float64  # Warp target 当前仍会拒绝该配置
    raise ValueError(f"unsupported geometry SSL dtype={name!r}")  # 不猜 autocast


def _resolve_assets(
    paths: tuple[str, ...],  # 显式 bundle roots
    *,
    source_kind: Literal["generated", "official"],  # 迁移/fail-closed 路由
) -> tuple[HandContainer, ...]:
    r"""通过 HandBank explicit route 解析资产；distill 不读取 sidecar/URDF 细节。

    ``paths`` 可同时列 pre-made 母体、post-mutate variants 和跨 family/generated 不同 DOF bundles；
    bank 为每项交付同一 ``HandContainer.geometry_semantics``。official 缺人工核验语义时在这里 fail closed。
    """

    if not paths:  # validation/official splits 允许为空
        return ()  # 空 tuple 保持不可变集合语义
    selection = HandBank(  # 所有 sidecar/URDF/path 细节封装在 assets 层
        HandBankCfg(  # explicit manifest 不依赖 collection root 目录布局
            source_mode="post_mutate",  # explicit route 下仅作为 provenance，不触发 post-mutate discovery
            selection_mode="explicit",  # paths 精确冻结到 resolved config
            containers=tuple(  # 每项声明 source kind，决定 legacy migration/official fail-closed
                HandContainerCfg(path=path, source_kind=source_kind) for path in paths
            ),
            require_geometry_semantics=True,  # distill 必须得到 owner/kinematics/anchor 语义
        )
    ).resolve()  # 此处执行文件 IO 与内容哈希验证
    return selection.assets  # 稳定顺序与配置 paths 一致


def _manifest_record(container: HandContainer) -> dict[str, str]:
    r"""提取不依赖训练代码命名的资产身份、形态与内容哈希。

    Returns:
        dict[str, str]: 可直接写 YAML/checkpoint 的单资产身份记录。
    """

    semantics = container.geometry_semantics  # bank 已验证的静态语义
    if semantics is None:  # manifest 不允许仅凭路径/ID 标识训练资产
        raise ValueError("manifest asset is missing geometry semantics")  # 要求调用方启用 geometry semantics
    return {  # 全部转换字符串，YAML/JSON consumer 无 Python enum 依赖
        "asset_id": container.asset_id,  # bank 稳定 ID
        "content_hash": semantics.content_hash,  # SHA-256 leakage/cache 主键
        "source_kind": semantics.source_kind,  # generated/official
        "topology_key": semantics.topology_key or "",  # morphology split 身份
        "family": semantics.family,  # leap/allegro/mixed/generic
        "handedness": semantics.handedness,  # left/right/unknown
        "joint_count": str(len(semantics.active_joint_names)),  # 实际 $N_J$
        "owner_count": str(len(semantics.owners)),  # 实际 $G$
    }


def _build_manifest(
    train_assets: tuple[HandContainer, ...],  # generated optimizer split
    validation_assets: tuple[HandContainer, ...],  # generated held-out split
    official_assets: tuple[HandContainer, ...],  # frozen evaluation split
) -> GeometrySSLAssetManifest:
    r"""冻结 split 并在开始物化 GPU cache 前拒绝内容泄漏。

    GeometrySSLAssetManifest 构造时比较 SHA-256 集合；任一内容跨 split 重用立即抛错。
    """

    return GeometrySSLAssetManifest(  # 构造即执行 leakage contract
        schema_version="1.0.0",  # manifest schema
        train=tuple(_manifest_record(asset) for asset in train_assets),  # optimizer assets
        validation=tuple(_manifest_record(asset) for asset in validation_assets),  # fixed bank
        official_evaluation=tuple(_manifest_record(asset) for asset in official_assets),  # isolated
    )


def _code_revision() -> str:
    r"""尽力记录当前 Git revision；非 Git 安装返回明确 ``unknown``。"""

    try:  # 只读 Git query 不影响工作树
        result = subprocess.run(  # 不使用 shell，避免路径/override 注入
            ["git", "rev-parse", "HEAD"],  # 当前 checkout commit
            check=True,  # 非 Git/失败进入明确 fallback
            capture_output=True,  # 不污染训练 stdout
            text=True,  # revision 以 str 记录
            timeout=5,  # 网络无关命令不应阻塞训练启动
        )
    except (OSError, subprocess.SubprocessError):  # git 缺失、非 repo、timeout
        return "unknown"  # metadata 显式不可用，不伪造 revision
    return result.stdout.strip() or "unknown"  # 空输出同样显式 unknown


def _package_version() -> str:
    r"""读取 installed AnyMani distribution version；editable 未登记时显式标注。"""

    try:  # importlib metadata 不 import Isaac Lab runtime
        return version("anymani")  # distribution version 字符串
    except PackageNotFoundError:  # source checkout 尚未安装 metadata
        return "editable-unknown"  # 与正式版本号区分


def _forward_objective(
    model: GeometrySSLModel,  # retained+disposable 网络
    objective: GeometrySSLObjective,  # 五项联合目标
    batch: PaddedOnlineGeometryBatch,  # q/evidence/query/teacher/masks
) -> tuple[GeometrySSLForward, GeometrySSLTerms]:
    r"""在物理 q 上保留 Sobolev 图并计算一份完整联合损失。

    sampler 与 Warp teacher 都停止梯度；新建 leaf q 后通过 encoder/decoder 对物理 rad 求导：
    $\hat g^{auto}=\partial\hat\rho/\partial q_i$。返回 prediction 供 dense 记录，terms 供 backward/logging。
    """

    q = batch.q.detach().requires_grad_(True)  # sampler/teacher stop-gradient；对物理 rad q 求导
    prediction = model(  # 同一次 density forward 服务重建与 Sobolev 导数
        q,  # `[B,20]` physical rad；padding coordinates masked
        batch.evidence,  # `[B,26,...]` static evidence
        batch.queries.query_points_h,  # `[B,26,N_Q,3]` fixed `{h}`，m
        owner_index=batch.sensitivity_targets.owner_index,  # `[B,E]`
        query_index=batch.sensitivity_targets.query_index,  # `[B,E]`
        joint_index=batch.sensitivity_targets.joint_index,  # `[B,E]`
    )
    terms = objective(  # mask-aware scalar reduction
        q=q,  # Sobolev autograd source
        density_prediction=prediction.density,  # `[B,26,N_Q,L]`
        kappa_prediction=prediction.kappa,  # `[B,E]`
        field_targets=batch.field_targets,  # $d/\\rho$ 与 query valid mask
        sensitivity_targets=batch.sensitivity_targets,  # $\\kappa/g$ 与 edge valid mask
    )
    return prediction, terms  # caller 决定 backward/eval/logging


def run_geometry_ssl_pretraining(
    config: GeometrySSLExperimentCfg,  # 完整 frozen resolved config
    *,
    output_dir_override: Path | None = None,  # tests/sanity 的显式隔离目录
) -> Path:
    r"""执行 asset materialization、online training、fixed validation、logging 与 checkpoint。

    生命周期顺序固定为：resolve+hash split → 写 resolved evidence → CPU owner materialization → GPU Warp cache →
    fixed validation bank → model/optimizer → online train。official assets 在 manifest 后不进入任何后续列表。

    Returns:
        Path: 本次运行的 artifact 根目录。

    Raises:
        ValueError: 无训练资产、split 泄漏或底层物理合同失败时抛出。
        RuntimeError: 配置 CUDA 不可用或 Warp cache 失败时抛出。
        FloatingPointError: 任一 optimizer step 梯度范数非有限时抛出。
    """

    if not config.assets.train_paths:  # optimizer split 不允许为空
        raise ValueError("geometry SSL requires at least one generated train asset path")  # 启动前失败
    torch.manual_seed(config.train.seed)  # model 初始化与 PyTorch 路径全局复现锚点
    device = torch.device(config.train.device)  # resolved GPU device
    if device.type == "cuda" and not torch.cuda.is_available():  # 不自动回退 CPU teacher
        raise RuntimeError(f"configured CUDA device is unavailable: {device}")  # 明确环境错误
    dtype = _torch_dtype(config.train.dtype)  # model/spec/evidence 共享 dtype

    train_assets = _resolve_assets(config.assets.train_paths, source_kind="generated")  # optimizer split
    validation_assets = _resolve_assets(config.assets.validation_paths, source_kind="generated")  # held-out
    official_assets = _resolve_assets(config.assets.official_evaluation_paths, source_kind="official")  # identity only
    manifest = _build_manifest(train_assets, validation_assets, official_assets)  # SHA-256 leakage 闸门
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")  # 可排序 UTC run identity
    output_dir = output_dir_override or (  # tests 可完全控制临时目录
        Path(config.train.output_dir) / config.train.experiment_name / timestamp  # 默认 run root
    )
    write_resolved_experiment_files(output_dir, config=config, manifest=manifest)  # 物化失败也留配置证据

    train_states = [  # 每项 generated train asset 只物化一次 CPU/GPU static state
        move_geometry_asset_to_device(  # 上传 spec/evidence 并构建 Warp BVH
            materialize_geometry_asset_runtime(  # strict Manifold/home/anchors/workspace
                asset,  # 当前 HandContainer
                query_config=config.query,  # workspace bank $N_W$
                config=config.materialization,  # home/anchor 点预算
            ),
            device=device,  # 训练 GPU
            dtype=dtype,  # 与模型一致
        )
        for asset in train_assets  # 保留 manifest 顺序供 round-robin
    ]
    validation_states = [  # generated held-out assets 独立 cache，不与 optimizer split 混合
        move_geometry_asset_to_device(  # 每项只构建一次 GPU BVH
            materialize_geometry_asset_runtime(  # 同一静态物化超参保证测度可比
                asset,  # held-out HandContainer
                query_config=config.query,  # 同 query 比例/预算
                config=config.materialization,  # 同 home/anchor 预算
            ),
            device=device,  # 同一 GPU
            dtype=dtype,  # 同一 dtype
        )
        for asset in validation_assets  # 固定 manifest 顺序
    ]
    train_batcher = OnlineGeometryBatcher(  # 每 step 轮转资产并在线生成 teacher
        train_states,  # generated-only GPU states
        seed=config.train.seed,  # 每资产独立 scrambled Sobol 序列
        query_config=config.query,  # 50/25/25 query
        target_config=config.target,  # $d/\\rho/\\kappa/g$
        padding=config.padding,  # 20 JOINT/26 owner
    )
    validation_batch: PaddedOnlineGeometryBatch | None = None  # 无 held-out assets 时显式 None
    if validation_states:  # validation q/query/target 只在启动时采一次
        validation_batch = OnlineGeometryBatcher(  # 使用独立 Sobol seed，不推进 train 序列
            validation_states,  # generated held-out GPU states
            seed=config.train.seed + 1_000_003,  # 与训练 seed 分离的确定性大质数偏移
            query_config=config.query,  # 同 query 测度
            target_config=config.target,  # 同 teacher
            padding=config.padding,  # 同稠密容器
        ).sample(batch_size=len(validation_states), step=0)  # 固定 validation q/query/target bank

    model = GeometrySSLModel(config.model).to(device=device, dtype=dtype)  # retained+disposable 全部训练
    objective = GeometrySSLObjective(config.objective)  # 五项权重冻结
    optimizer = torch.optim.AdamW(  # 第一版统一 optimizer
        model.parameters(),  # encoder 与两个 SSL-only decoders
        lr=config.optimizer.learning_rate,  # $\\eta$
        weight_decay=config.optimizer.weight_decay,  # 解耦衰减
    )
    metadata = GeometrySSLCheckpointMetadata(  # 全部 checkpoints 共享启动时冻结 metadata
        code_revision=_code_revision(),  # Git commit/unknown
        package_version=_package_version(),  # AnyMani version
        geometry_semantics_schema=SEMANTICS_SCHEMA_VERSION,  # assets schema
        asset_manifest=asdict(manifest),  # split 内容哈希
        resolved_config=resolved_config_dict(config),  # 完整实验图
    )
    logger = GeometrySSLRunLogger(output_dir)  # TensorBoard/JSONL/NPZ
    last_batch: PaddedOnlineGeometryBatch | None = None  # 最终 train dense snapshot 数据
    last_prediction: GeometrySSLForward | None = None  # 最终 train dense snapshot 预测

    try:  # 无论训练成功/失败都关闭 TensorBoard writer
        for step in range(1, config.train.steps + 1):  # optimizer step 从 1 计数
            model.train()  # 启用训练模式；默认 dropout=0 仍保持生命周期明确
            optimizer.zero_grad(set_to_none=True)  # 避免旧梯度累积到下一 optimizer step
            for accumulation_index in range(config.train.gradient_accumulation_steps):  # microbatches
                batch = train_batcher.sample(  # 新资产路由、新 Sobol q、新 shell/adjacent query/teacher
                    batch_size=config.train.batch_size,  # 当前 microbatch $B$
                    step=(step - 1) * config.train.gradient_accumulation_steps + accumulation_index,  # 唯一采样步
                )
                prediction, terms = _forward_objective(model, objective, batch)  # 五项 loss + Sobolev graph
                (terms.total / config.train.gradient_accumulation_steps).backward()  # microbatch 均值梯度
                last_batch, last_prediction = batch, prediction  # 只保留最后 microbatch 用于 NPZ
            gradient_norm = torch.nn.utils.clip_grad_norm_(  # clip 前返回全参数 L2 norm
                model.parameters(), config.optimizer.max_gradient_norm
            )
            if not torch.isfinite(gradient_norm):  # 非有限更新不得写入参数
                raise FloatingPointError(f"non-finite gradient norm at step={step}: {float(gradient_norm)}")
            optimizer.step()  # 完成一次 AdamW update

            if step % config.train.log_every_steps == 0 or step == 1:  # 首步永远保留启动证据
                logger.log_terms(  # 当前记录最后 microbatch terms；累积语义在 resolved config 可见
                    step=step,  # optimizer step
                    split="train",  # TensorBoard/JSONL namespace
                    terms=terms,  # 六项标量
                    asset_ids=batch.asset_ids,  # 当前 microbatch 路由
                    gradient_norm=float(gradient_norm),  # clip 前范数
                )
            if validation_batch is not None and (  # 无 validation split 时整段不执行
                step % config.train.validation_every_steps == 0 or step == config.train.steps  # 周期+最终
            ):
                model.eval()  # 同一参数，固定 validation bank
                validation_prediction, validation_terms = _forward_objective(  # 仍需 grad 构造 Sobolev
                    model, objective, validation_batch
                )
                logger.log_terms(  # validation 不记录 gradient norm
                    step=step,  # 当前 optimizer step
                    split="validation",  # 独立 namespace
                    terms=validation_terms,  # 固定 bank 五项 loss
                    asset_ids=validation_batch.asset_ids,  # held-out asset IDs
                )
                logger.save_dense_snapshot(  # latent/mask/error post-hoc 证据
                    step=step,  # 文件名 step
                    split="validation",  # 文件名前缀
                    prediction=validation_prediction,  # held-out prediction
                    batch=validation_batch,  # 固定 target/masks
                )
            if step % config.train.checkpoint_every_steps == 0 or step == config.train.steps:  # 周期+最终
                save_geometry_ssl_checkpoint(  # 完整 resume + retained transfer state
                    output_dir / "checkpoints" / f"step_{step:08d}.pt",  # 稳定可排序路径
                    model=model,  # encoder+decoders
                    optimizer=optimizer,  # AdamW moments
                    step=step,  # 生命周期坐标
                    metadata=metadata,  # 启动时冻结科研合同
                )

        if last_batch is not None and last_prediction is not None:  # 至少完成一个 microbatch
            logger.save_dense_snapshot(  # 最终 train dense artifact
                step=config.train.steps,  # 最终 optimizer step
                split="train",  # train 文件前缀
                prediction=last_prediction,  # 最后一份在线 q 预测
                batch=last_batch,  # 对应 teacher/masks
            )
    finally:  # checkpoint/target/backward 抛错也执行
        logger.close()  # flush event writer
    return output_dir  # CLI/测试获得 artifact root


ConfigStore.instance().store(  # import 时注册唯一默认 geometry SSL 配置
    name="geometry_ssl",  # Hydra config name
    node=resolved_config_dict(GeometrySSLExperimentCfg()),  # 可变 mapping 支持 CLI overrides
)


@hydra.main(version_base="1.3", config_name="geometry_ssl")
def main(config: DictConfig) -> None:
    r"""Hydra CLI：解析全部 overrides、重建冻结配置后启动训练。

    Args:
        config (DictConfig): Hydra 合成的可变 mapping；进入 trainer 前转成 validated dataclasses。
    """

    payload = OmegaConf.to_container(config, resolve=True)  # interpolation 全部求值
    if not isinstance(payload, dict):  # 根配置必须 mapping
        raise TypeError("resolved Hydra geometry SSL config must be a mapping")  # 不接受 list/scalar
    normalized_payload = {str(key): value for key, value in payload.items()}  # 收窄 DictKeyType
    resolved = experiment_config_from_dict(normalized_payload)  # 逐层 dataclass 验证
    output_dir = run_geometry_ssl_pretraining(resolved)  # 执行完整生命周期
    print(output_dir)  # shell/调度器唯一 stdout 结果：artifact root


if __name__ == "__main__":  # `python -m anymani.distill.ssl.pretrain`
    main()  # Hydra 接管 CLI overrides


__all__ = ["main", "run_geometry_ssl_pretraining"]  # CLI 与 programmatic 入口
