r"""Contact sensor layout utilities for GM generated-hand tasks.

本模块是 `tasks/gm` 对 hand sidecar 的最小消费面：它只读取上游已经生成好的
`hand.yaml -> hand_cfg`，把“哪些 link 是 fingertip，哪些 link 是 non-tip 接触部位”
转换成 Isaac Lab scene 里的 per-link `ContactSensorCfg`。它不生成资产、不验证几何
闭包，也不决定 asset bank 的训练切分；这些仍分别属于 `assets` 与 `distill`。

核心设计决策：

1. contact topology 的唯一真源是 selected hand 的 sidecar，而不是 GM env 中硬编码
   `index/middle/ring/thumb` 四指名称；
2. 默认只读取第一个 selected asset，因为当前训练 slice 由 `HandSpawnCfg.validate_same_schema=True`
   约束为 same-topology，多资产全量 sidecar 校验可通过 `validate_all_assets=True` 打开；
3. 每个 link 单独声明一个 `ContactSensorCfg`，并设置
   `filter_prim_paths_expr=["{ENV_REGEX_NS}/object"]`。Isaac Lab filtered contact 的可靠语义是
   “一个 sensor body 对多个 filtered bodies”，不是“多个 robot bodies 聚合到一个 sensor 再过滤到
   一个 object”，因此这里不使用 regex aggregate sensor。

TODO(tactile rotation baseline):
    palm-supported tactile rotation 需要把当前 `non_tip` 集合拆成两个任务角色：

    - palm：合法支撑，进入 privileged critic 与独立 support-force metric，不进入 bad-contact penalty；
    - finger non-tip：19 个非指尖 finger links，进入 bad-contact penalty 与接触归因诊断。

    这个拆分必须由 sidecar link role 推导，不能依赖 `non_tip_sensor_names[1:]` 之类的偶然顺序。
    contact layout 仍只描述 topology，不拥有 EMA、reward curriculum 或 palm-supported 任务权重。

TODO(shared contact state):
    新基线需要一个 policy-rate、reset-aware 的 object-contact state owner。每个 sensor 先在
    body/filter pair 上取最大力幅值，随后做：

    $$
    \bar f_t=0.5\bar f_{t-1}+0.5f_t,
    \qquad
    c_t=\mathbf{1}[\bar f_t>0.25\ \mathrm{N}].
    $$

    actor observation、good-tip reward 与 bad-finger-non-tip reward 必须读取同一 buffer，避免
    同一个物理接触在不同 consumer 中得到互相矛盾的 0/1 判定。该 state 每个 policy step
    更新一次；ContactSensor physics history 不能冒充 policy history。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from anymani.robots.hand_spawn import HandSpawnCfg


@dataclass(frozen=True)
class GmContactSensorLayout:
    r"""由 hand sidecar 推导出的 GM contact sensor 拓扑布局。

    `layout` 保存的是任务语义需要的 link / sensor 名称，而不是 Isaac Lab runtime sensor
    对象本身。这样 observation / reward 只依赖 `sensor_names`，scene 装配阶段再把这些
    名称 lower 成真正的 `ContactSensorCfg`。

    Args:
        source_asset_id (str): 产生该 layout 的 hand asset id，仅用于报错和日志追踪。
        palm_link_name (str): palm/root link 名称，通常为 `palm`。
        finger_link_chains (tuple[tuple[str, ...], ...]): 每根 finger 的 child link 链；
            该字段服务 generated structural collision filter，不要求固定四指名称。
        fingertip_link_names (tuple[str, ...]): `is_tip=True` joint 的 child link，顺序沿 sidecar finger/joint 顺序。
        finger_non_tip_link_names (tuple[str, ...]): 所有 `is_tip=False` joint child link，顺序沿运动链展开。
        fingertip_sensor_names (tuple[str, ...]): 与 fingertip link 一一对应的 scene sensor 名称。
        finger_non_tip_sensor_names (tuple[str, ...]): 与 finger non-tip link 一一对应的 scene sensor 名称。
    """

    source_asset_id: str
    """产生该 layout 的 hand asset id；用于定位 sidecar / bank selection 问题。"""

    palm_link_name: str
    """掌部 root link 名称；bad-contact penalty 默认把 palm 接触视为 non-tip 接触。"""

    finger_link_chains: tuple[tuple[str, ...], ...]
    """每根 generated finger 的 link 链；结构碰撞过滤用它区分 same-finger 与 cross-finger。"""

    fingertip_link_names: tuple[str, ...]
    """所有 fingertip link 名称；`is_tip=True` 是 sidecar 中的显式语义标签。"""

    finger_non_tip_link_names: tuple[str, ...]
    """所有非指尖 finger child link；palm 不在该集合中，因此可直接用于 bad-contact penalty。"""

    fingertip_sensor_names: tuple[str, ...]
    """所有 fingertip sensor 名称；obs/reward 按该顺序拼接接触信号。"""

    finger_non_tip_sensor_names: tuple[str, ...]
    """所有 finger non-tip sensor 名称；bad-contact reward 使用该集合做 OR 聚合。"""

    @property
    def palm_sensor_name(self) -> str:
        r"""返回 palm object-filtered sensor 名称。

        Palm 是 tactile-rotation 任务中的合法支撑面，必须能独立进入 privileged critic 和
        support metric，不能通过 `non_tip_sensor_names[0]` 这类偶然位置恢复角色。
        """

        return _sensor_name_for_link(self.palm_link_name)  # 单一 palm role 对应单一 object-filtered sensor

    @property
    def non_tip_link_names(self) -> tuple[str, ...]:
        r"""返回旧 GM probe 使用的 palm-first non-tip 聚合视图。

        该 property 只维护已有环境的 contact contract；新 tactile rotation reward 必须使用
        `finger_non_tip_link_names`，从而把 palm 接触保持为中性支撑。
        """

        return (self.palm_link_name, *self.finger_non_tip_link_names)  # palm + 纯 finger non-tip links

    @property
    def non_tip_sensor_names(self) -> tuple[str, ...]:
        r"""返回与 `non_tip_link_names` 同序的旧聚合 sensor 视图。"""

        return (self.palm_sensor_name, *self.finger_non_tip_sensor_names)  # 旧 reward 保持 palm-first 语义

    @property
    def all_sensor_names(self) -> tuple[str, ...]:
        r"""返回 scene 中应安装的全部 ContactSensor 名称。"""

        return self.fingertip_sensor_names + self.non_tip_sensor_names  # 先 tip 后 non-tip，便于视觉/日志核对

    @property
    def all_link_names(self) -> tuple[str, ...]:
        r"""返回全部被 ContactSensor 覆盖的 robot link 名称。"""

        return self.fingertip_link_names + self.non_tip_link_names  # 与 `all_sensor_names` 保持一一对应的顺序

    @property
    def sensor_link_pairs(self) -> tuple[tuple[str, str], ...]:
        r"""返回 `(sensor_name, link_name)` 对，供 scene 安装阶段逐项 lower。"""

        return tuple(zip(self.all_sensor_names, self.all_link_names, strict=True))  # 每个 sensor 精确绑定一个 body link


def build_contact_sensor_layout_from_hand_spawn(
    hand_spawn_cfg: HandSpawnCfg,
    *,
    validate_all_assets: bool = False,
) -> GmContactSensorLayout:
    r"""从 `HandSpawnCfg` 当前选中的 hand bank 构造 contact sensor layout。

    默认只读取第一个 selected asset：这与当前 same-topology training slice 的物理假设一致，
    即所有 selected assets 共享同一 link / joint schema，几何参数可以变，但 contact topology
    不变。若正在调试跨 topology bank，可打开 `validate_all_assets=True` 做全量 sidecar 对照。

    Args:
        hand_spawn_cfg (HandSpawnCfg): GM hand spawn 配置，内部含 asset bank selection contract。
        validate_all_assets (bool): 是否检查 selection 中每个 asset 的 tip/non-tip link 序列完全一致。

    Returns:
        GmContactSensorLayout: 从 selected hand sidecar 推导出的 contact sensor 布局。
    """

    # 延迟导入 `HandSpawnAdapter`，使本模块的纯 sidecar 解析函数可在无 IsaacLab stub 的单测中导入。
    from anymani.robots.hand_spawn import HandSpawnAdapter

    adapter = HandSpawnAdapter(hand_spawn_cfg)  # runtime adapter；首次访问 selection 时解析 HandBank
    return build_contact_sensor_layout_from_assets(adapter.selection.assets, validate_all_assets=validate_all_assets)


def build_contact_sensor_layout_from_assets(
    assets: Iterable[Any],
    *,
    validate_all_assets: bool = False,
) -> GmContactSensorLayout:
    r"""从 resolved hand assets 构造 contact sensor layout。

    该函数只要求 asset 对象具有 `asset_id` 与 `sidecar` 两个属性，因此可直接用于纯单测。
    GM env 的生产路径由 `build_contact_sensor_layout_from_hand_spawn(...)` 包一层，把
    `HandSpawnCfg -> HandBank -> HandContainer` 的资产解析留给 `hand_spawn.py`。

    Args:
        assets (Iterable[Any]): resolved hand asset 序列，元素通常是 `HandContainer`。
        validate_all_assets (bool): 是否要求所有 asset 的 contact layout 与第一个 asset 完全一致。

    Returns:
        GmContactSensorLayout: 第一个 asset 的 contact sensor 布局。

    Raises:
        ValueError: 当 selection 为空，或 strict validation 发现 topology 不一致。
    """

    asset_list = tuple(assets)  # 固化 selection 顺序，便于既取首个又做可选全量校验
    if not asset_list:
        raise ValueError("Cannot build GM contact sensor layout from an empty hand selection.")

    # 第一项是默认布局真源；same-topology 训练默认信任 bank/schema 约束，不重复扫描所有 sidecar。
    first_asset = asset_list[0]  # `HandContainer` 或测试中的轻量 fake object
    first_layout = build_contact_sensor_layout_from_sidecar(
        getattr(first_asset, "sidecar"),
        asset_id=str(getattr(first_asset, "asset_id", "<unknown>")),
    )

    # 可选 strict 模式只比较 contact topology，不比较几何数值；post-mutate 几何变化不应触发失败。
    if validate_all_assets:
        first_signature = _layout_signature(first_layout)  # `(tip_links, non_tip_links)`，contact 语义签名
        for asset in asset_list[1:]:
            layout = build_contact_sensor_layout_from_sidecar(
                getattr(asset, "sidecar"),
                asset_id=str(getattr(asset, "asset_id", "<unknown>")),
            )
            if _layout_signature(layout) != first_signature:
                raise ValueError(
                    "Selected hand assets do not share the same GM contact layout: "
                    f"{first_layout.source_asset_id!r} has {first_signature}, "
                    f"but {layout.source_asset_id!r} has {_layout_signature(layout)}."
                )

    return first_layout


def build_contact_sensor_layout_from_sidecar(
    sidecar: Mapping[str, Any],
    *,
    asset_id: str = "<unknown>",
) -> GmContactSensorLayout:
    r"""从 `hand.yaml` sidecar 的 `hand_cfg` 字段解析 contact sensor layout。

    解析规则直接对应 `assets.asset_schema_embodiment.HandCfg` 的 joint-centric 表达：

    - palm/root link 来自 `hand_cfg.palm.name`；
    - fingertip link 来自每个 `joint.is_tip=True` 的 `joint.child`；
    - non-tip link 是 palm 加每个 `joint.is_tip=False` 的 `joint.child`；
    - sensor 名称稳定写作 `contact_<link_name>`，使 scene cfg / obs / reward 可共享同一字符串。

    Args:
        sidecar (Mapping[str, Any]): `HandContainer.sidecar` 或 YAML 解析后的 `dict`。
        asset_id (str): 当前 sidecar 所属 asset id，用于错误消息定位。

    Returns:
        GmContactSensorLayout: tip / non-tip link 与 sensor 名称布局。

    Raises:
        ValueError: 当 sidecar 缺少 `hand_cfg`、palm、fingers、joint child 或 fingertip 标记时抛出。
    """

    hand_cfg = _require_mapping(sidecar.get("hand_cfg"), f"asset {asset_id!r} sidecar['hand_cfg']")
    palm_cfg = _require_mapping(hand_cfg.get("palm"), f"asset {asset_id!r} hand_cfg['palm']")
    palm_link_name = _require_nonempty_string(palm_cfg.get("name"), f"asset {asset_id!r} palm.name")

    # 沿 sidecar finger/joint 顺序展开 child links；该顺序也就是当前 generated hand 的 semantic joint order。
    finger_link_chains = _finger_link_chains_from_hand_cfg(hand_cfg, asset_id=asset_id)  # 每根 finger 的 child link 链
    tip_links: list[str] = []  # `is_tip=True` 的 child links，服务 fingertip obs / good contact
    finger_non_tip_links: list[str] = []  # 只含 finger links；palm 是独立合法支撑角色
    for joint_cfg in _iter_joint_cfgs(hand_cfg, asset_id=asset_id):
        child_link = _require_nonempty_string(joint_cfg.get("child"), f"asset {asset_id!r} joint.child")
        if bool(joint_cfg.get("is_tip", False)):
            tip_links.append(child_link)  # 指尖 link：鼓励多指与 object 接触
        else:
            finger_non_tip_links.append(child_link)  # 非指尖 finger link：用于 bad-contact 与归因诊断

    tip_links = _dedupe_preserve_order(tip_links)  # 防御性去重；schema 本身也应保证 link 名唯一
    finger_non_tip_links = _dedupe_preserve_order(finger_non_tip_links)  # 纯 finger non-tip links，保持 sidecar 顺序
    if not tip_links:
        raise ValueError(f"asset {asset_id!r} hand_cfg does not mark any joint with is_tip=True.")

    fingertip_sensor_names = tuple(_sensor_name_for_link(link_name) for link_name in tip_links)
    finger_non_tip_sensor_names = tuple(_sensor_name_for_link(link_name) for link_name in finger_non_tip_links)

    return GmContactSensorLayout(
        source_asset_id=str(asset_id),
        palm_link_name=palm_link_name,
        finger_link_chains=finger_link_chains,
        fingertip_link_names=tuple(tip_links),
        finger_non_tip_link_names=tuple(finger_non_tip_links),
        fingertip_sensor_names=fingertip_sensor_names,
        finger_non_tip_sensor_names=finger_non_tip_sensor_names,
    )


def make_contact_sensor_cfg(
    link_name: str,
    *,
    robot_prim_path: str = "{ENV_REGEX_NS}/Robot",
    object_prim_path: str = "{ENV_REGEX_NS}/object",
    debug_vis: bool = False,
):
    r"""构造单个 hand link 对 object 的 Isaac Lab `ContactSensorCfg`。

    每个 sensor 只绑定一个 robot body prim：

    $$
    \texttt{prim\_path}=\texttt{\{ENV\_REGEX\_NS\}/Robot/<link>}
    $$

    并只过滤到被操作物体：

    $$
    \texttt{filter\_prim\_paths\_expr}=[\texttt{\{ENV\_REGEX\_NS\}/object}].
    $$

    这样 `force_matrix_w` / `friction_forces_w` 的中间 filter 维度只表达该 link 与 object
    的接触，避免手指间自碰污染 good/bad contact reward。

    Args:
        link_name (str): hand articulation 中的 link/body 名称。
        robot_prim_path (str): robot articulation root prim path，默认 `"{ENV_REGEX_NS}/Robot"`。
        object_prim_path (str): filtered object prim path，默认 `"{ENV_REGEX_NS}/object"`。
        debug_vis (bool): 是否打开 Isaac Lab contact sensor debug visualization。

    Returns:
        ContactSensorCfg: 可挂到 `InteractiveSceneCfg` 的 per-link contact sensor 配置。
    """

    # 延迟导入，让纯 sidecar 解析测试不需要安装 IsaacLab sensors module stub。
    from isaaclab.sensors import ContactSensorCfg

    return ContactSensorCfg(
        prim_path=f"{robot_prim_path}/{link_name}",
        filter_prim_paths_expr=[object_prim_path],
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=64,
        force_threshold=0.125,
        debug_vis=debug_vis,
    )


def install_contact_sensors(
    scene_cfg: Any,
    layout: GmContactSensorLayout,
    *,
    robot_prim_path: str = "{ENV_REGEX_NS}/Robot",
    object_prim_path: str = "{ENV_REGEX_NS}/object",
    debug_vis: bool = False,
    overwrite: bool = True,
) -> None:
    r"""把 layout 中的 per-link sensors 动态安装到 `InteractiveSceneCfg` 实例。

    Isaac Lab `InteractiveScene` 会遍历 scene cfg 的实例属性来发现 assets / sensors。
    因此这里使用 `setattr(scene_cfg, sensor_name, ContactSensorCfg(...))`，而不是在
    `GmInHandSceneCfg` class body 中硬编码一组 topology-specific `contact_<finger>_tip = ...` 字段。

    Args:
        scene_cfg (Any): `InteractiveSceneCfg` 实例。
        layout (GmContactSensorLayout): 从 hand sidecar 推导出的 contact layout。
        robot_prim_path (str): robot articulation root prim path。
        object_prim_path (str): filtered object prim path。
        debug_vis (bool): 是否打开 contact sensor debug visualization。
        overwrite (bool): 重复安装时是否覆盖同名属性；默认 True 使 `__post_init__` 幂等。

    Raises:
        AttributeError: 当 `overwrite=False` 且 scene 已有同名字段时抛出。
    """

    # 每个 link 一个 sensor，严格保留 object filter 的 single-body filtered contact 语义。
    for sensor_name, link_name in layout.sensor_link_pairs:
        if not overwrite and hasattr(scene_cfg, sensor_name):
            raise AttributeError(f"Scene cfg already has a contact sensor field named {sensor_name!r}.")
        setattr(
            scene_cfg,
            sensor_name,
            make_contact_sensor_cfg(
                link_name,
                robot_prim_path=robot_prim_path,
                object_prim_path=object_prim_path,
                debug_vis=debug_vis,
            ),
        )


def sensor_total_force_w(env: Any, sensor_name: str) -> torch.Tensor:
    r"""读取单个 `ContactSensor` 对 object 的总接触力。

    Args:
        env (Any): Isaac Lab manager-based RL env，需提供 `env.scene[sensor_name]`。
        sensor_name (str): scene 中 ContactSensor 的名称。

    Returns:
        torch.Tensor: 总接触力，形状 `[num_envs, 3]`，单位 N，世界系 `{w}` 表达。
    """

    total_force_w = _sensor_force_tensor_w(env, sensor_name)  # `[B,...,3]`，normal + friction，NaN 已置零
    while total_force_w.ndim > 2:
        total_force_w = total_force_w.sum(dim=1)  # `[B,...,3] -> [B,3]`，合并 body/filter 维度
    return total_force_w  # `[B,3]`，该 sensor 对 object 的总接触力


def sensor_contact_indicator(env: Any, sensor_name: str, force_threshold: float) -> torch.Tensor:
    r"""判断单个 `ContactSensor` 是否发生超过阈值的 object 接触。

    与 `sensor_total_force_w(...)` 不同，本函数对 body/filter pair 的力幅值取最大，避免多个
    接触点方向相反时向量求和相互抵消。该语义更适合 good/bad contact 的二值判定：只要任意
    contact pair 的 $\|F\|_2$ 超过阈值，就认为该 link 与 object 有效接触。

    Args:
        env (Any): Isaac Lab manager-based RL env。
        sensor_name (str): scene 中 ContactSensor 的名称。
        force_threshold (float): 接触判定阈值，单位 N。

    Returns:
        torch.Tensor: bool tensor，形状 `[num_envs]`。
    """

    return sensor_contact_magnitude(env, sensor_name) > float(force_threshold)  # `[B]`，二值有效接触指示


def sensor_contact_magnitude(env: Any, sensor_name: str) -> torch.Tensor:
    r"""读取单个 sensor 内最大的 body/filter-pair 接触力幅值。

    二值触觉不能先把多个接触向量相加：若两个接触法向相反，向量和可能接近零，但两个
    物理接触都真实存在。这里先对每个 pair 计算 $\|F\|_2$，再对非 batch 维取最大值。

    Args:
        env (Any): Isaac Lab manager-based env。
        sensor_name (str): scene 中 object-filtered ContactSensor 名称。

    Returns:
        torch.Tensor: 每个 env 的最大 pair 力幅值，形状 `[num_envs]`，单位 N。
    """

    total_force_w = _sensor_force_tensor_w(env, sensor_name)  # `[B,...,3]`，normal + tangential force
    force_norm = torch.linalg.norm(total_force_w, dim=-1)  # `[B,...]`，逐 body/filter pair 的 $\|F\|_2$
    if force_norm.ndim > 1:
        force_norm = force_norm.amax(dim=tuple(range(1, force_norm.ndim)))  # `[B]`，不允许 pair 间方向抵消
    return force_norm  # `[B]`，单位 N


def _sensor_force_tensor_w(env: Any, sensor_name: str) -> torch.Tensor:
    r"""读取 ContactSensor force tensor，并把 normal/friction 统一成世界系接触力。"""

    sensor = env.scene[sensor_name]  # ContactSensor；由 scene cfg 显式声明 prim_path/filter
    force_w = getattr(sensor.data, "force_matrix_w", None)  # `[B,body,filter,3]`，object-filtered normal force
    if force_w is None:
        force_w = getattr(sensor.data, "net_forces_w", None)  # `[B,body,3]`，fallback：未过滤 normal force
    if force_w is None:
        raise RuntimeError(f"Contact sensor {sensor_name!r} does not expose force data.")

    total_force_w = torch.nan_to_num(force_w, nan=0.0)  # 无接触 / NaN filtered pair 视为 0N normal force
    friction_w = getattr(sensor.data, "friction_forces_w", None)  # `[B,body,filter,3]`，切向摩擦力
    if friction_w is not None:
        total_force_w = total_force_w + torch.nan_to_num(friction_w, nan=0.0)  # normal + tangential contact force
    return total_force_w  # `[B,...,3]`，世界系接触力张量


def _iter_joint_cfgs(hand_cfg: Mapping[str, Any], *, asset_id: str) -> Iterable[Mapping[str, Any]]:
    r"""按 sidecar 中的 finger/joint 顺序迭代所有 joint cfg。"""

    for _, _, joint_cfg in _iter_finger_joint_cfgs(hand_cfg, asset_id=asset_id):
        yield joint_cfg


def _iter_finger_joint_cfgs(
    hand_cfg: Mapping[str, Any],
    *,
    asset_id: str,
) -> Iterable[tuple[int, Mapping[str, Any], Mapping[str, Any]]]:
    r"""按 sidecar 中的 finger/joint 顺序迭代 `(finger_index, finger_cfg, joint_cfg)`。"""

    fingers = hand_cfg.get("fingers")  # `HandCfg.fingers`，应为 list[dict]
    if not isinstance(fingers, Sequence) or isinstance(fingers, str):
        raise ValueError(f"asset {asset_id!r} hand_cfg['fingers'] must be a sequence of finger mappings.")

    for finger_index, finger_cfg in enumerate(fingers):
        finger_mapping = _require_mapping(finger_cfg, f"asset {asset_id!r} finger[{finger_index}]")
        joints = finger_mapping.get("joints")  # `FingerCfg.joints`，应为 list[dict]
        if not isinstance(joints, Sequence) or isinstance(joints, str):
            raise ValueError(f"asset {asset_id!r} finger[{finger_index}]['joints'] must be a sequence.")
        for joint_index, joint_cfg in enumerate(joints):
            yield (
                finger_index,
                finger_mapping,
                _require_mapping(joint_cfg, f"asset {asset_id!r} finger[{finger_index}].joints[{joint_index}]"),
            )


def _finger_link_chains_from_hand_cfg(
    hand_cfg: Mapping[str, Any],
    *,
    asset_id: str,
) -> tuple[tuple[str, ...], ...]:
    r"""从 sidecar 解析每根 finger 的 child link 链。

    Returns:
        tuple[tuple[str, ...], ...]: 外层按 finger 顺序，内层按 joint 顺序。
    """

    chains_by_finger: dict[int, list[str]] = {}  # finger index -> child link 链，保持 sidecar 顺序
    for finger_index, _, joint_cfg in _iter_finger_joint_cfgs(hand_cfg, asset_id=asset_id):
        child_link = _require_nonempty_string(joint_cfg.get("child"), f"asset {asset_id!r} joint.child")
        chains_by_finger.setdefault(finger_index, []).append(child_link)  # 同一 finger 内部链，用于 same-finger filter

    finger_link_chains = tuple(
        tuple(_dedupe_preserve_order(chains_by_finger[index])) for index in sorted(chains_by_finger)
    )  # 防御性去重，但保留每根 finger 的顺序
    if not finger_link_chains:
        raise ValueError(f"asset {asset_id!r} hand_cfg does not contain any finger link chain.")
    return finger_link_chains


def _sensor_name_for_link(link_name: str) -> str:
    r"""把 hand link 名稳定映射为 scene sensor 字段名。"""

    return f"contact_{link_name}"  # link 名来自 asset schema sanitize，适合作为 Python config 字段名


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    r"""校验 YAML 节点为 mapping，并保留清晰上下文错误消息。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping, got {type(value).__name__}.")
    return value


def _require_nonempty_string(value: Any, context: str) -> str:
    r"""校验 YAML 字段为非空字符串。"""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string, got {value!r}.")
    return value


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    r"""对 link 名按首次出现顺序去重，防御异常 sidecar 重复项。"""

    seen: set[str] = set()  # 已输出 link 名集合
    deduped: list[str] = []  # 保持输入顺序的去重结果
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _layout_signature(
    layout: GmContactSensorLayout,
) -> tuple[tuple[tuple[str, ...], ...], str, tuple[str, ...], tuple[str, ...]]:
    r"""提取用于 same-topology validation 的 contact / structural-collision 语义签名。"""

    return (
        layout.finger_link_chains,
        layout.palm_link_name,
        layout.fingertip_link_names,
        layout.finger_non_tip_link_names,
    )


__all__ = [
    "GmContactSensorLayout",
    "build_contact_sensor_layout_from_assets",
    "build_contact_sensor_layout_from_hand_spawn",
    "build_contact_sensor_layout_from_sidecar",
    "install_contact_sensors",
    "make_contact_sensor_cfg",
    "sensor_contact_magnitude",
    "sensor_contact_indicator",
    "sensor_total_force_w",
]
