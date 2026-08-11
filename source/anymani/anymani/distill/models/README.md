# Shared Models

`models` 保存 SSL、RL 与 IL 共用的可学习组件。它的目标不是为每一种 geometry target 建一套独立 policy，
而是让不同 target 训练同一条可部署表示路径。

## Retained boundary

```text
X = route-valid deployable geometry evidence
    -> route-specific input adapter + retained backbone
        -> Z^(0) = entity zeroth-order latents
        -> Z^(1) = actuated-joint first-order latents
            -> action / value / history modules    # RL deployment side
            -> representation decoder              # pretraining only by default
```

**input adapter + backbone + $z^{(0)}/z^{(1)}$ heads** 共同构成 retained geometry encoder，也是 SSL
checkpoint 迁入 PPO 的默认边界。$Z^{(0)}=\{z_i^{(0)}\}$ 与 physical entity 对齐，$Z^{(1)}=\{z_j^{(1)}\}$
与 actuated joint 对齐；二者不是把一个 latent 任意切成两半，而是分别承载 configuration-level state 与
joint-sensitive local differential state。

更换 BPS、SDF、density 或 Gaussian-field target，不应顺便更换 policy trunk，否则无法判断收益来自 target
还是模型容量。

## 目录

| 路径 | 职责 | 生命周期 |
| --- | --- | --- |
| `input_adapters/` | 把当前 $q$ 与静态手型证据投影为 owner-aware entity/joint features | retained |
| `backbones/` | 处理同结构组 PALM/JOINT/TIP entity sequence 并输出 latent | retained |
| `decoders/representations/` | 从 latent 重建 field、FK 或 Gaussian-induced field | 默认 training-only |
| `heads/` | JOINT action、value 与非 field-specific auxiliary output | retained 或按 stage 选择 |
| `policy.py` | stage-independent model assembly contract | executable draft |
| `tokens.py`、`config.py` | 已有 grouped-token/config draft | executable draft，API 尚未冻结 |
| `temporal_encoder.py` | single-asset tactile TCN baseline | 可运行 baseline |
| `relations.py`、`attention_bias.py` | relation 与 dynamic bias 候选 | deferred，不是公共默认 |

`backbones/candidates/spatial_transformer.py` 只是候选位置。Transformer、GRU、TCN、时空顺序、pooling 和
attention bias 均未被选为跨实验默认方案。

## Retained input 与 latent contract

当前隐式主线只读取当前 $q$ 与静态手型证据：基准 screw、$q_{home}$、topology、按 PALM/JOINT/TIP
归属组织的基准表面采样点、palm normal 与 physical anchors。解析直接压缩仅作保留候选；若后续激活，
可读取缓存支撑点经当前 FK/刚体位姿得到的 physical points。current distance、最近点、surface Jacobian
或场标签不得进入 retained encoder。home geometry 的磁盘读取、CPU 解析和 GPU materialization 可以缓存；
PPO full fine-tune 时，依赖 learnable weights 的 activation 不能永久缓存。

contact、command、history 与 object state 可以由下游 policy 模块消费，但不属于当前 geometry SSL 的
$X\rightarrow(Z^{(0)},Z^{(1)})$ 命题。current $\rho/\kappa/g$ label、current posed surface 与未来状态均不得进入
retained encoder。

当前共同 owner 角色是 `PALM`、`JOINT` 与 `TIP`；第 $g$ 个实体、表面归属体与 SSL 解码轴直接同索引：

- 网络原生支持可变 $N_E/N_J/K$；跨结构同一次前向可填充到 20 JOINT、5 TIP、26 owner，并以 entity/joint masks 严格屏蔽无效槽；逐结构独立前向保留为数值与梯度 oracle；
- owner、asset routing 与 action routing 来自 metadata，而不是固定 tensor slice；
- 只有 `JOINT` token 直接产生 joint action；
- `PALM` 与 `TIP` 参加整手上下文，但不输出动作；
- current target field、future state 与其他仅训练期 privileged label 不得进入 retained adapter。

## 位姿与方向输入

每组 pose feature 都应说明它是绝对姿态、相对变换、局部误差还是移动 reference 下的量。几何 composition
先在 $SE(3)$ 中定义，再选择 matrix、rot6d、quaternion 或 local $se(3)$ coordinates 作为网络编码。

- quaternion 必须注明 `(w,x,y,z)`、归一化与 $q\sim -q$ 的符号策略；
- Euler/RPY 不作为匿名 feature；
- 隐藏且移动的 goal/reference 会制造部分可观测，不应靠换一种旋转编码掩盖；
- 当前 teacher action 是 joint-space scalar，不因输入含 orientation 就自动变成 Cartesian action。

hand frame `{h}` 只固定有向 palm normal $n_p=z_h$；绕 $n_p$ 的 $x/y$ 选择是 gauge，而不是可依赖的类人手方向语义。
$Z^{(0)}$ 与 $z_i^{(1)}$ 都对这一 $SO(2)$ 重写不变；在 joint-sign 成对改写下，$Z^{(0)}$ 为偶，
$z_i^{(1)}$ 对自身 $s_i$ 为奇。reflection/chirality、link-local URDF reparameterization 与 hand-frame gauge
必须分别测试；文献证据由独立 research dossier 维护，不作为源码运行依赖。

## Checkpoint lifecycle

默认迁移：

```text
SSL checkpoint
    -> load input adapter + backbone + z0/z1 heads
    -> drop representation decoder
    -> attach history encoder + action/value heads
    -> PPO full fine-tune
```

冻结、部分冻结、保留 decoder 或 joint auxiliary 都必须成为显式消融，而不是 checkpoint loader 的隐藏行为。
当前硬门槛是在 RTX 5070 Ti、$B=4096$、单结构组下，隐式主线完整在线 retained geometry encoder
50 次计时 p95 不超过 40 ms，计时从 GPU-resident $q$ 与静态证据开始。未来若激活解析直接候选，
同一门槛还必须计入批量 FK/刚体支撑点变换。离线 cache materialization、decoder、policy、
host-to-device copy 与 Isaac Sim 不计入。

## 最小验证

- 同结构组的 entity/joint/anchor shape、owner、type id 与标准顺序；
- batch permutation 与归属体/锚点集合聚合合同；
- paired $SO(2)$ gauge invariance、axis sign/zero rewrite 与 scale law；
- $Z^{(0)}$ entity routing、$Z^{(1)}$ joint routing 与 decoder query contract；
- JOINT-only action routing；
- decoder 不进入 PPO forward；
- checkpoint key 与 retained/discarded parameter 集合；
- temporal history 的顺序、长度、episode reset 与 causal contract；
- 真实性能命题再使用 rollout smoke 或短训练，而不是只看单元测试。
