# Geometry Representation Model

跨手型模型面临两个同时存在的变化：entity 集合随手指与链长改变，关节坐标又携带方向和局部运动语义。把整只手压成一个全局向量会丢失 owner/joint 对齐；为每种手型建立独立网络则无法区分表示迁移与参数记忆。当前模型因此把整手状态拆成逐物理 owner 的零阶包与逐活动关节的一阶包，并让所有手型共享前端、上下文主干和读取规则。

## Retained 表示

对一项手型 $\mathfrak m$，encoder 接收当前物理关节角 q 与静态 evidence：$q_{home}$、ordered space screws、运动学图、PALM/JOINT/TIP home boundary、palm normal 和完整 physical anchor 集合。输出为

$$
Z^{(0)}\in\mathbb R^{B\times G\times D_0},
\qquad
Z^{(1)}\in\mathbb R^{B\times N_J\times D_1},
$$

其中 canonical $D_0=128$、$D_1=64$。$Z^{(0)}_g$ 与第 $g$ 个 surface owner 同索引，表达当前构型下该实体的 configuration-level geometry；$z_i^{(1)}$ 与第 $i$ 个活动 JOINT 同索引，提供 joint-sensitive differential carrier。PALM 与 TIP 参与整手上下文但不伪造独立关节轴。

current distance、closest point、surface Jacobian、query stratum、contact、object state 与 teacher field 都不属于输入。joint limits 只定义 q sampling domain；两个只有限位不同而物理映射相同的资产，在相同 q 下必须得到相同表示。

## 多锚点关系前端

hand frame `{h}` 只固定有向 palm normal；origin 与面内 x/y basis 是 gauge。网络不读取 raw hand-frame 坐标作为绝对方向标签，而是把 home points、queries 和空间旋量表达为相对完整 anchor constellation 的关系。每个 point-anchor pair 使用面内径向、切向、height 与 handed scalar 等 $SO(2)$-compatible quantities，共享 MLP 后沿无序 anchor 集合聚合。

home surface 先沿 anchors 编码，再在每个 owner 的真实 boundary points 上做 masked attention pooling。对 JOINT，空间旋量 $[\omega_i,v_i]$ 被转换为 axis point 与有向 screw-anchor relations，形成唯一的 $f_i^{screw}$。该 feature 同时进入 JOINT entity token 和一阶 residual head，不建立 `screw_even/screw_odd` 两套网络，也不输入裸 sign bit。

## Graph-biased 整手上下文

owner tokens 进入 encoder-only、Pre-LN、全连接 self-attention。运动学图不作为 hard mask；任意两个有效 entities 都能交换信息。图结构只通过每个 attention head 的加性 bias 注入：

$$
b_{ij}^{(h)}
=b_{shortest}^{(h)}(d_{ij})
+b_{parent}^{(h)}(d_{ij}^{p})
+b_{child}^{(h)}(d_{ij}^{c}).
$$

三种距离分别查表并相加，超过 8 的距离进入末桶。canonical backbone 使用 hidden width 128、2 layers、4 heads、FFN width 256、dropout 0。它不读取 current dynamic all-pairs $SE(3)$ pose answer；当前构型变化必须通过 q 与 screw-conditioned entity features进入模型。

跨结构 batch 可 padding 到最多 20 JOINT、5 TIP、26 owners。entity mask 同时屏蔽 key/value、query row、projection bias 与最终输出；joint mask 屏蔽运动与一阶 head。逐结构独立前向是 padded valid outputs 和参数梯度的数值 oracle。

## 类型化一阶 head

整手主干先产生 owner zero-order latent。第 i 个 JOINT 的一阶包由共享 residual head 构造：

$$
z_i^{(1)}=H_1\!\left([z_i^{(0)}\Vert f_i^{screw}]\right).
$$

这里不把 parity 写死进普通 MLP，而是用成对坐标改写监督它：joint-axis sign 改写同步作用于 q、$q_{home}$ 与 space screw 后，$Z^{(0)}$ 应保持偶，$z_i^{(1)}$ 应按对应 sign 变为奇。reflection/chirality 不是同一 gauge，模型必须允许镜像形态得到不同零阶表示。

## SSL-only readers

密度 reader 对每个 `(owner, query, sigma)` 输出一个 scalar，因此 query 数与 sigma 数都是可变数据轴，不是固定输出通道。query feature 和 $\log(\sigma/16\,\mathrm{mm})$ 进入 hidden width 128 的主路径；三个 residual blocks 的每一层都由同一 owner 的 $z_g^{(0)}$ 通过 FiLM 调制：

$$
\widetilde h=(1+\gamma(z_g^{(0)}))\operatorname{LN}(h)+\beta(z_g^{(0)}).
$$

sensitivity reader 从 $z_g^{(0)}$ 与 query 产生偶 coefficient，再与 $z_i^{(1)}$ 作无偏置内积：

$$
\hat\kappa_{g,i}=\frac{a(z_g^{(0)},u(x))^Tz_i^{(1)}}{\sqrt{D_1}}.
$$

无偏置读取使 joint-sign parity 成为结构合同。两个 readers 只在 SSL 中存在；它们不进入 retained checkpoint，也不计入部署延迟门槛。

## 生命周期与性能边界

SSL checkpoint 的迁移边界是 input adapter、graph-biased backbone 和 zero/first-order heads。PPO/IL loader 只接受 `encoder.` namespace，并严格报告 missing/unexpected keys；density/sensitivity readers、Warp teacher 与 objective 全部删除。若 full fine-tune encoder，依赖 learned weights 的 activation 不得跨 policy steps 永久缓存。

canonical 完整模型有 590856 个参数，其中 retained encoder 350407 个、SSL-only readers 240449 个；state-dict key 集合由 contract digest 固定。RTX 5070 Ti 上，retained encoder 在 $B=4096$、单结构、20 次预热和 50 次 CUDA Event 下测得 median 19.79 ms、p95 20.13 ms、max 20.22 ms，peak allocated memory 843.55 MiB。正式 $B=4,G=21,N_Q=64,N_\sigma=3,E=42$ 的 disposable readers forward 另测得 median 1.22 ms、p95 1.37 ms、peak 60.51 MiB。

40 ms 门槛只从 GPU-resident q/static evidence 计到 $Z^{(0)}/Z^{(1)}$，排除 source materialization、H2D、decoder、policy、Isaac Sim 与 `env.step`。因此它是 20 Hz 控制周期中的 retained 子预算，不是完整 PPO/env latency 结论。

## 证据与未决边界

contracts 已检查三种 graph-bias 的精确 lookup 与相加、末距离桶仍参与全连接 attention、anchor permutation、$SO(2)$ rewrite、joint-sign parity、variable sigma、sigma detach、每层 FiLM、跨结构 output/gradient 和 retained/disposable keys。真实 LEAP integration 已完成 encoder、readers、五项 method objective 与 backward。

这些证据尚未选择最优容量；2 layers、128/64 latent 和 FiLM width 128 只是 canonical 可执行锚点。任何缩小网络以满足吞吐的候选都必须在相同 q/query/target 与训练预算下比较，不能通过改变物理监督或缓存 stale learned activation 获得表面加速。正式 PPO transfer、cross-topology/cross-DOF 与 official hands 仍待实验。
