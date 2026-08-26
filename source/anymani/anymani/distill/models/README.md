# Geometry Representation Model

跨手型模型面临两个同时存在的变化：entity 集合随手指与链长改变，关节坐标又携带方向和局部运动语义。把整只手压成一个全局向量会丢失 owner/joint 对齐；为每种手型建立独立网络则无法区分表示迁移与参数记忆。当前模型使用与 PALM/JOINT/TIP owner 同索引的统一 token 序列，让所有手型共享前端、上下文主干和读取规则。

## Retained 表示

对一项手型 $\mathfrak m$，encoder 接收当前物理关节角 q 与静态 evidence：$q_{home}$、ordered space screws、运动学图、PALM/JOINT/TIP home boundary、palm normal 和完整 physical anchor 集合。输出直接取 graph-biased Transformer final-norm tokens：

$$
Z\in\mathbb R^{B\times G\times128}.
$$

$Z_g$ 与第 $g$ 个 surface owner 同索引。逐 JOINT 消费者通过 `joint_entity_index` 从同一 $Z$ gather；没有第二 latent、zero/first-order heads 或主干后的 raw-screw bypass。PALM 与 TIP 参与整手上下文但不伪造独立关节轴。

current distance、closest point、surface Jacobian、query stratum、contact、object state 与 teacher field 都不属于输入。joint limits 只定义 q sampling domain；两个只有限位不同而物理映射相同的资产，在相同 q 下必须得到相同表示。

## 多锚点关系前端

hand frame `{h}` 只固定有向 palm normal；origin 与面内 x/y basis 是 gauge。网络不读取 raw hand-frame 坐标作为绝对方向标签，而是把 home points、queries 和空间旋量表达为相对完整 anchor constellation 的关系。每个 point-anchor pair 使用面内径向、切向、height 与 handed scalar 等 $SO(2)$-compatible quantities，共享 MLP 后沿无序 anchor 集合聚合。

home surface 先沿 anchors 编码，再在每个 owner 的真实 boundary points 上做 masked attention pooling。对 JOINT，空间旋量 $[\omega_i,v_i]$ 被转换为 axis point 与有向 screw-anchor relations，形成唯一的 $f_i^{screw}$。该 feature 与当前 $q_i$ 只在 Transformer 前进入 JOINT entity token，不建立第二网络分支，也不输入裸 sign bit。

## Graph-biased 整手上下文

owner tokens 进入 encoder-only、Pre-LN、全连接 self-attention。运动学图不作为 hard mask；任意两个有效 entities 都能交换信息。图结构只通过每个 attention head 的加性 bias 注入：

$$
b_{ij}^{(h)}
=b_{shortest}^{(h)}(d_{ij})
+b_{parent}^{(h)}(d_{ij}^{p})
+b_{child}^{(h)}(d_{ij}^{c}).
$$

三种距离分别查表并相加，超过 8 的距离进入末桶。canonical backbone 使用 hidden width 128、2 layers、4 heads、FFN width 256、dropout 0。它不读取 current dynamic all-pairs $SE(3)$ pose answer；当前构型变化必须通过 q 与 screw-conditioned entity features进入模型。

跨结构 batch 可 padding 到最多 20 JOINT、5 TIP、26 owners。entity mask 同时屏蔽 key/value、query row、projection bias 与最终输出；joint mask 屏蔽运动输入与 selector。逐结构独立前向是 padded valid outputs 和参数梯度的数值 oracle。

## JOINT view 与坐标改写

第 i 个 JOINT 的 view 只是统一 $Z$ 的索引操作：

$$
z_i=Z_{\operatorname{joint\_entity\_index}(i)}.
$$

joint-axis sign 改写同步作用于 q、$q_{home}$ 与 space screw，但不约束 latent 本身的 parity；只在完整模型输出上检查 density 不变与对应 κ 变号。reflection/chirality 不是同一 gauge，模型必须允许镜像形态得到不同表示。

## SSL-only readers

密度 reader 对每个 `(owner, query, sigma)` 输出一个 scalar，因此 query 数与 sigma 数都是可变数据轴，不是固定输出通道。query feature 和 $\log(\sigma/16\,\mathrm{mm})$ 进入 hidden width 128 的主路径；三个 residual blocks 的每一层都由同一 owner 的 $z_g$ 通过 FiLM 调制：

$$
\widetilde h=(1+\gamma(z_g))\operatorname{LN}(h)+\beta(z_g).
$$

sensitivity reader 将 query 投影到 width 128，并在三个独立 residual blocks 中由 $[z_o\Vert z_i]$ 调制：

$$
h_{l+1}=h_l+F_l((1+\gamma_l[z_o,z_i])\operatorname{LN}(h_l)+\beta_l[z_o,z_i]),\qquad
\hat\kappa=w^Th_3+b.
$$

输出不使用 sigmoid/tanh，允许任意 signed scalar；reader 不读取 sigma、raw screw 或 task one-hot。两个 readers 只在 SSL 中存在，不进入 retained checkpoint 或部署延迟门槛。

## 生命周期与性能边界

SSL checkpoint 的迁移边界是 input adapter 与 graph-biased backbone。schema-5 PPO/IL loader 只接受 `encoder.` namespace 和 unified feature spec，并严格报告 missing/unexpected keys；density/sensitivity readers、Warp teacher 与 objective 全部删除。若 full fine-tune encoder，依赖 learned weights 的 activation 不得跨 policy steps 永久缓存。

canonical 完整模型有 830,537 个参数，其中 retained encoder 317,383 个、SSL-only readers 513,154 个；state-dict key 集合由 contract digest 固定。RTX 5070 Ti 上，retained encoder 在 $B=4096$、单结构、20 次预热和 50 次 CUDA Event 下测得 median 17.844 ms、p95 18.275 ms、max 18.472 ms，peak allocated memory 811.42 MiB。正式 $B=4,G=21,N_Q=64,N_\sigma=3,E=42$ 的 disposable readers forward 另测得 median 1.324 ms、p95 1.461 ms、peak 61.41 MiB。

40 ms 门槛只从 GPU-resident q/static evidence 计到统一 $Z$，排除 source materialization、H2D、decoder、policy、Isaac Sim 与 `env.step`。因此它是 20 Hz 控制周期中的 retained 子预算，不是完整 PPO/env latency 结论。

## 证据与未决边界

contracts 已检查三种 graph-bias、anchor 与 entity permutation、$SO(2)$ rewrite、unified routing、variable sigma、双 reader FiLM 依赖、observable sign probe、density JVP、跨结构 output/gradient 和 retained/disposable keys。真实 LEAP integration 已完成 encoder、readers、双 objective 与普通参数 backward。

这些证据尚未选择最优容量；2 layers、unified width 128 和 reader width 128 只是 canonical 可执行锚点。任何容量候选都必须在相同 q/query/target 与训练预算下比较，不能通过改变物理监督或缓存 stale learned activation 获得表面加速。正式 256-epoch SSL、PPO transfer、cross-topology/cross-DOF 与 official hands 仍待实验。
