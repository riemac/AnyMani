# Coupled Pregrasp Synthesis for Heterogeneous Hands

## Motivation

跨手型共享控制要求策略面对不同handedness、手指数、关节数、指长、安装位置与掌面几何。若所有资产复用一个world-frame物体位置，训练难度会被初始穿透、悬空、完全张手或控制器预载瞬态主导，策略首先学习的是补救错误reset，而非手内操作。Pregrasp合成的目标是为每只手提供物理安静、几何可操作且可审计的共同起点，同时不利用后续强化学习表现筛选“容易旋转”的手。

## Coupled Initial-State Object

对hand asset $h_i$、object identity $o$与absolute scale $s$，schema-3将初态定义为耦合集合：

$$
(h_i,o,s)\longmapsto\mathcal B_{i,o,s}=\left\{\left(q_0^{(k)},u_0^{(k)},T_{ho,0}^{(k)}\right)\right\}_{k=0}^{7}.
$$

$q_0^{(k)},u_0^{(k)}\in\mathbb R^{n_i}$分别表示实际关节状态与PD位置目标，单位rad；实现以canonical `[16]` tensor和active mask表示不同$n_i$，inactive slots严格为零。$T_{ho,0}^{(k)}\in SE(3)$表示object在hand semantic frame中的位姿，translation单位m。三者共同决定接触与控制器势能，因而不能把独立的$q$列表与物体位置列表做笛卡尔积。当前MVP固定DexCube scale 1.1、upright identity quaternion、$u_0=q_0$、零joint/object velocity，并保存Top-8；训练只消费rank-0。

Catalog key绑定source content、physical geometry、canonical schema、active routing、DexCube bytes、absolute scale、physics configuration与generation algorithm的digest。Dataset row只承担provenance和分层统计。`GoodPregraspCatalog`使用canonical JSON、content-addressed payload、排序index和同文件系统原子replace；同一exact key不能映射到另一组Top-8。

## Strict Feasible Set

Strict v5把候选$c$的可行域写成九个同时成立的条件：

$$
\mathcal F=\left\{c:\ m_q\ge0.10,\ d_{tip}^{max}\le0.10,\ \alpha_{sector}^{min}\ge30^\circ,\ \delta_{pen}^{max}\le0.5\text{ mm},\ \Delta p^{max}\le5\text{ mm},\ \theta_{tilt}^{max}\le10^\circ,\ v_{0:0.2}^{max}\le0.25\text{ m/s},\ \omega_{0:0.2}^{max}\le2\text{ rad/s},\ \rho_{palm}^{tail}\ge0.5\right\}.
$$

$m_q$是active joints到最近limit的最小归一化余量；$d_{tip}^{max}$和$\alpha_{sector}^{min}$由thumb与两根active non-thumb的联合包络计算。其余量来自$q_0=u_0$、object静止写入后1 s、120 Hz的cold-reset replay；contact以20 Hz、EMA系数0.5和0.25 N阈值统计，$\rho_{palm}^{tail}$覆盖最后0.5 s。峰值角速度使用hand-frame总模$\|\omega_h\|_2$：zero-action reset若绕目标轴自发旋转，同样会向策略注入未请求的运动。PALM、JOINT与TIP contact fractions完整保存为metadata，TIP接触数量不定义准入等级。

## Geometry Proposal

每个资产首先生成256个13维scrambled Sobol提案。一个latent控制N000 role-wise template与joint-limit midpoint的blend，四个depth synergies与四个finger synergies形成低维关节扰动；其余变量控制opposition-center mix、hand-frame $x/y$位置与掌面clearance。所有active joints在proposal阶段保留11% range margin，为10%硬门留出数值余量。

候选$q$通过真实articulation FK得到四个TIP origins。三个non-thumb pairs分别与thumb组成联合包络，object center由opposition geometry和掌内anchor共同提出。Cheap screen使用6–10 cm TIP-center clearance band与30° sector优先级：6 cm下界只是减少明显穿指提案的搜索代理，最终准入仍由真实PhysX penetration决定。每资产只有geometry score最高的32项进入首次完整1 s物理筛选，因此初始并行scene为$80\times32=2560$ environments。

## Physics-Guided Low-Rank CEM

若某资产不足8个严格候选，搜索最多追加三轮，每轮提出并物理验证128项。对已测physical elites的active-joint states做PCA，保留前四个协同方向$V_i\in\mathbb R^{16\times4}$，再与三维object position组成7维局部分布：

$$
q=q_e+V_i\epsilon_q,\qquad p_{ho}=p_e+\epsilon_p.
$$

单一均值会把不同接触模态平均成不稳定构型，因此v5使用按elite质量分配样本的mixture CEM。PhysX packed contact buffer同时返回separation与normal；对最深初始重叠，下一轮position center沿负contact normal移动$1.10\delta_{pen}+0.25$ mm。对已无初始穿透、但自由落距偏大的候选，center按相对4.5 mm目标的超量向掌面回移。若候选已严格通过或normalized gate violation不超过0.35，下一轮最多96项围绕该模态做微扰，以估计并扩展狭窄稳定盆。上述反馈只改变proposal分布，每个发布成员仍重新满足同一$\mathcal F$，没有asset-specific参数或门限调整。

## Pair-Aware Cohort Construction

代表性manifest以left/right morphology pair为选择原子。每个handedness-neutral `(tip count, thumb DoF)` cell预排序32组候选pair，目标配额为前10组拥有完整strict Top-8的pairs。若一侧在固定预算内不足8项，整组退出并记录双方row、rank与失败证据，再取下一组；通过侧不会单独留下。Fallback物理筛选可以在后续batch完成，`assemble_heterogeneous_mvp80_strict_catalog.py`只组合generation、physics和strict-gate digests完全相同的证据，随后重新选择40组pairs并对最终640个members逐项重放门限。

## Current MVP80 Evidence

当前active artifact位于：

| Artifact | Path / SHA-256 |
| --- | --- |
| 80-row manifest | `assets/datasets/cross_embodiment_balanced_v1/ppo_mvp80.yaml` / `5f6af6db9e823cd6cc0b0f9c822446c59da167db901a09d5e8b5234f4f9eb707` |
| strict Top-8 catalog index | `outputs/pregrasp/catalogs/heterogeneous_rotation_mvp80_dexcube_s1p1_v5/index.json` / `ee5f2cee135ef1cde55e2bfd6c160c03e3179a8ae75624c5d50f4fdac9765f8c` |
| final assembly summary | `outputs/pregrasp/search/heterogeneous_rotation_mvp80_dexcube_s1p1_v5/final-summary.json` / `cb741e65472a9f26e62503f649521adeee131b7fa778470576a3ad503abbdc8d` |
| 5-page visual evidence | `outputs/pregrasp/visual/heterogeneous_rotation_mvp80_dexcube_s1p1_v5/visual-evidence.json` / `990f2411e8e7ffb7bc95cdfadb9968ddb628f034f38717fb18a83b741dabf1a7` |

最终catalog包含80个entries和640个strict members，并跳过9组预排序pairs。Rank-0跨资产边界为：minimum joint margin `0.1100`、maximum TIP-center distance `0.09819 m`、minimum sector `30.338°`、maximum penetration `0.000498 m`、maximum displacement `0.004982 m`、maximum tilt `6.601°`、maximum linear speed `0.24525 m/s`、maximum total angular speed `1.97891 rad/s`、minimum tail PALM fraction `1.0`。最终80项在正式task reset路径执行20个zero-action policy steps后为80/80存活，未触发drop、axis failure或timeout。

视觉协议对每16只手保存写入后、0.2 s和1 s全景，再在1 s状态冻结physics并逐环境保存原分辨率close-up。最终5页、15张时间全景与80张近景均通过manifest offset/row/asset identity核对；多模态审查未发现明显方块侧翻、悬空、弹出、严重互穿或脱离有效手指包络。视觉判断解释形态合理性，动态安全仍以strict数值门和80/80 task hold为主证据。

这些证据只说明初态满足当前simulation和contact观测下的reset质量标准。它们不回答共享策略能否学习连续旋转，也不支持其他object、scale、yaw、ADR、TIP-only sensing或未见手型泛化结论。

## Reproduction

```bash
source /home/hac/isaac/env_isaaclab/bin/activate

pytest -q source/anymani/anymani/pregrasp/tests
python scripts/research/generate_heterogeneous_mvp80_pregrasp_strict.py
python scripts/research/assemble_heterogeneous_mvp80_strict_catalog.py \
  --source <strict-summary-a.json> --source <strict-summary-b.json>
python scripts/research/finalize_heterogeneous_mvp80_selection.py \
  --summary outputs/pregrasp/search/heterogeneous_rotation_mvp80_dexcube_s1p1_v5/final-summary.json

python scripts/research/view_hetero_pregrasp.py \
  --offset 0 --count 16 --rank 0 --mode hold \
  --capture-dir outputs/pregrasp/visual/heterogeneous_rotation_mvp80_dexcube_s1p1_v5 \
  --capture-steps 1,24,120 --capture-closeups --auto-exit
```

宽松v4 identity保存在`tasks/hetero/config/generated/good_pregrasp_identity_v4.py`，仅用于重现历史对照；active runtime identity指向strict v5并在首次reset预载全部entries、验证key/payload digest和Top-8 gate，后续partial reset只做内存gather。
