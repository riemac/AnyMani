# Embodiment MVP (URDF -> HIR -> URDF)

本目录提供跨手型资产最小实现链路：

1. 解析 URDF 到统一中间表示 HIR。
2. 在 HIR 上执行基础变异（拓扑/运动学/几何）。
3. 回写 URDF 与 metadata/manifest。
4. 对生成变体输出 PNG 预览图（结构渲染）与 URDF Visualizer 索引。

## 目录

- schema/hir_v01.py: HIR 数据结构定义
- io/urdf_to_hir.py: URDF 解析器
- io/hir_to_urdf.py: URDF 生成器
- mutate/topology.py: 拓扑变异
- mutate/kinematics.py: 运动学变异
- mutate/geometry.py: 几何变异
- validate/checks.py: 分级校验（error/warning）
- cli/gen_assets.py: 批量生成入口
- visualize/render_variants.py: 批量 PNG 可视化

## 快速开始

### 1) 生成 Leap 变体（20个）

python source/anymani/embodiment/cli/gen_assets.py \
  --input_urdf source/anymani/assets/leap_hand_sim_urdf/leap_hand/robot.urdf \
  --out_dir source/anymani/embodiment/_out/leap_mvp \
  --family leap \
  --count 20

### 2) 生成 Allegro 变体（20个）

python source/anymani/embodiment/cli/gen_assets.py \
  --input_urdf ../hora/assets/allegro/allegro.urdf \
  --out_dir source/anymani/embodiment/_out/allegro_mvp \
  --family allegro \
  --count 20

### 3) 单独渲染 PNG 预览

python source/anymani/embodiment/visualize/render_variants.py \
  --urdf_dir source/anymani/embodiment/_out/leap_mvp/urdf \
  --output_dir source/anymani/embodiment/_out/leap_mvp/png

## URDF Visualizer 使用

如果你安装了 VS Code 扩展 URDF Visualizer（morningfrog.urdf-visualizer），
可直接打开以下文件中的 URDF 列表：

- source/anymani/embodiment/_out/<family>_mvp/urdf_visualizer_index.md

然后在资源管理器中打开对应 urdf 文件进行可视化。

## 已知限制

- 当前 PNG 渲染为运动学结构预览，不是完整网格光照渲染。
- 几何编码首版仅支持 box/capsule/sphere/cylinder/mesh。
- 复杂分叉拓扑下 finger 推断采用保守策略（优先线性链）。
