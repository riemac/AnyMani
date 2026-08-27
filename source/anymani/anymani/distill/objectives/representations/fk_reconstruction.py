r"""Frame-origin FK 与 physical-landmark FK baseline objective contract。

该 objective 在相同 owner mask、asset/q split 与 backbone budget 下回归 3D points 或明确
命名的 pose target。frame-origin baseline 尽量复现 prior art；physical-landmark baseline
使用 collision centroid、surface landmark 或 distal physical point，减少 arbitrary URDF
origin 带来的 gauge weakness。

loss 必须声明 point 表达 frame、单位 m、是否按 hand scale 归一化、每个 joint/group 的
权重与缺失 landmark mask。若使用 orientation/pose 而非 position，必须另行定义 $SO(3)$
representation 与 branch/sign contract，不能把不同 target 统称为 FK MSE。
"""
