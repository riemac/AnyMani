# robots/tests 约定

本目录只验证 `assets -> robots` 的纯配置 adapter：bundle 路径、articulation/spawner cfg、关节/执行器映射、frame、单位与 schema。默认测试不得启动 Isaac Sim、Kit、USD 或 `AppLauncher`。

POE/FK/Jacobian、owner collision union、surface/anchor provenance 与 physical identity 已归属 `distill/tests/contracts/representations/`，不得在这里复制第二套 oracle。依赖 importer pose、PhysX handle 或 spawn 生命周期的命题放在 `source/anymani/anymani/smokes/robots/`，通过显式 runtime 命令运行。
