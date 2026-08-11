# robots/tests 约定

本目录验证 `assets -> robots` 的纯 Python/PyTorch embodiment contract。测试可以读取真实生成资产的 sidecar、调用 bank 迁移、lower 静态运动学、构造 owner-local collision union 和检查 surface/anchor provenance，但不得启动 Isaac Sim、Kit、USD 或 `AppLauncher`。

运动学、owner geometry、asset-to-robot lowering 的测试归属本目录；Gaussian field、query/target、模型、objective 与 SSL 生命周期测试归属 `distill/tests`。Isaac importer pose parity、PhysX handle 和 spawn 生命周期放在 `source/anymani/anymani/smokes/robots/`，只通过显式 runtime 命令运行。

每个测试应证伪明确的 frame、单位、owner/joint routing、SE(3) 公式、Boolean/surface 真值或 cache provenance 命题，不以导入成功或对象存在代替科学合同。
