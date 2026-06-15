# IsaacLab GUI / NVIDIA 驱动排障记录

本文记录 2026-06-14 在 AnyMani GM heterogeneous hand 测试环境中遇到的 IsaacLab GUI 崩溃问题。核心目的是区分“AnyMani 异构资产问题”和“Isaac Sim / IsaacLab GUI / NVIDIA 驱动栈问题”，并保留可复用的修复与回滚路径。

## 机器与软件背景

- OS: Ubuntu 24.04.4 LTS, kernel `6.17.0-35-generic`。
- GPU: `NVIDIA GeForce RTX 5070 Ti`。
- Isaac Sim: `5.1.0.0`。
- IsaacLab repo: `v2.3.2-35-g8ddf2c7ac2c`，`VERSION=2.3.2`。
- AnyMani 测试任务: `AnyMani-GM-Heterogeneous-Test-v0`。

## 原始现象

GUI 模式启动 IsaacLab / AnyMani 环境时段错误崩溃，backtrace 多次落在以下 RTX 渲染栈：

- `librtx.scenedb.plugin.so`
- `libcarb.scenerenderer-rtx.plugin.so`
- `libomni.hydra.rtx.plugin.so`
- `libomni.usd.so`
- `libcarb.tasking.plugin.so`

同时 Kit 弹出 `IOMMU Enabled` 警告。该警告真实存在，但后续验证说明它不是本次 GUI 崩溃的主因。

## 排除路径

### 不是 AnyMani 异构资产专属问题

- AnyMani heterogeneous hand env 的 headless reset / step 正常。
- 官方单资产任务 `Isaac-Cartpole-v0` 在 GUI 下同样崩溃。
- 官方异构 articulation demo `scripts/demos/multi_asset.py` 在 GUI 下同样崩溃。
- 因此崩溃发生在 IsaacLab GUI experience / RTX renderer 启动栈，而不是 AnyMani URDF 或 heterogeneous hand cfg 的必然问题。

### 不是 headless 物理路径问题

- 官方 `Isaac-Cartpole-v0` headless 可创建、reset、step、close。
- 官方 `multi_asset.py --headless --num_envs 9` 可进入稳定 reset 循环。
- AnyMani `AnyMani-GM-Heterogeneous-Test-v0` headless 可输出：
  - action space shape: `(9, 16)`
  - policy obs shape: `(9, 32)`
  - reward shape: `(9,)`

### 不是安装方式错误

- IsaacSim 是按 IsaacLab 官方推荐的 `uv pip` 路径安装。
- IsaacLab 2.3.x 与 Isaac Sim 5.1 是正确主版本组合。
- Compatibility checker 给出 `PASSED`，但它只检查最低兼容门槛，不保证当前 NVIDIA 小版本是 Isaac Sim 5.1 GUI RTX 栈的稳定组合。

## 驱动历史与判断

故障发生前系统运行：

- `nvidia-driver-595-open 595.71.05`
- `linux-modules-nvidia-595-open-6.17.0-35-generic`

尝试切到 `nvidia-driver-595` 专有驱动后出现黑屏，需要 Timeshift 恢复。该结果说明不能继续赌专有内核模块路径。

apt / dpkg 历史显示，机器过去长期使用过 `nvidia-driver-580-open`，并在 2026-05-05 才切换到 `595-open`。因此更安全的修复路线是回到 R580 open kernel module 分支，而不是切到 R580 专有驱动。

## 最终修复

安装目标：

```bash
sudo apt install \
  nvidia-driver-580-open \
  linux-modules-nvidia-580-open-generic-hwe-24.04 \
  linux-modules-nvidia-580-open-6.17.0-35-generic
sudo update-initramfs -u -k 6.17.0-35-generic
sudo update-grub
```

执行时不要自动 `apt autoremove`。保留旧 firmware / 相关包的残留有助于降低立刻回滚时的风险。

重启后确认：

```bash
nvidia-smi
cat /proc/driver/nvidia/version
apt list --installed 'nvidia-driver-*' 'linux-modules-nvidia-*'
```

修复后稳定状态：

- `nvidia-driver-580-open 580.159.03`
- `linux-modules-nvidia-580-open-6.17.0-35-generic`
- runtime NVIDIA kernel module: `580.159.03`

## 修复后验证

### 官方单资产 GUI

```bash
cd /home/hac/isaac/IsaacLab
timeout 90s /home/hac/isaac/env_isaaclab/bin/python scripts/environments/random_agent.py \
  --task Isaac-Cartpole-v0 \
  --num_envs 4
```

结果：进入 `app ready`，环境创建成功，运行到 timeout 后正常 shutdown，无 segfault。

### 官方异构 articulation GUI

```bash
cd /home/hac/isaac/IsaacLab
timeout 90s /home/hac/isaac/env_isaaclab/bin/python scripts/demos/multi_asset.py \
  --num_envs 9
```

结果：进入 `app ready` 和 `Simulation App Startup Complete`，运行到 timeout 后正常 shutdown，无 segfault。

### AnyMani 异构手 GUI

```bash
cd /home/hac/isaac/AnyMani
PYTHONUNBUFFERED=1 /home/hac/isaac/env_isaaclab/bin/python -u -c "from isaaclab.app import AppLauncher; app_launcher=AppLauncher({'headless': False}); simulation_app=app_launcher.app; import gymnasium as gym, torch, isaaclab_tasks, anymani.tasks; from isaaclab_tasks.utils import parse_env_cfg; task='AnyMani-GM-Heterogeneous-Test-v0'; cfg=parse_env_cfg(task, device='cuda:0', num_envs=9); env=gym.make(task, cfg=cfg); obs,_=env.reset(); print('ANYMANI_GUI_RESET', obs['policy'].shape, flush=True); actions=torch.zeros(env.action_space.shape, device=env.unwrapped.device); [env.step(actions) for _ in range(10)]; print('ANYMANI_GUI_OK', flush=True); env.close(); simulation_app.close()"
```

结果：

```text
ANYMANI_GUI_RESET torch.Size([9, 32])
ANYMANI_GUI_OK
```

## 回滚方案

若切换 R580 后黑屏但可进入 TTY：

```bash
sudo apt install \
  nvidia-driver-595-open \
  linux-modules-nvidia-595-open-generic-hwe-24.04 \
  linux-modules-nvidia-595-open-6.17.0-35-generic
sudo update-initramfs -u -k 6.17.0-35-generic
sudo update-grub
sudo reboot
```

若连 TTY 都进不去，使用 Timeshift 或 GRUB recovery mode 恢复。双系统机器上不建议在没有恢复点时直接切换专有驱动或修改 IOMMU 启动参数。

## 后续注意事项

- 暂时不要升级回 `595-open`，也不要再尝试 `595` 专有驱动作为 Isaac Sim 5.1 GUI 的默认路径。
- 暂时不要执行 `apt autoremove` 清理 NVIDIA 相关残留，除非确认 GUI 路径长期稳定。
- `IOMMU Enabled` 警告仍会出现；在 `580-open` 下 GUI 已恢复，因此本次不应优先改 BIOS / GRUB IOMMU 参数。
- 若未来 Ubuntu 内核升级后再次出现 `nvidia-smi` 无法通信，优先检查当前内核是否有对应的 `linux-modules-nvidia-<branch>-open-$(uname -r)` 包。
