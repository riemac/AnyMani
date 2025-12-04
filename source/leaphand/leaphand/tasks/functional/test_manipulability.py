#!/usr/bin/env python3

r"""启动 Isaac Sim 中的 LeapHand 场景，并提供实时位姿监控和可操作度分析面板。

继承自 `launch_with_leaphand.py` 的功能，额外增加了可操作度指标 (Manipulability Index) 和雅可比条件数 (Condition Number) 的实时计算与显示。

NOTE：坐标系统一约定：{w}-World坐标系，{e}-Env坐标系， {s}-Base/Root坐标系，{b}-End Effector坐标系（USD关节链中的最后一层刚体的坐标），{b'}-虚拟Xform坐标系（人为设置的指尖坐标系）
旋量、雅可比等均遵循 Modern Robotics 的约定

主要数学工具调用 `source/leaphand/leaphand/tasks/manager_based/leaphand/mdp/utils/math.py`

可操作度应是指尖末端 {b'} 相对于控制关节的可操作度，因此雅可比矩阵应为 Jb'，参考点在{b'}，参考系在{b'}
FIXME：操作度、雅可比条件数等检测都应为Jb'的，但现在都为Jb

使用方法:
    ./isaaclab.sh -p source/leaphand/leaphand/tasks/functional/test_manipulability.py
不要启动headless模式！
"""

import argparse
import sys
import os
from dataclasses import dataclass

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="启动LeapHand场景并显示可操作度分析面板")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.ui as ui
import torch
import numpy as np

from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.utils import get_current_stage, resolve_prim_pose

# 尝试导入 launch_with_leaphand 中的类
# 确保当前目录在 sys.path 中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from launch_with_leaphand import LeapHandPoseMonitor, LeapHandControlPanel, LeapHandSceneCfg
except ImportError:
    print("[ERROR] 无法导入 launch_with_leaphand.py。请确保该文件在同一目录下。")
    sys.exit(1)

from leaphand.tasks.manager_based.leaphand.mdp.utils import math as math_leap


@dataclass
class FingerConfig:
    name: str
    body: str
    joint_names: list[str]
    marker_path: str | None = None


class FingerJacobianProbe:
    """负责从 PhysX 提取并转换到 {b'} 表达的雅可比。"""

    def __init__(self, robot):
        self.robot = robot
        self.device = robot.device
        self.is_fixed_base = robot.is_fixed_base
        self.body_names = robot.data.body_names
        self.joint_names = robot.joint_names

    def resolve_indices(self, cfg: FingerConfig) -> dict:
        body_ids, _ = self.robot.find_bodies(cfg.body)
        if not body_ids:
            raise RuntimeError(f"未找到刚体 {cfg.body}")

        body_idx = body_ids[0]
        jacobian_idx = body_idx - 1 if self.is_fixed_base else body_idx
        joint_indices = [self.joint_names.index(name) for name in cfg.joint_names]
        jac_joint_ids = (
            joint_indices if self.is_fixed_base else [idx + 6 for idx in joint_indices]
        )

        return {
            "body_idx": body_idx,
            "jacobian_idx": jacobian_idx,
            "joint_indices": joint_indices,
            "jacobian_joint_ids": jac_joint_ids,
        }

    def jacobian_bprime(self, indices: dict, T_bb_prime: torch.Tensor | None) -> torch.Tensor:
        all_jacobians = self.robot.root_physx_view.get_jacobians()
        jac_physx = all_jacobians[0, indices["jacobian_idx"], :, :][:, indices["jacobian_joint_ids"]]
        jac_world = torch.cat([jac_physx[3:, :], jac_physx[:3, :]], dim=0).unsqueeze(0)

        body_pos = self.robot.data.body_pos_w[:, indices["body_idx"]]
        body_quat = self.robot.data.body_quat_w[:, indices["body_idx"]]
        T_wb = math_leap.transform_from_pos_quat(body_pos, body_quat)
        T_bw = math_leap.inverse_transform(T_wb)
        Ad_bw = math_leap.adjoint_transform(T_bw)
        jac_b = torch.matmul(Ad_bw, jac_world).squeeze(0)

        if T_bb_prime is not None:
            Ad = math_leap.adjoint_transform(math_leap.inverse_transform(T_bb_prime))
            jac_b = torch.matmul(Ad, jac_b)

        return jac_b


class LeapHandManipulabilityPanel:
    """LeapHand 可操作度分析面板
    
    实时显示各手指的可操作度指标 (Manipulability Index) 和雅可比条件数 (Condition Number)。
    支持原始雅可比 J_b 和加权雅可比 J_weight_b = W_x @ J_b @ W_q^{-1} 的对比分析。
    
    Attributes:
        W_x: 任务空间权重矩阵 (6×6)，用于调节位置/姿态的相对重要性
        W_q: 关节空间权重矩阵 (4×4)，用于约束关节运动
    """
    
    # ========== 可配置的权重参数 ==========
    # 任务空间权重 (PhysX格式: [v; ω])
    L_CHAR = 0.05          # 特征长度 (m)，用于单位归一化
    W_OMEGA = 1.0          # 角速度权重
    W_V = 1.0 / L_CHAR      # 线速度权重 (默认 1/L_CHAR = 20)
    
    # 关节空间权重 (4个关节)
    W_Q_DIAG = [1.0, 1.0, 1.0, 1.0]  # 各关节的权重
    # =====================================
    
    def __init__(self, robot):
        """初始化面板
        
        Args:
            robot: LeapHand机器人实例
        """
        self.robot = robot
        self.device = robot.device
        self.probe = FingerJacobianProbe(robot)
        
        finger_defs = {
            "Index": ("fingertip", ["a_0", "a_1", "a_2", "a_3"], "index_tip_head"),
            "Middle": ("fingertip_2", ["a_4", "a_5", "a_6", "a_7"], "middle_tip_head"),
            "Ring": ("fingertip_3", ["a_8", "a_9", "a_10", "a_11"], "ring_tip_head"),
            "Thumb": ("thumb_fingertip", ["a_12", "a_13", "a_14", "a_15"], "thumb_tip_head"),
        }
        self.fingers = {
            name: {
                "cfg": FingerConfig(name=name, body=body, joint_names=joint_names, marker_path=marker_path),
                "indices": None,
            }
            for name, (body, joint_names, marker_path) in finger_defs.items()
        }
        self._T_bb_prime: dict[str, torch.Tensor | None] = {name: None for name in self.fingers}
        
        # 构建权重矩阵
        self._build_weight_matrices()
        
        self._resolve_indices()
        self._compute_virtual_finger_frames()
        
        # 创建UI窗口
        self._window = ui.Window(
            "LeapHand Manipulability Analysis", 
            width=450, 
            height=600,
            flags=ui.WINDOW_FLAGS_NO_COLLAPSE,
            dock_preference=ui.DockPreference.RIGHT_BOTTOM
        )
        
        # 存储UI标签引用
        self.labels = {}
        
        with self._window.frame:
            with ui.ScrollingFrame(
                height=ui.Fraction(1),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                with ui.VStack(spacing=10, height=0):
                    # 标题
                    ui.Label("Manipulability Analysis", 
                            height=30, 
                            style={"font_size": 18, "color": 0xFF00AAFF})
                    
                    # 显示权重配置
                    self._create_weight_info_row()
                    
                    ui.Separator(height=2)
                    
                    # 为每个手指创建显示区域
                    for finger_name in self.fingers.keys():
                        self._create_finger_row(finger_name)
                    
                    ui.Separator(height=10)
                    
                    # 打印按钮
                    ui.Button("Print All Fingers Jacobian Info to Terminal", 
                             height=40,
                             clicked_fn=self._print_info,
                             style={"background_color": 0xFF00AAFF, "font_size": 14})
    
    def _build_weight_matrices(self):
        """构建权重矩阵 W_x 和 W_q"""
        # W_x: 任务空间权重 (6×6)
        # Modern Robotics 格式: [ω_x, ω_y, ω_z, v_x, v_y, v_z]
        # 注意：PhysX 返回的是 [v; ω]，但我们在 FingerJacobianProbe 中转换为了 [ω; v]
        # 因此权重矩阵也必须遵循 [ω; v] 的顺序
        w_x_diag = [self.W_OMEGA, self.W_OMEGA, self.W_OMEGA,
                    self.W_V, self.W_V, self.W_V]
        self._W_x = torch.diag(torch.tensor(w_x_diag, device=self.device, dtype=torch.float32))
        
        # W_q: 关节空间权重 (4×4)
        self._W_q = torch.diag(torch.tensor(self.W_Q_DIAG, device=self.device, dtype=torch.float32))
        self._W_q_inv = torch.diag(1.0 / torch.tensor(self.W_Q_DIAG, device=self.device, dtype=torch.float32))
        
        print(f"[INFO] 权重矩阵已构建:")
        print(f"  W_x (任务空间): diag([{self.W_OMEGA:.2f}, {self.W_OMEGA:.2f}, {self.W_OMEGA:.2f}, "
              f"{self.W_V:.2f}, {self.W_V:.2f}, {self.W_V:.2f}])")
        print(f"  W_q (关节空间): diag({self.W_Q_DIAG})")
    
    def _create_weight_info_row(self):
        """创建权重配置信息显示行"""
        with ui.CollapsableFrame(
            title="Weight Configuration",
            height=0,
            collapsed=True,
            style={"color": 0xFFCCCCCC}
        ):
            with ui.VStack(spacing=3, height=0):
                ui.Label(f"L_char = {self.L_CHAR} m", style={"color": 0xFFAAAAAA, "font_size": 10})
                ui.Label(f"W_x: w_ω = {self.W_OMEGA:.2f}, w_v = {self.W_V:.2f}", 
                        style={"color": 0xFFAAAAAA, "font_size": 10})
                ui.Label(f"W_q: diag({self.W_Q_DIAG})", 
                        style={"color": 0xFFAAAAAA, "font_size": 10})

    def _resolve_indices(self):
        for name, entry in self.fingers.items():
            try:
                entry["indices"] = self.probe.resolve_indices(entry["cfg"])
                print(f"[INFO] {name} indices ready: {entry['indices']}")
            except RuntimeError as exc:
                print(f"[WARNING] {exc}")
                entry["indices"] = None

    def _compute_virtual_finger_frames(self):
        """计算每根手指的虚拟指尖坐标系 T_bb'"""

        stage = get_current_stage()
        robot_root = self.robot.root_physx_view.prim_paths[0]

        for name, entry in self.fingers.items():
            cfg = entry["cfg"]
            marker_path = cfg.marker_path
            if marker_path is None:
                self._T_bb_prime[name] = None
                continue

            parent_prim_path = f"{robot_root}/{cfg.body}"
            marker_prim_path = f"{parent_prim_path}/{marker_path}"
            parent_prim = stage.GetPrimAtPath(parent_prim_path)
            marker_prim = stage.GetPrimAtPath(marker_prim_path)

            if not parent_prim.IsValid() or not marker_prim.IsValid():
                print(f"[WARNING] 无法找到 {name} 指尖标记: {marker_prim_path}")
                self._T_bb_prime[name] = None
                continue

            offset_pos, offset_quat = resolve_prim_pose(marker_prim, ref_prim=parent_prim)
            offset_pos_tensor = torch.tensor(offset_pos, device=self.device, dtype=torch.float32)
            offset_quat_tensor = torch.tensor(offset_quat, device=self.device, dtype=torch.float32)
            T_bb_prime = math_leap.transform_from_pos_quat(
                offset_pos_tensor.unsqueeze(0), offset_quat_tensor.unsqueeze(0)
            ).squeeze(0)
            self._T_bb_prime[name] = T_bb_prime
            print(
                f"[INFO] {name} 指尖偏移加载完成: {marker_path}, 平移={offset_pos_tensor.cpu().numpy()}"
            )

    def _create_finger_row(self, finger_name: str):
        """创建单个手指的显示行
        
        显示两组指标：
        1. 原始雅可比 J_b (6x4)：直接从 PhysX 获取
        2. 加权雅可比 J_weight_b (6x4)：J_weight_b = W_x @ J_b @ W_q^{-1}
        """
        with ui.CollapsableFrame(
            title=finger_name,
            height=0,
            collapsed=False,
            style={"color": 0xFFEEEEEE}
        ):
            with ui.VStack(spacing=5, height=0):
                # ===== 原始雅可比 J_b (6x4) =====
                ui.Label("── Original J_b (6×4) ──", 
                        height=18, style={"color": 0xFFAAAAAA, "font_size": 11})
                
                # 可操作度指标
                with ui.HStack(spacing=10, height=20):
                    ui.Label("Manipulability (w):", width=150, style={"color": 0xFF88FF88})
                    w_label = ui.Label("0.0000", style={"color": 0xFFFFFFFF})
                
                # 条件数
                with ui.HStack(spacing=10, height=20):
                    ui.Label("Condition No. (κ):", width=150, style={"color": 0xFFFF8888})
                    k_label = ui.Label("0.0000", style={"color": 0xFFFFFFFF})
                
                # 最小奇异值
                with ui.HStack(spacing=10, height=20):
                    ui.Label("Min Singular Val:", width=150, style={"color": 0xFF8888FF})
                    s_label = ui.Label("0.0000", style={"color": 0xFFFFFFFF})
                
                ui.Separator(height=5)
                
                # ===== 加权雅可比 J_weight_b (6x4) =====
                ui.Label("── Weighted J_weight_b (6×4) ──", 
                        height=18, style={"color": 0xFFAAAAAA, "font_size": 11})
                
                # 加权可操作度
                with ui.HStack(spacing=10, height=20):
                    ui.Label("Manip (w_w):", width=150, style={"color": 0xFF88FFFF})
                    w_w_label = ui.Label("0.0000", style={"color": 0xFFFFFFFF})
                
                # 加权条件数
                with ui.HStack(spacing=10, height=20):
                    ui.Label("Cond No. (κ_w):", width=150, style={"color": 0xFFFFFF88})
                    k_w_label = ui.Label("0.0000", style={"color": 0xFFFFFFFF})
                
                # 加权最小奇异值
                with ui.HStack(spacing=10, height=20):
                    ui.Label("Min Singular (σ_w):", width=150, style={"color": 0xFFFF88FF})
                    s_w_label = ui.Label("0.0000", style={"color": 0xFFFFFFFF})
                
                # 单独打印按钮
                ui.Button(f"Print {finger_name} Jacobian", 
                         height=25,
                         clicked_fn=lambda fn=finger_name: self._print_single_finger_info(fn),
                         style={"background_color": 0xFF666666, "font_size": 12})
                
                self.labels[finger_name] = {
                    "w": w_label,
                    "k": k_label,
                    "s": s_label,
                    "w_w": w_w_label,
                    "k_w": k_w_label,
                    "s_w": s_w_label
                }

    def update(self):
        """更新指标显示
        
        计算两组可操作度指标：
        1. 原始雅可比 J_b (6×4): w = sqrt(det(J^T * J))
        2. 加权雅可比 J_weight_b (6×4): J_weight_b = W_x @ J_b @ W_q^{-1}
        
        Notes:
            PhysX 返回的雅可比矩阵格式为 [v; ω]（线速度在前，角速度在后）。
            对于欠驱动系统 (m=6 > n=4)，可操作度公式为 w = sqrt(det(J^T * J))。
        """
        # 获取所有刚体的雅可比矩阵
        for name, entry in self.fingers.items():
            indices = entry.get("indices")
            if not indices:
                continue

            try:
                jac_b = self.probe.jacobian_bprime(indices, self._T_bb_prime.get(name))
            except Exception as exc:
                print(f"[ERROR] 获取 {name} 雅可比失败: {exc}")
                continue

            jac_w = self._W_x @ jac_b @ self._W_q_inv
            jac_b_batch = jac_b.unsqueeze(0)
            jac_w_batch = jac_w.unsqueeze(0)
            _, S, _ = math_leap.svd(jac_b_batch)
            _, S_w, _ = math_leap.svd(jac_w_batch)
            metrics = {
                "w": math_leap.manipulability(jac_b_batch)[0].item(),
                "cond": math_leap.condition_number(jac_b_batch)[0].item(),
                "sigma_min": S[0, -1].item(),
                "w_w": math_leap.manipulability(jac_w_batch)[0].item(),
                "cond_w": math_leap.condition_number(jac_w_batch)[0].item(),
                "sigma_min_w": S_w[0, -1].item(),
            }

            self.labels[name]["w"].text = f"{metrics['w']:.4f}"
            self.labels[name]["k"].text = f"{metrics['cond']:.2f}" if metrics['cond'] < 1e6 else "∞"
            self.labels[name]["s"].text = f"{metrics['sigma_min']:.4f}"
            self.labels[name]["w_w"].text = f"{metrics['w_w']:.4f}"
            self.labels[name]["k_w"].text = f"{metrics['cond_w']:.2f}" if metrics['cond_w'] < 1e6 else "∞"
            self.labels[name]["s_w"].text = f"{metrics['sigma_min_w']:.4f}"

    def _print_single_finger_info(self, finger_name: str):
        """打印单个手指的详细雅可比信息到终端
        
        Args:
            finger_name: 手指名称
        """
        entry = self.fingers.get(finger_name)
        if entry is None:
            print(f"[ERROR] Unknown finger: {finger_name}")
            return
        indices = entry.get("indices")
        if not indices:
            print(f"[ERROR] Missing indices for finger {finger_name}")
            return

        jac_b = self.probe.jacobian_bprime(indices, self._T_bb_prime.get(finger_name))
        jac_w = self._W_x @ jac_b @ self._W_q_inv
        jac_b_batch = jac_b.unsqueeze(0)
        jac_w_batch = jac_w.unsqueeze(0)
        _, S, _ = math_leap.svd(jac_b_batch)
        _, S_w, _ = math_leap.svd(jac_w_batch)
        w = math_leap.manipulability(jac_b_batch)[0].item()
        cond = math_leap.condition_number(jac_b_batch)[0].item()
        w_w = math_leap.manipulability(jac_w_batch)[0].item()
        cond_w = math_leap.condition_number(jac_w_batch)[0].item()
        sigma_min = S[0, -1].item()
        sigma_min_w = S_w[0, -1].item()
        
        # 格式化打印设置
        np.set_printoptions(precision=6, suppress=True, linewidth=120)
        
        # 获取当前关节角度
        joint_indices = indices["joint_indices"]
        joint_pos = self.robot.data.joint_pos[0, joint_indices].cpu().numpy()
        joint_names = [self.robot.joint_names[i] for i in joint_indices]
        
        # 打印详细信息
        print("\n" + "="*35 + f" {finger_name} Finger Jacobian " + "="*35)
        print(f"Body Idx: {indices['body_idx']}, Jacobi Idx: {indices['jacobian_idx']}")
        print(f"Joint Indices: {joint_indices}")
        
        # --- 当前关节角度 ---
        print(f"\n[Current Joint Positions (rad)]")
        for i, (name, pos) in enumerate(zip(joint_names, joint_pos)):
            print(f"  {name}: {pos:.4f} rad ({np.degrees(pos):.2f}°)")
        
        # --- 权重配置 ---
        print(f"\n[Weight Configuration]")
        print(f"  W_x (task space): w_ω = {self.W_OMEGA:.2f}, w_v = {self.W_V:.2f}")
        print(f"  W_q (joint space): diag({self.W_Q_DIAG})")
        
        # --- 原始雅可比 (6x4) ---
        print(f"\n[Original J_b (6×4)] - Modern Robotics Convention:")
        print(f"  Row 0-2: Angular velocity [ωx, ωy, ωz]")
        print(f"  Row 3-5: Linear velocity  [vx, vy, vz]")
        print(jac_b.cpu().numpy())
        print(f"  Singular Values: {S.cpu().numpy()}")
        print(f"  Condition Number (κ): {cond:.4f}")
        print(f"  Manipulability (w): {w:.6f}")
        
        # --- 加权雅可比 (6x4) ---
        print(f"\n[Weighted J_weight_b (6×4)] - J_weight_b = W_x @ J_b @ W_q^{{-1}}:")
        print(f"  Row 0-2: Weighted angular velocity")
        print(f"  Row 3-5: Weighted linear velocity")
        print(jac_w.cpu().numpy())
        print(f"  Singular Values: {S_w.cpu().numpy()}")
        print(f"  Condition Number (κ_w): {cond_w:.4f}")
        print(f"  Manipulability (w_w): {w_w:.6f}")
        
        print("="*95 + "\n")

    def _print_info(self):
        """打印所有手指的详细信息到终端
        
        以表格形式对比原始雅可比和加权雅可比的指标。
        """
        print("\n" + "="*45 + " Jacobian Analysis Summary " + "="*45)
        print("\nWeight Configuration:")
        print(f"  W_x (task space): w_ω = {self.W_OMEGA:.2f}, w_v = {self.W_V:.2f}")
        print(f"  W_q (joint space): diag({self.W_Q_DIAG})")
        print(f"  L_char = {self.L_CHAR} m")
        
        # 收集所有手指数据
        finger_data_list = []
        
        for finger_name, entry in self.fingers.items():
            indices = entry.get("indices")
            if not indices:
                continue

            jac_b = self.probe.jacobian_bprime(indices, self._T_bb_prime.get(finger_name))
            jac_w = self._W_x @ jac_b @ self._W_q_inv
            jac_b_batch = jac_b.unsqueeze(0)
            jac_w_batch = jac_w.unsqueeze(0)
            _, S, _ = math_leap.svd(jac_b_batch)
            _, S_w, _ = math_leap.svd(jac_w_batch)

            finger_data_list.append(
                {
                    "name": finger_name,
                    "joint_names": [self.robot.joint_names[i] for i in indices["joint_indices"]],
                    "joint_pos": self.robot.data.joint_pos[0, indices["joint_indices"]].cpu().numpy(),
                    "w": math_leap.manipulability(jac_b_batch)[0].item(),
                    "cond": math_leap.condition_number(jac_b_batch)[0].item(),
                    "sigma_min": S[0, -1].item(),
                    "w_w": math_leap.manipulability(jac_w_batch)[0].item(),
                    "cond_w": math_leap.condition_number(jac_w_batch)[0].item(),
                    "sigma_min_w": S_w[0, -1].item(),
                }
            )
        
        # 打印关节角度表格
        print("\n[Current Joint Positions]")
        print("-" * 100)
        for fd in finger_data_list:
            pos_str = ", ".join([f"{np.degrees(p):.1f}°" for p in fd["joint_pos"]])
            print(f"  {fd['name']:<8}: [{pos_str}]")
            joint_detail = "  ".join([f"{n}={p:.3f}" for n, p in zip(fd["joint_names"], fd["joint_pos"])])
            print(f"           ({joint_detail})")
        
        # 打印操作度表格
        print("\n[Manipulability Metrics]")
        print(f"{'Finger':<8} | {'w(J_b)':<10} | {'κ(J_b)':<10} | {'σ_min':<8} | {'w(J_weight_b)':<10} | {'κ(J_weight_b)':<10} | {'σ_min_w':<8}")
        print("-" * 85)
        
        for fd in finger_data_list:
            cond_str = f"{fd['cond']:.2f}" if fd['cond'] < 1e6 else "∞"
            cond_w_str = f"{fd['cond_w']:.2f}" if fd['cond_w'] < 1e6 else "∞"
            print(f"{fd['name']:<8} | {fd['w']:<10.4f} | {cond_str:<10} | {fd['sigma_min']:<8.4f} | "
                  f"{fd['w_w']:<10.4f} | {cond_w_str:<10} | {fd['sigma_min_w']:<8.4f}")
            
        print("="*115 + "\n")


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """运行仿真循环"""
    # 获取机器人实例
    robot = scene["robot"]
    
    # 定义需要监控的自定义Prim（标记点）
    custom_marker_prims = [
        ("fingertip", "index_tip_head"),        # 食指指尖标记
        ("thumb_fingertip", "thumb_tip_head"),  # 拇指指尖标记
        ("fingertip_2", "middle_tip_head"),     # 中指指尖标记
        ("fingertip_3", "ring_tip_head"),       # 无名指指尖标记
    ]
    
    # 创建UI面板
    pose_monitor = LeapHandPoseMonitor(robot, custom_prims=custom_marker_prims)
    control_panel = LeapHandControlPanel(robot)
    manip_panel = LeapHandManipulabilityPanel(robot)  # 新增可操作度面板
    
    # 仿真参数
    sim_dt = sim.get_physics_dt()
    count = 0
    
    print("\n" + "=" * 80)
    print("LeapHand 可操作度分析场景已启动")
    print("  - UI面板:")
    print("    * LeapHand Body Pose Monitor: 实时位姿监控")
    print("    * LeapHand Joint Control: 关节角度控制")
    print("    * LeapHand Manipulability Analysis: 可操作度分析")
    print("=" * 80 + "\n")
    
    # 仿真循环
    while simulation_app.is_running():
        # 将关节目标位置写入仿真
        scene.write_data_to_sim()
        
        # 执行仿真步进
        sim.step()
        
        # 更新场景
        scene.update(sim_dt)
        
        # 更新UI面板 (每10帧更新一次)
        if count % 10 == 0:
            pose_monitor.update()
            manip_panel.update()
        
        count += 1


def main():
    """主函数"""
    # 创建仿真上下文
    sim_cfg = SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # 设置相机视角
    sim.set_camera_view([1.5, 1.5, 1.5], [0.0, 0.0, 0.5])
    
    # 创建场景
    scene_cfg = LeapHandSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # 初始化仿真
    sim.reset()
    
    print("[INFO]: 场景设置完成，开始运行...")
    
    # 运行仿真器
    run_simulator(sim, scene)


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()
