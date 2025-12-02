#!/usr/bin/env python3

"""
启动 Isaac Sim 中的 LeapHand 场景，并提供实时位姿监控和可操作度分析面板。

继承自 `launch_with_leaphand.py` 的功能，额外增加了可操作度指标 (Manipulability Index)
和雅可比条件数 (Condition Number) 的实时计算与显示。

使用方法:
    ./isaaclab.sh -p source/leaphand/leaphand/tasks/functional/test_manipulability.py
"""

import argparse
import sys
import os

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


class LeapHandManipulabilityPanel:
    """LeapHand 可操作度分析面板
    
    实时显示各手指的可操作度指标 (Manipulability Index) 和雅可比条件数 (Condition Number)。
    支持原始雅可比 J_b 和加权雅可比 J_w = W_x @ J_b @ W_q^{-1} 的对比分析。
    
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
        
        # 定义手指及其对应的末端刚体名称
        self.fingers = {
            "Index": {"body": "fingertip", "joint_indices": []},
            "Thumb": {"body": "thumb_fingertip", "joint_indices": []},
            "Middle": {"body": "fingertip_2", "joint_indices": []},
            "Ring": {"body": "fingertip_3", "joint_indices": []},
        }
        
        # 构建权重矩阵
        self._build_weight_matrices()
        
        # 解析各手指对应的关节索引
        self._resolve_indices()
        
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
        # PhysX 格式: [v_x, v_y, v_z, ω_x, ω_y, ω_z]
        w_x_diag = [self.W_V, self.W_V, self.W_V, 
                    self.W_OMEGA, self.W_OMEGA, self.W_OMEGA]
        self._W_x = torch.diag(torch.tensor(w_x_diag, device=self.device, dtype=torch.float32))
        
        # W_q: 关节空间权重 (4×4)
        self._W_q = torch.diag(torch.tensor(self.W_Q_DIAG, device=self.device, dtype=torch.float32))
        self._W_q_inv = torch.diag(1.0 / torch.tensor(self.W_Q_DIAG, device=self.device, dtype=torch.float32))
        
        print(f"[INFO] 权重矩阵已构建:")
        print(f"  W_x (任务空间): diag([{self.W_V:.2f}, {self.W_V:.2f}, {self.W_V:.2f}, "
              f"{self.W_OMEGA:.2f}, {self.W_OMEGA:.2f}, {self.W_OMEGA:.2f}])")
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
                ui.Label(f"W_x: w_v = {self.W_V:.2f}, w_ω = {self.W_OMEGA:.2f}", 
                        style={"color": 0xFFAAAAAA, "font_size": 10})
                ui.Label(f"W_q: diag({self.W_Q_DIAG})", 
                        style={"color": 0xFFAAAAAA, "font_size": 10})

    def _resolve_indices(self):
        """解析刚体和关节索引
        
        Notes:
            对于固定基座机器人，PhysX 返回的雅可比矩阵不包含根刚体，
            因此 body 索引需要减 1 才能正确访问雅可比矩阵。
        """
        joint_names_all = self.robot.joint_names
        
        # 检查是否固定基座
        self._is_fixed_base = self.robot.is_fixed_base
        
        # 备用映射表 (LeapHand 关节命名)
        finger_joint_map_alt = {
            "Index": ["a_0", "a_1", "a_2", "a_3"],
            "Middle": ["a_4", "a_5", "a_6", "a_7"],
            "Ring": ["a_8", "a_9", "a_10", "a_11"],
            "Thumb": ["a_12", "a_13", "a_14", "a_15"],
        }

        print(f"[INFO] LeapHand is_fixed_base: {self._is_fixed_base}")
        print(f"[INFO] Joint names: {joint_names_all}")
        print(f"[INFO] Body names: {self.robot.data.body_names}")

        # 获取刚体索引和关节索引
        for finger_name, data in self.fingers.items():
            # 1. Body Index
            body_name = data["body"]
            body_ids, _ = self.robot.find_bodies(body_name)
            if len(body_ids) > 0:
                data["body_idx"] = body_ids[0]
                # 对于固定基座，雅可比矩阵的索引需要 -1
                if self._is_fixed_base:
                    data["jacobi_idx"] = body_ids[0] - 1
                else:
                    data["jacobi_idx"] = body_ids[0]
                print(f"[INFO] {finger_name}: body_idx={data['body_idx']}, jacobi_idx={data['jacobi_idx']}")
            else:
                print(f"[WARNING] 无法找到刚体: {body_name}")
                data["body_idx"] = None
                data["jacobi_idx"] = None
            
            # 2. Joint Indices
            target_joints = finger_joint_map_alt.get(finger_name, [])
            
            indices = []
            for j_name in target_joints:
                if j_name in joint_names_all:
                    indices.append(joint_names_all.index(j_name))
            
            data["joint_indices"] = indices
            if len(indices) != 4:
                print(f"[WARNING] 手指 {finger_name} 只有 {len(indices)} 个关节被找到 (期望 4 个)")
            else:
                print(f"[INFO] {finger_name} joint_indices: {indices}")

    def _create_finger_row(self, finger_name: str):
        """创建单个手指的显示行
        
        显示两组指标：
        1. 原始雅可比 J_b (6x4)：直接从 PhysX 获取
        2. 加权雅可比 J_w (6x4)：J_w = W_x @ J_b @ W_q^{-1}
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
                
                # ===== 加权雅可比 J_w (6x4) =====
                ui.Label("── Weighted J_w (6×4) ──", 
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
        2. 加权雅可比 J_w (6×4): J_w = W_x @ J_b @ W_q^{-1}
        
        Notes:
            PhysX 返回的雅可比矩阵格式为 [v; ω]（线速度在前，角速度在后）。
            对于欠驱动系统 (m=6 > n=4)，可操作度公式为 w = sqrt(det(J^T * J))。
        """
        # 获取所有刚体的雅可比矩阵
        all_jacobians = self.robot.root_physx_view.get_jacobians()
        num_jacobi_bodies = all_jacobians.shape[1]
        
        # 收集需要计算的 Jacobian
        jac_list = []       # 原始雅可比 J_b (6, 4)
        jac_w_list = []     # 加权雅可比 J_w (6, 4)
        valid_fingers = []
        
        for finger_name, data in self.fingers.items():
            jacobi_idx = data.get("jacobi_idx")
            joint_indices = data.get("joint_indices")
            
            if jacobi_idx is not None and joint_indices:
                if jacobi_idx < 0 or jacobi_idx >= num_jacobi_bodies:
                    if not hasattr(self, "_warned_indices"):
                        self._warned_indices = set()
                    if jacobi_idx not in self._warned_indices:
                        print(f"[WARNING] Jacobi index {jacobi_idx} out of bounds for finger {finger_name}.")
                        self._warned_indices.add(jacobi_idx)
                    continue

                # 提取第0个环境，指定body
                jac_full = all_jacobians[0, jacobi_idx, :, :]
                
                # 只提取该手指相关的关节列 (6, 4)
                jac_b = jac_full[:, joint_indices]
                
                # 计算加权雅可比 J_w = W_x @ J_b @ W_q^{-1}
                jac_w = self._W_x @ jac_b @ self._W_q_inv
                
                jac_list.append(jac_b)
                jac_w_list.append(jac_w)
                valid_fingers.append(finger_name)
        
        if not jac_list:
            return

        # 堆叠成 batch
        batched_jac = torch.stack(jac_list)      # (B, 6, 4)
        batched_jac_w = torch.stack(jac_w_list)  # (B, 6, 4)
        
        # ===== 1. 原始雅可比可操作度 (欠驱动: m=6 > n=4) =====
        # w = sqrt(det(J^T * J))
        jt_j = torch.bmm(batched_jac.transpose(1, 2), batched_jac)
        w_vals = torch.sqrt(torch.abs(torch.det(jt_j)))
        
        # ===== 2. 加权雅可比可操作度 =====
        jt_j_w = torch.bmm(batched_jac_w.transpose(1, 2), batched_jac_w)
        w_w_vals = torch.sqrt(torch.abs(torch.det(jt_j_w)))
        
        # ===== 3. SVD for Condition Numbers =====
        try:
            S_vals = torch.linalg.svdvals(batched_jac)      # (B, 4)
            S_w_vals = torch.linalg.svdvals(batched_jac_w)  # (B, 4)
            
            # 同步到 CPU
            w_cpu = w_vals.cpu().tolist()
            w_w_cpu = w_w_vals.cpu().tolist()
            S_cpu = S_vals.cpu().tolist()
            S_w_cpu = S_w_vals.cpu().tolist()
            
            # 更新 UI
            for i, finger_name in enumerate(valid_fingers):
                # --- 原始雅可比指标 ---
                w = w_cpu[i]
                S = S_cpu[i]
                sigma_max = S[0]
                sigma_min = S[-1]
                cond_num = sigma_max / sigma_min if sigma_min > 1e-6 else float('inf')
                
                self.labels[finger_name]["w"].text = f"{w:.4f}"
                self.labels[finger_name]["k"].text = f"{cond_num:.2f}"
                self.labels[finger_name]["s"].text = f"{sigma_min:.4f}"
                
                # --- 加权雅可比指标 ---
                w_w = w_w_cpu[i]
                S_w = S_w_cpu[i]
                sigma_max_w = S_w[0]
                sigma_min_w = S_w[-1]
                cond_w = sigma_max_w / sigma_min_w if sigma_min_w > 1e-6 else float('inf')
                
                self.labels[finger_name]["w_w"].text = f"{w_w:.4f}"
                self.labels[finger_name]["k_w"].text = f"{cond_w:.2f}"
                self.labels[finger_name]["s_w"].text = f"{sigma_min_w:.4f}"
                
        except Exception as e:
            print(f"[ERROR] SVD calculation failed: {e}")

    def _print_single_finger_info(self, finger_name: str):
        """打印单个手指的详细雅可比信息到终端
        
        Args:
            finger_name: 手指名称
        """
        data = self.fingers.get(finger_name)
        if data is None:
            print(f"[ERROR] Unknown finger: {finger_name}")
            return
        
        jacobi_idx = data.get("jacobi_idx")
        joint_indices = data.get("joint_indices")
        body_idx = data.get("body_idx")
        
        if jacobi_idx is None or not joint_indices:
            print(f"[ERROR] Invalid indices for finger: {finger_name}")
            return
        
        all_jacobians = self.robot.root_physx_view.get_jacobians()
        num_jacobi_bodies = all_jacobians.shape[1]
        
        if jacobi_idx < 0 or jacobi_idx >= num_jacobi_bodies:
            print(f"[ERROR] Jacobi index {jacobi_idx} out of bounds for finger {finger_name}")
            return
        
        # 提取该手指的雅可比子矩阵 (6, 4)
        # PhysX 返回格式: [v; w] (线速度在前，角速度在后)
        jac_physx = all_jacobians[0, jacobi_idx, :, :][:, joint_indices]
        
        # 转换为 Modern Robotics 约定: [w; v] (角速度在前，线速度在后)
        jac = torch.cat([jac_physx[3:, :], jac_physx[:3, :]], dim=0)
        
        # 计算加权雅可比 (在 PhysX 格式下计算，因为 W_x 是按 PhysX 格式构建的)
        jac_w = self._W_x @ jac_physx @ self._W_q_inv
        # 转换为 Modern Robotics 约定
        jac_w_mr = torch.cat([jac_w[3:, :], jac_w[:3, :]], dim=0)
        
        # 原始雅可比 SVD 和可操作度
        S = torch.linalg.svdvals(jac)
        jt_j = torch.mm(jac.t(), jac)
        w = torch.sqrt(torch.abs(torch.det(jt_j))).item()
        sigma_max = S[0].item()
        sigma_min = S[-1].item()
        cond = sigma_max / sigma_min if sigma_min > 1e-6 else float('inf')
        
        # 加权雅可比 SVD 和可操作度
        S_w = torch.linalg.svdvals(jac_w_mr)
        jt_j_w = torch.mm(jac_w_mr.t(), jac_w_mr)
        w_w = torch.sqrt(torch.abs(torch.det(jt_j_w))).item()
        sigma_max_w = S_w[0].item()
        sigma_min_w = S_w[-1].item()
        cond_w = sigma_max_w / sigma_min_w if sigma_min_w > 1e-6 else float('inf')
        
        # 格式化打印设置
        np.set_printoptions(precision=6, suppress=True, linewidth=120)
        
        # 获取当前关节角度
        joint_pos = self.robot.data.joint_pos[0, joint_indices].cpu().numpy()
        joint_names = [self.robot.joint_names[i] for i in joint_indices]
        
        # 打印详细信息
        print("\n" + "="*35 + f" {finger_name} Finger Jacobian " + "="*35)
        print(f"Body Idx: {body_idx}, Jacobi Idx: {jacobi_idx}")
        print(f"Joint Indices: {joint_indices}")
        
        # --- 当前关节角度 ---
        print(f"\n[Current Joint Positions (rad)]")
        for i, (name, pos) in enumerate(zip(joint_names, joint_pos)):
            print(f"  {name}: {pos:.4f} rad ({np.degrees(pos):.2f}°)")
        
        # --- 权重配置 ---
        print(f"\n[Weight Configuration]")
        print(f"  W_x (task space): w_v = {self.W_V:.2f}, w_ω = {self.W_OMEGA:.2f}")
        print(f"  W_q (joint space): diag({self.W_Q_DIAG})")
        
        # --- 原始雅可比 (6x4) ---
        print(f"\n[Original J_b (6×4)] - Modern Robotics Convention:")
        print(f"  Row 0-2: Angular velocity [ωx, ωy, ωz]")
        print(f"  Row 3-5: Linear velocity  [vx, vy, vz]")
        print(jac.cpu().numpy())
        print(f"  Singular Values: {S.cpu().numpy()}")
        print(f"  Condition Number (κ): {cond:.4f}")
        print(f"  Manipulability (w): {w:.6f}")
        
        # --- 加权雅可比 (6x4) ---
        print(f"\n[Weighted J_w (6×4)] - J_w = W_x @ J_b @ W_q^{{-1}}:")
        print(f"  Row 0-2: Weighted angular velocity")
        print(f"  Row 3-5: Weighted linear velocity")
        print(jac_w_mr.cpu().numpy())
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
        print(f"  W_x (task space): w_v = {self.W_V:.2f}, w_ω = {self.W_OMEGA:.2f}")
        print(f"  W_q (joint space): diag({self.W_Q_DIAG})")
        print(f"  L_char = {self.L_CHAR} m")
        
        all_jacobians = self.robot.root_physx_view.get_jacobians()
        num_jacobi_bodies = all_jacobians.shape[1]
        
        # 收集所有手指数据
        finger_data_list = []
        
        for finger_name, data in self.fingers.items():
            jacobi_idx = data.get("jacobi_idx")
            joint_indices = data.get("joint_indices")
            
            if jacobi_idx is None or not joint_indices:
                continue
            
            if jacobi_idx < 0 or jacobi_idx >= num_jacobi_bodies:
                print(f"[WARNING] Jacobi index {jacobi_idx} out of bounds for finger {finger_name}")
                continue
            
            # 获取关节角度
            joint_pos = self.robot.data.joint_pos[0, joint_indices].cpu().numpy()
            joint_names = [self.robot.joint_names[i] for i in joint_indices]
            
            # 提取该手指的雅可比 (6, 4)
            jac_b = all_jacobians[0, jacobi_idx, :, :][:, joint_indices]
            
            # 计算加权雅可比
            jac_w = self._W_x @ jac_b @ self._W_q_inv
            
            # === 原始雅可比 ===
            S = torch.linalg.svdvals(jac_b)
            jt_j = torch.mm(jac_b.t(), jac_b)
            w = torch.sqrt(torch.abs(torch.det(jt_j))).item()
            sigma_max = S[0].item()
            sigma_min = S[-1].item()
            cond = sigma_max / sigma_min if sigma_min > 1e-6 else float('inf')
            
            # === 加权雅可比 ===
            S_w = torch.linalg.svdvals(jac_w)
            jt_j_w = torch.mm(jac_w.t(), jac_w)
            w_w = torch.sqrt(torch.abs(torch.det(jt_j_w))).item()
            sigma_max_w = S_w[0].item()
            sigma_min_w = S_w[-1].item()
            cond_w = sigma_max_w / sigma_min_w if sigma_min_w > 1e-6 else float('inf')
            
            finger_data_list.append({
                "name": finger_name,
                "joint_names": joint_names,
                "joint_pos": joint_pos,
                "w": w, "cond": cond, "sigma_min": sigma_min,
                "w_w": w_w, "cond_w": cond_w, "sigma_min_w": sigma_min_w
            })
        
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
        print(f"{'Finger':<8} | {'w(J_b)':<10} | {'κ(J_b)':<10} | {'σ_min':<8} | {'w(J_w)':<10} | {'κ(J_w)':<10} | {'σ_min_w':<8}")
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
