# 这里是本项目所用到的一些数学计算相关的工具函数和类的集合文件。主要和se(3)有关
# 实现风格仿照 source/isaaclab/isaaclab/utils/math.py, 但更专注于se(3)相关的计算需求，两者互补共同支持本项目
# 数学特点：原生更支持矩阵与旋量表示的运算，符合机器人经典著作Modern Robotics的李代数和旋量表示的主线，尽量避免欧拉角、四元数方面的参与。
# needed to import for allowing type-hinting: torch.Tensor | np.ndarray
from __future__ import annotations

import logging
import math
import numpy as np
import torch
import torch.nn.functional
from typing import Literal

from isaaclab.utils import math as math_utils

# import logger
logger = logging.getLogger(__name__)


"""
Jacobian inverse operations.
"""


@torch.jit.script
def pseudo_inv(J: torch.Tensor) -> torch.Tensor:
    r"""计算雅可比矩阵的 Moore-Penrose 伪逆。

    使用 PyTorch 的 SVD 分解计算伪逆，适用于一般情况的逆运动学求解。

    Args:
        J: 雅可比矩阵。形状为 ``(N, 6, num_joints)`` 或 ``(6, num_joints)``。

    Returns:
        雅可比伪逆矩阵。形状为 ``(N, num_joints, 6)`` 或 ``(num_joints, 6)``。

    Note:
        rcond=1e-5 是截断阈值，防止极小奇异值导致数值爆炸。
        该值在 IsaacLab 场景下通常够用。

    Reference:
        - Modern Robotics, Section 6.1: Inverse Kinematics
        - https://pytorch.org/docs/stable/generated/torch.linalg.pinv.html
    """
    return torch.linalg.pinv(J, rcond=1e-5)  # 如果一个方向上的运动能力（奇异值 σ_i）太弱，pinv 会直接切断该方向的控制，认为该方向不可控


@torch.jit.script
def dls_cholesky_inv(J: torch.Tensor, damping: float) -> torch.Tensor:
    r"""计算雅可比矩阵的阻尼最小二乘(DLS)伪逆（Cholesky 加速版本）。

    利用 Cholesky 分解求解线性方程组，避免显式求逆，速度更快且数值更稳定。

    **为什么不用 torch.linalg.inv？**

    1. **数值稳定性**：直接求逆容易放大浮点数误差，特别是当矩阵接近奇异时
    2. **计算效率**：对于 SPD 矩阵，Cholesky 分解 + 前后代求解 (O(n³/3)) 比直接求逆 (O(n³)) 更快
    3. **最优算法**：Cholesky 是求解 SPD 线性系统的黄金标准

    **数学推导:**

    目标：计算 :math:`J_{dls}^{\dagger} = J^T (JJ^T + \lambda^2 I)^{-1}`

    令 :math:`A = JJ^T + \lambda^2 I`，则 A 是 SPD 矩阵。

    我们需要求解：:math:`X = J^T A^{-1}`

    等价于求解线性方程组：:math:`A X^T = J \Rightarrow X^T = A^{-1} J`

    然后转置得到结果。

    **实现步骤:**

    1. 计算 :math:`A = JJ^T + \lambda^2 I`
    2. Cholesky 分解：:math:`A = LL^T`
    3. 求解 :math:`A Y = J`，得到 :math:`Y = A^{-1} J`
    4. 返回 :math:`J_{dls}^{\dagger} = Y^T`

    Args:
        J: 雅可比矩阵。形状为 ``(N, 6, num_joints)`` 或 ``(6, num_joints)``。
        damping: 阻尼系数 :math:`\lambda`。

    Returns:
        DLS 伪逆矩阵。形状为 ``(N, num_joints, 6)`` 或 ``(num_joints, 6)``。

    Reference:
        - Wampler, C. W. (1986). "Manipulator Inverse Kinematic Solutions Based on Vector
          Formulations and Damped Least-Squares Methods"
        - Modern Robotics, Section 6.2: Numerical Inverse Kinematics
        - Golub & Van Loan (2013). "Matrix Computations", Chapter 4: The Cholesky Decomposition
    """
    # 1. 计算 A = JJ^T + lambda^2 I
    # J shape: (..., 6, num_joints)
    J_T = J.transpose(-2, -1)
    A = torch.matmul(J, J_T)  # (..., 6, 6)

    # 2. 添加阻尼项到对角线
    # 原地操作，节省显存
    A.diagonal(dim1=-2, dim2=-1).add_(damping**2)

    # 3. Cholesky 分解: A = L * L^T
    L = torch.linalg.cholesky(A)

    # 4. 求解 A * Y = J，即 Y = A^{-1} * J
    # cholesky_solve 接受 (B, L)，求解 A * X = B
    Y = torch.cholesky_solve(J, L)

    # 5. 返回 J_dls^+ = Y^T
    return Y.transpose(-2, -1)


@torch.jit.script
def dls_svd_inv(
    J: torch.Tensor | None = None,
    damping=None,
    U: torch.Tensor | None = None,
    S: torch.Tensor | None = None,
    Vh: torch.Tensor | None = None,
    singular_threshold: float | None = None,
    selective: bool = False,
) -> torch.Tensor:
    r"""计算雅可比矩阵的阻尼最小二乘(DLS)伪逆（SVD 版本）。

    使用 SVD 分解求解，相比 Cholesky 分解具有极高的数值稳定性，
    且能正确处理奇异值阻尼。支持两种阻尼模式和两种调用方式。

    **数学原理:**

    目标：计算 :math:`J_{dls}^{\dagger} = J^T (JJ^T + \lambda^2 I)^{-1}`

    通过 SVD 分解 :math:`J = U \Sigma V^T`，可得：

    .. math::

        J_{dls}^{\dagger} = V \Lambda U^T

    **阻尼模式:**

    1. **全局阻尼 (selective=False)**:
       所有奇异值共用一个阻尼系数 :math:`\lambda`:

       .. math::

           \Lambda_{ii} = \frac{\sigma_i}{\sigma_i^2 + \lambda^2}

    2. **选择性阻尼 (selective=True)**:
       对每个奇异值独立判断是否需要阻尼:

       .. math::

           \lambda_i = \begin{cases}
           0 & \text{if } \sigma_i \ge \epsilon \\
           \lambda_{max} \left(1 - \frac{\sigma_i}{\epsilon}\right)^2 & \text{if } \sigma_i < \epsilon
           \end{cases}

       然后计算:

       .. math::

           \Lambda_{ii} = \frac{\sigma_i}{\sigma_i^2 + \lambda_i^2}

    **调用方式:**

    1. **传入雅可比矩阵 J**: 函数内部进行 SVD 分解
    2. **传入预计算的 SVD 结果 (U, S, Vh)**: 跳过 SVD 分解，避免重复计算

    Args:
        J: 雅可比矩阵。形状为 ``(N, 6, num_joints)`` 或 ``(6, num_joints)``。
           如果提供了 U, S, Vh，则此参数可为 None。
        damping: 阻尼系数 :math:`\lambda` 或 :math:`\lambda_{max}`。可以是标量或张量。
        U: SVD 分解的左奇异向量矩阵。形状为 ``(..., 6, K)``，K = min(6, num_joints)。
        S: SVD 分解的奇异值向量。形状为 ``(..., K)``。
        Vh: SVD 分解的右奇异向量矩阵的共轭转置。形状为 ``(..., K, num_joints)``。
        singular_threshold: 奇异值阈值 :math:`\epsilon`。仅在 selective=True 时使用。
        selective: 是否启用选择性阻尼。

    Returns:
        DLS 伪逆矩阵。形状为 ``(N, num_joints, 6)`` 或 ``(num_joints, 6)``。

    Raises:
        ValueError: 当参数组合无效时（如未提供 J 也未提供 SVD 结果）。
        ValueError: 当 selective=True 但未提供 singular_threshold 时。

    Examples:
        >>> # 方式1: 传入雅可比矩阵，全局阻尼
        >>> J = torch.randn(100, 6, 4)
        >>> J_inv = dls_inv(J, damping=0.01)
        
        >>> # 方式2: 传入雅可比矩阵，选择性阻尼
        >>> J_inv = dls_inv(J, damping=0.1, singular_threshold=0.05, selective=True)
        
        >>> # 方式3: 传入预计算的 SVD 结果，避免重复计算
        >>> U, S, Vh = torch.linalg.svd(J, full_matrices=False)
        >>> J_inv = dls_inv(U=U, S=S, Vh=Vh, damping=0.01)

    Reference:
        - Nakamura & Hanafusa (1986). "Inverse Kinematic Solutions With Singularity Robustness"
        - Buss & Kim (2004). "Selectively Damped Least Squares for Inverse Kinematics"
        - Modern Robotics, Section 6.2
    """
    # 1. 验证参数并获取 SVD 结果
    if U is None or S is None or Vh is None:
        if J is None:
            raise ValueError("必须提供 J 或 (U, S, Vh) 中的至少一组参数")
        # 执行 SVD 分解
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)
    
    # 2. 验证选择性阻尼参数
    if selective and singular_threshold is None:
        raise ValueError("启用选择性阻尼 (selective=True) 时必须提供 singular_threshold")
    
    # 3. 计算阻尼后的奇异值倒数
    if selective and singular_threshold is not None:
        # 选择性阻尼模式: 为每个奇异值独立计算阻尼系数
        # lambda_i = lambda_max * (1 - sigma_i / epsilon)^2  if sigma_i < epsilon
        #          = 0                                       if sigma_i >= epsilon
        
        # 处理 damping 的形状
        if isinstance(damping, (int, float)):
            lambda_max = damping
        else:
            lambda_max = damping
            # 如果是批量阻尼，需要广播到奇异值的形状
            if isinstance(lambda_max, torch.Tensor) and lambda_max.dim() > 0:
                lambda_max = lambda_max.unsqueeze(-1)  # (..., 1)
        
        # 计算每个奇异值的独立阻尼系数
        ratio = torch.clamp(S / singular_threshold, min=0.0, max=1.0)  # (..., K)
        lambda_i = torch.where(
            S >= singular_threshold,
            torch.zeros_like(S),  # 健康方向: λ_i = 0
            lambda_max * (1.0 - ratio) ** 2  # 奇异方向: 自适应阻尼
        )
        
        # 计算阻尼后的奇异值倒数
        S_dls = S / (S**2 + lambda_i**2)
    else:
        # 全局阻尼模式: 所有奇异值共用一个阻尼系数
        if isinstance(damping, (int, float)):
            lam_sq = damping ** 2
        else:
            lam_sq = damping ** 2
            if isinstance(lam_sq, torch.Tensor) and lam_sq.dim() > 0:
                # 广播到 S 的形状 (..., 1)
                lam_sq = lam_sq.unsqueeze(-1)
        
        S_dls = S / (S**2 + lam_sq)
    
    # 4. 重构伪逆
    # J_dls^+ = V @ diag(S_dls) @ U^T
    # V = Vh.mH (实数矩阵即转置)
    V = Vh.transpose(-2, -1)
    U_T = U.transpose(-2, -1)
    
    # 利用广播机制进行矩阵乘法
    # V: (..., num_joints, K), S_dls: (..., K), U_T: (..., K, 6)
    # 其中 K = min(6, num_joints)
    
    # 先将 S_dls 乘到 V 的列上 (相当于右乘对角矩阵)
    J_inv = (V * S_dls.unsqueeze(-2)) @ U_T
    
    return J_inv


def dls_inv(
    J: torch.Tensor | None = None,
    damping: float | None = None,
    method: Literal["cholesky", "svd"] = "cholesky",
    U: torch.Tensor | None = None,
    S: torch.Tensor | None = None,
    Vh: torch.Tensor | None = None,
    singular_threshold: float | None = None,
    selective: bool = False,
) -> torch.Tensor:
    r"""计算雅可比矩阵的阻尼最小二乘(DLS)伪逆（统一接口）。
    
    根据 method 参数选择不同的实现方式：
    
    - **cholesky**: 使用 Cholesky 分解，速度快，适合一般场景（默认）
    - **svd**: 使用 SVD 分解，数值稳定性最高，支持选择性阻尼
    
    Args:
        J: 雅可比矩阵。形状为 ``(N, 6, num_joints)`` 或 ``(6, num_joints)``。
           当 method="svd" 且提供了 (U, S, Vh) 时可为 None。
        damping: 阻尼系数 :math:`\lambda`。
        method: 计算方法，"cholesky" 或 "svd"。默认 "cholesky"。
        U: SVD 分解的左奇异向量矩阵（仅 svd 方法）。
        S: SVD 分解的奇异值向量（仅 svd 方法）。
        Vh: SVD 分解的右奇异向量矩阵的共轭转置（仅 svd 方法）。
        singular_threshold: 奇异值阈值（仅 svd + selective 时使用）。
        selective: 是否启用选择性阻尼（仅 svd 方法支持）。
    
    Returns:
        DLS 伪逆矩阵。形状为 ``(N, num_joints, 6)`` 或 ``(num_joints, 6)``。
    
    Raises:
        ValueError: 当 method 参数无效，或 cholesky 方法尝试使用 selective 时。
    
    Examples:
        >>> # 默认使用 Cholesky 方法（最快）
        >>> J_inv = dls_inv(J, damping=0.01)
        
        >>> # 使用 SVD 方法（数值稳定性最高）
        >>> J_inv = dls_inv(J, damping=0.01, method="svd")
        
        >>> # 使用选择性阻尼（仅 SVD 支持）
        >>> J_inv = dls_inv(J, damping=0.1, method="svd", 
        ...                 singular_threshold=0.05, selective=True)
        
        >>> # 传入预计算的 SVD 结果
        >>> U, S, Vh = torch.linalg.svd(J, full_matrices=False)
        >>> J_inv = dls_inv(damping=0.1, method="svd", U=U, S=S, Vh=Vh,
        ...                 singular_threshold=0.05, selective=True)
    """
    if method == "cholesky":
        if selective:
            raise ValueError(
                "Cholesky 方法不支持选择性阻尼。请使用 method='svd' 并设置 selective=True。"
            )
        if J is None or damping is None:
            raise ValueError("Cholesky 方法需要提供 J 和 damping 参数。")
        return dls_cholesky_inv(J, damping)
    elif method == "svd":
        return dls_svd_inv(
            J=J,
            damping=damping,
            U=U,
            S=S,
            Vh=Vh,
            singular_threshold=singular_threshold,
            selective=selective,
        )
    else:
        raise ValueError(
            f"未知的 DLS 方法: '{method}'。支持的方法: 'cholesky', 'svd'。"
        )


"""
SE(3) transformations and Adjoint map.
"""


@torch.jit.script
def skew_symmetric(vec: torch.Tensor) -> torch.Tensor:
    r"""从 3D 向量构造反对称矩阵（skew-symmetric matrix）。

    给定向量 :math:`v = [v_x, v_y, v_z]^T`，构造其反对称矩阵：

    .. math::

        [v] = \begin{bmatrix}
        0 & -v_z & v_y \\
        v_z & 0 & -v_x \\
        -v_y & v_x & 0
        \end{bmatrix}

    该矩阵满足性质：:math:`[v] x = v \times x`（叉乘）

    Args:
        vec: 输入 3D 向量。形状为 ``(..., 3)``。

    Returns:
        反对称矩阵。形状为 ``(..., 3, 3)``。

    Note:
        该函数支持批处理，可以同时处理多个向量。
        实现使用显式展开以确保 JIT 编译友好。
    """
    # 提取各分量
    x = vec[..., 0]
    y = vec[..., 1]
    z_val = vec[..., 2]

    # 创建零张量（用于构造矩阵）
    batch_shape = vec.shape[:-1]
    zero = torch.zeros(batch_shape, device=vec.device, dtype=vec.dtype)

    # 构造三行
    # row0: [0, -z, y]
    row0 = torch.stack([zero, -z_val, y], dim=-1)
    # row1: [z, 0, -x]
    row1 = torch.stack([z_val, zero, -x], dim=-1)
    # row2: [-y, x, 0]
    row2 = torch.stack([-y, x, zero], dim=-1)

    # 堆叠成 3x3 矩阵
    return torch.stack([row0, row1, row2], dim=-2)


@torch.jit.script
def adjoint_transform(T: torch.Tensor) -> torch.Tensor:
    r"""计算 SE(3) 齐次变换矩阵的伴随变换矩阵。

    **重要定义说明:**

    本实现针对的旋量定义为 :math:`\mathcal{V} = [\omega^T, v^T]^T`，
    即**角速度在前，线速度在后**。

    对于这种定义，伴随矩阵的正确形式为：

    .. math::

        \text{Ad}_T = \begin{bmatrix} R & 0 \\ [p]R & R \end{bmatrix} \in \mathbb{R}^{6 \times 6}

    其中 :math:`[p]` 是向量 :math:`p` 的反对称矩阵。

    **注意:**
    如果旋量定义为 :math:`\mathcal{V} = [v^T, \omega^T]^T`（线速度在前），
    或者是力旋量（wrench），则伴随矩阵的形式会不同（:math:`[p]R` 会在右上角）。

    **旋量变换关系:**

    .. math::

        \mathcal{V}_a = \text{Ad}_{T_{ab}} \mathcal{V}_b

    这意味着在坐标系 {b} 下的旋量 :math:`\mathcal{V}_b`，可以通过伴随变换
    转换为在坐标系 {a} 下的旋量 :math:`\mathcal{V}_a`。

    Args:
        T: SE(3) 齐次变换矩阵。形状为 ``(N, 4, 4)`` 或 ``(4, 4)``。

    Returns:
        伴随变换矩阵。形状为 ``(N, 6, 6)`` 或 ``(6, 6)``。

    Reference:
        - Modern Robotics, Section 3.3.2: The Adjoint Representation
        - Murray, Li, Sastry (1994). "A Mathematical Introduction to Robotic Manipulation"
    """
    # 提取旋转矩阵 R 和平移向量 p
    R = T[..., :3, :3]  # (..., 3, 3)
    p = T[..., :3, 3]  # (..., 3)

    # 构造 p 的反对称矩阵 [p]
    p_skew = skew_symmetric(p)  # (..., 3, 3)

    # 计算 [p] * R
    p_skew_R = p_skew @ R  # (..., 3, 3)

    # 构造 6x6 伴随矩阵
    # Ad_T = [[ R,    0   ],
    #         [ [p]R, R   ]]
    batch_shape = T.shape[:-2]
    Ad = torch.zeros(batch_shape + (6, 6), device=T.device, dtype=T.dtype)

    # 填充块
    # 左上: R (角速度部分到角速度部分)
    Ad[..., :3, :3] = R
    # 右下: R (线速度部分到线速度部分)
    Ad[..., 3:, 3:] = R
    # 左下: [p]R (角速度部分到线速度部分的耦合)
    Ad[..., 3:, :3] = p_skew_R
    # 右上: 0 (线速度部分不影响角速度部分)
    # (已经初始化为零，无需额外设置)

    return Ad


def transform_from_pos_quat(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    r"""根据位姿构造齐次变换矩阵。

    Args:
        pos: 位置向量，形状 ``(..., 3)``。
        quat: 四元数 ``(w, x, y, z)``，形状 ``(..., 4)``。

    Returns:
        齐次变换矩阵，形状 ``(..., 4, 4)``，表示从局部坐标系到世界坐标系的变换。
    """

    rot = math_utils.matrix_from_quat(quat)
    batch_shape = pos.shape[:-1]
    T = torch.zeros(batch_shape + (4, 4), device=pos.device, dtype=pos.dtype)
    T[..., :3, :3] = rot
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1.0
    return T


def inverse_transform(T: torch.Tensor) -> torch.Tensor:
    r"""计算 SE(3) 变换的逆。

    Args:
        T: 齐次变换矩阵，形状 ``(..., 4, 4)``。

    Returns:
        逆变换矩阵，形状 ``(..., 4, 4)``。
    """

    R = T[..., :3, :3]
    p = T[..., :3, 3]
    R_T = R.transpose(-1, -2)
    p_inv = -(R_T @ p.unsqueeze(-1)).squeeze(-1)

    T_inv = torch.zeros_like(T)
    T_inv[..., :3, :3] = R_T
    T_inv[..., :3, 3] = p_inv
    T_inv[..., 3, 3] = 1.0
    return T_inv

'''
Mathematical measures related to manipulability.
'''

def manipulability(J: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    r"""计算 Yoshikawa 操作度指标。

    Args:
        J: 几何雅可比矩阵，形状 ``(..., m, n)``。
        eps: 数值稳定用的下限，默认 ``1e-12``。

    Returns:
        操作度值，形状 ``(...)``。
    """

    if J.dim() < 2:
        raise ValueError("Jacobian must have at least 2 dimensions (m, n).")

    gram = (
        J.transpose(-2, -1) @ J
        if J.shape[-2] >= J.shape[-1]
        else J @ J.transpose(-2, -1)
    )

    det = torch.linalg.det(gram)
    det = torch.clamp(det, min=0.0)
    return torch.sqrt(det + eps)


def condition_number(J: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    r"""计算雅可比矩阵的 2-范数条件数。"""

    S = torch.linalg.svdvals(J)
    sigma_max = S[..., 0]
    sigma_min = torch.clamp(S[..., -1], min=eps)
    return sigma_max / sigma_min


def svd(J: torch.Tensor, full_matrices: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""对雅可比矩阵执行奇异值分解。

    Args:
        J: 输入矩阵 ``(..., m, n)``，支持批量。
        full_matrices: 是否返回完整的 U/Vh。默认 ``False`` 获取经济型分解。

    Returns:
        ``(U, S, Vh)``：分别为左奇异向量、奇异值、右奇异向量转置，形状与 ``torch.linalg.svd`` 一致。
    """

    return torch.linalg.svd(J, full_matrices=full_matrices)