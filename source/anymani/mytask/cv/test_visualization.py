#!/usr/bin/env python3
"""
测试点云 vs RGB vs 深度图的可视化对比
从 rgb_vs_pc_investigate.ipynb 中提取的代码
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端,不显示窗口
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ============================================
# 1. 生成一个简单的3D立方体场景
# ============================================
def generate_cube_point_cloud(center, size, num_points_per_face=20):
    """
    生成立方体表面的点云
    
    Args:
        center: (x, y, z) 立方体中心
        size: 立方体边长
        num_points_per_face: 每个面采样的点数
    
    Returns:
        points: (N, 3) 点云坐标
        colors: (N, 3) 每个点的RGB颜色
    """
    cx, cy, cz = center
    half_size = size / 2
    
    points = []
    colors = []
    
    # 定义6个面及其颜色
    faces = [
        # (normal_axis, offset, color)
        ('x', half_size, [1, 0, 0]),   # 右面 - 红色
        ('x', -half_size, [0, 1, 1]),  # 左面 - 青色
        ('y', half_size, [0, 1, 0]),   # 前面 - 绿色
        ('y', -half_size, [1, 0, 1]),  # 后面 - 品红
        ('z', half_size, [0, 0, 1]),   # 上面 - 蓝色
        ('z', -half_size, [1, 1, 0]),  # 下面 - 黄色
    ]
    
    for axis, offset, color in faces:
        # 在每个面上均匀采样点
        u = np.linspace(-half_size, half_size, num_points_per_face)
        v = np.linspace(-half_size, half_size, num_points_per_face)
        uu, vv = np.meshgrid(u, v)
        
        if axis == 'x':
            x = np.full_like(uu, cx + offset)
            y = cy + uu
            z = cz + vv
        elif axis == 'y':
            x = cx + uu
            y = np.full_like(uu, cy + offset)
            z = cz + vv
        else:  # z
            x = cx + uu
            y = cy + vv
            z = np.full_like(uu, cz + offset)
        
        face_points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        face_colors = np.tile(color, (len(face_points), 1))
        
        points.append(face_points)
        colors.append(face_colors)
    
    return np.vstack(points), np.vstack(colors)

# 生成立方体点云
cube_points, cube_colors = generate_cube_point_cloud(
    center=(0, 0, 0.3),  # 立方体中心在(0, 0, 0.3)
    size=0.1,            # 边长10cm
    num_points_per_face=15
)

print(f"生成的点云: {cube_points.shape[0]} 个点")
print(f"点云数据量: {cube_points.shape[0] * 3} 个浮点数")

# ============================================
# 2. 模拟相机投影生成RGB图像和深度图
# ============================================
def project_points_to_image(points, colors, img_width=320, img_height=240, focal_length=300):
    """
    将3D点云投影到2D图像平面
    
    简化的针孔相机模型:
    u = fx * x / z + cx
    v = fy * y / z + cy
    """
    rgb_image = np.zeros((img_height, img_width, 3))
    depth_image = np.full((img_height, img_width), np.inf)
    
    cx, cy = img_width / 2, img_height / 2
    fx, fy = focal_length, focal_length
    
    for i, (point, color) in enumerate(zip(points, colors)):
        x, y, z = point
        
        if z <= 0:  # 跳过相机后面的点
            continue
        
        # 投影到图像平面
        u = int(fx * x / z + cx)
        v = int(fy * y / z + cy)
        
        # 检查是否在图像范围内
        if 0 <= u < img_width and 0 <= v < img_height:
            # Z-buffer: 只保留最近的点
            if z < depth_image[v, u]:
                rgb_image[v, u] = color
                depth_image[v, u] = z
    
    # 处理深度图中的无效值
    depth_image[depth_image == np.inf] = 0
    
    return rgb_image, depth_image

# 生成RGB和深度图
rgb_image, depth_image = project_points_to_image(cube_points, cube_colors)

print(f"\nRGB图像: {rgb_image.shape}")
print(f"RGB数据量: {rgb_image.size} 个浮点数")
print(f"\n深度图像: {depth_image.shape}")
print(f"深度数据量: {depth_image.size} 个浮点数")

# ============================================
# 3. 可视化对比
# ============================================
fig = plt.figure(figsize=(18, 5))

# 子图1: 3D点云
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(cube_points[:, 0], cube_points[:, 1], cube_points[:, 2],
           c=cube_colors, s=10, alpha=0.8)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.set_title('Point Cloud\nDirect 3D Representation', fontsize=14, fontweight='bold')
ax1.text2D(0.05, 0.95, f'Data: {cube_points.shape[0] * 3} values\nShape: ({cube_points.shape[0]}, 3)',
          transform=ax1.transAxes, fontsize=10, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 子图2: RGB图像
ax2 = fig.add_subplot(132)
ax2.imshow(rgb_image)
ax2.set_title('RGB Image\n2D Projection + Color', fontsize=14, fontweight='bold')
ax2.axis('off')
ax2.text(10, 20, f'Data: {rgb_image.size} values\nShape: {rgb_image.shape}',
        fontsize=10, color='white',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# 子图3: 深度图
ax3 = fig.add_subplot(133)
depth_display = ax3.imshow(depth_image, cmap='viridis')
ax3.set_title('Depth Image\n2D Projection + Distance', fontsize=14, fontweight='bold')
ax3.axis('off')
cbar = plt.colorbar(depth_display, ax=ax3, fraction=0.046, pad=0.04)
cbar.set_label('Distance (m)', rotation=270, labelpad=15)
ax3.text(10, 20, f'Data: {depth_image.size} values\nShape: {depth_image.shape}',
        fontsize=10, color='white',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

plt.tight_layout()
output_path = '/home/hac/isaac/AnyRotate/source/leaphand/mytask/cv/pointcloud_vs_rgb_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n图像已保存到: {output_path}")
plt.close()  # 关闭图形,不显示

# ============================================
# 4. 数据量对比总结
# ============================================
print("\n" + "="*60)
print("数据量对比总结:")
print("="*60)
print(f"点云:      {cube_points.shape[0]:>6} 个点 × 3 = {cube_points.shape[0]*3:>8} 个数值")
print(f"RGB图像:   {rgb_image.shape[0]:>6} × {rgb_image.shape[1]} × 3 = {rgb_image.size:>8} 个数值")
print(f"深度图像:  {depth_image.shape[0]:>6} × {depth_image.shape[1]}     = {depth_image.size:>8} 个数值")
print("="*60)
print(f"RGB图像数据量是点云的 {rgb_image.size / (cube_points.shape[0]*3):.1f} 倍")
print("="*60)
print("\n关键观察:")
print("• 点云: 数据量小,但直接表达3D几何")
print("• RGB: 数据量大,包含颜色但缺少深度")
print("• 深度图: 可以转换为点云,但受分辨率限制")
print("="*60)

