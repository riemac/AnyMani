1. 从urdf提取极简运动学树
2. 规模化生存资产的方法论（由我主导）(某种意义上相当于“数据清洗”这一机器学习领域的上游工作)
   - 难点之一: <inertial> 的计算（资产生成）
   > 在实际工程中，为了防止物理引擎（如 Gazebo, PyBullet）在计算微小物体时出现除以接近 0 的数值导致崩溃，开发者经常会故意把 inertial 的值写得比实际大一些（这被称为 Inertia Padding 惯性填充）
   - 坐标系语义对齐？主要是不同<joint>和child link的<collision>坐标系（网络架构）
3. recolored urdf区分不同的link，从base/palm的树根节点开始，同层级的link color mesh颜色相同