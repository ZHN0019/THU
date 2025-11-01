import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# 全局仿真参数定义
# =============================================================================

# 仿真参数
MESH_LENGTH = 0.05          # 网格的单元长度
SUB_STEP = 5                # 每一帧的子步数
DT = 0.001

# 布料物理参数
DEFAULT_STIFFNESS_STRETCH_TENSION = 1000.0      # 拉伸弹簧拉伸刚度 (N/m)        织物的经纬线方向（主要限制拉伸
DEFAULT_STIFFNESS_SHEAR = 800.0                 # 剪切弹簧刚度 (N/m)           织物在经纬线对角线方向（主要限制对角拉伸
DEFAULT_STIFFNESS_BEND_TENSION = 50.0           # 弯曲弹簧拉伸刚度 (N/m)        织物的柔软程度（主要限制下塌成都
DEFAULT_COMPRESSION_TO_TENSION_RATIO = 0.1      # 拉伸与弯曲弹簧的 压缩/拉伸刚度 的比值，描述弹簧力各向异性的参数

# 阻尼参数
DEFAULT_DAMPING = 0.997                # 速度阻尼系数 (每步衰减)
DEFAULT_SPRING_DAMPING_COEFF = 0.3    # 弹簧阻尼系数

# 碰撞参数
DEFAULT_MIN_DISTANCE = 0.02           # 粒子间最小距离 (m)
DEFAULT_MOMENTUM_LOSS = 0.4           # 动量损失系数
DEFAULT_COLLISION_RESTITUTION = 0.3   # 碰撞恢复系数

# 重力参数
GRAVITY = np.array([0, 0, -9.81])     # 重力加速度 (m/s²)

class SoftClothSimulator:
    """
    柔性布料仿真器 - 基于质点-弹簧模型，类似MuJoCo中的布料仿真方式
    """
    
    def __init__(self, width, height, mesh_length,
                 stiffness_stretch_tension=DEFAULT_STIFFNESS_STRETCH_TENSION,
                 stiffness_bend_tension=DEFAULT_STIFFNESS_BEND_TENSION,
                 stiffness_shear=DEFAULT_STIFFNESS_SHEAR,
                 compression_to_tension_ratio=DEFAULT_COMPRESSION_TO_TENSION_RATIO,
                 damping=DEFAULT_DAMPING,
                 spring_damping_coeff=DEFAULT_SPRING_DAMPING_COEFF,
                 min_distance=DEFAULT_MIN_DISTANCE,
                 momentum_loss=DEFAULT_MOMENTUM_LOSS,
                 collision_restitution=DEFAULT_COLLISION_RESTITUTION,
                 substep = SUB_STEP):
        """
        初始化柔性布料仿真器
        
        Parameters:
        - width, height: 布料尺寸 (米)
        - nx, ny: 布料在x和y方向的粒子数
        - mass_per_particle: 每个粒子的质量
        - stiffness_stretch_tension: 拉伸弹簧拉伸刚度
        - stiffness_bend_tension: 弯曲弹簧拉伸刚度
        - stiffness_shear: 剪切弹簧刚度
        - compression_to_tension_ratio: 压缩/拉伸刚度比值
        - damping: 速度阻尼系数
        - spring_damping_coeff: 弹簧阻尼系数
        - min_distance: 粒子间最小距离
        - momentum_loss: 动量损失系数
        - collision_restitution: 碰撞恢复系数
        """
        self.nx = int(width / mesh_length)
        self.ny = int(height / mesh_length)
        self.width = width
        self.height = height
        self.suofang_fact = mesh_length / 0.05       # 当前仿真的参数是基于 1m 的来调试的，这个尺度变化会有所影响，在这里进行一个缩放
        self.substep = substep
        
        # 物理参数
        self.mass_per_particle = 0.01 * self.suofang_fact
        self.stiffness_stretch_tension = stiffness_stretch_tension * self.suofang_fact
        self.stiffness_stretch_compression = stiffness_stretch_tension * compression_to_tension_ratio * self.suofang_fact
        self.stiffness_bend_tension = stiffness_bend_tension * self.suofang_fact
        self.stiffness_bend_compression = stiffness_bend_tension * compression_to_tension_ratio * self.suofang_fact
        self.stiffness_shear = stiffness_shear * self.suofang_fact
        self.compression_to_tension_ratio = compression_to_tension_ratio
        
        # 阻尼参数
        self.damping = damping
        self.spring_damping_coeff = spring_damping_coeff  * self.suofang_fact
        
        # 碰撞参数
        self.min_distance = min_distance
        self.momentum_loss = momentum_loss
        self.collision_restitution = collision_restitution
        
        # 重力
        self.gravity = GRAVITY
        
        # 初始化粒子位置 (均匀分布在x-y平面上)
        self.positions = np.zeros((self.nx, self.ny, 3))
        self.velocities = np.zeros((self.nx, self.ny, 3))
        self.forces = np.zeros((self.nx, self.ny, 3))
        
        dx = mesh_length
        dy = mesh_length
        
        for i in range(self.nx):
            for j in range(self.ny):
                self.positions[i, j] = np.array([i * dx, j * dy, 0.0])
        
        # 初始化弹簧连接
        self._init_springs()
        
        # 固定点标记 (用于固定布料的某些点)
        self.fixed_points = np.zeros((self.nx, self.ny), dtype=bool)
        
    def _init_springs(self):
        """
        初始化三种类型的弹簧:
        1. 结构弹簧 (拉伸/压缩)
        2. 剪切弹簧
        3. 弯曲弹簧
        """
        self.springs = []
        
        # 结构弹簧 (相邻粒子之间)
        for i in range(self.nx):
            for j in range(self.ny):
                # 与右侧粒子的连接
                if i < self.nx - 1:
                    self.springs.append({
                        'type': 'stretch',
                        'p1': (i, j),
                        'p2': (i+1, j),
                        'rest_length': np.linalg.norm(
                            self.positions[i, j] - self.positions[i+1, j]
                        )
                    })
                
                # 与下方粒子的连接
                if j < self.ny - 1:
                    self.springs.append({
                        'type': 'stretch',
                        'p1': (i, j),
                        'p2': (i, j+1),
                        'rest_length': np.linalg.norm(
                            self.positions[i, j] - self.positions[i, j+1]
                        )
                    })
        
        # 剪切弹簧 (对角线连接)
        for i in range(self.nx - 1):
            for j in range(self.ny - 1):
                # 右下对角线
                self.springs.append({
                    'type': 'shear',
                    'p1': (i, j),
                    'p2': (i+1, j+1),
                    'rest_length': np.linalg.norm(
                        self.positions[i, j] - self.positions[i+1, j+1]
                    )
                })
                
                # 左下对角线
                self.springs.append({
                    'type': 'shear',
                    'p1': (i+1, j),
                    'p2': (i, j+1),
                    'rest_length': np.linalg.norm(
                        self.positions[i+1, j] - self.positions[i, j+1]
                    )
                })
        
        # 弯曲弹簧 (跳过一个粒子的连接)
        for i in range(self.nx):
            for j in range(self.ny):
                # 水平方向弯曲弹簧
                if i < self.nx - 2:
                    self.springs.append({
                        'type': 'bend',
                        'p1': (i, j),
                        'p2': (i+2, j),
                        'rest_length': np.linalg.norm(
                            self.positions[i, j] - self.positions[i+2, j]
                        )
                    })
                
                # 垂直方向弯曲弹簧
                if j < self.ny - 2:
                    self.springs.append({
                        'type': 'bend',
                        'p1': (i, j),
                        'p2': (i, j+2),
                        'rest_length': np.linalg.norm(
                            self.positions[i, j] - self.positions[i, j+2]
                        )
                    })
    
    def fix_point(self, i, j):
        """
        固定指定位置的粒子
        
        Parameters:
        - i, j: 粒子索引
        """
        self.fixed_points[i, j] = True
    
    def reset_forces(self):
        """重置所有粒子上的力"""
        self.forces.fill(0.0)
        
        # 应加重力
        for i in range(self.nx):
            for j in range(self.ny):
                self.forces[i, j] += self.gravity * self.mass_per_particle
    
    def compute_spring_forces(self):
        """计算所有弹簧产生的力（含阻尼）"""
        for spring in self.springs:
            p1_idx = spring['p1']
            p2_idx = spring['p2']
            p1 = self.positions[p1_idx]
            p2 = self.positions[p2_idx]
            v1 = self.velocities[p1_idx]
            v2 = self.velocities[p2_idx]

            delta = p2 - p1
            dist = np.linalg.norm(delta)
            if dist == 0:
                continue
            direction = delta / dist
            rest_length = spring['rest_length']
            displacement = dist - rest_length

            # 选择弹簧刚度（根据弹簧类型和拉伸/压缩状态）
            if spring['type'] == 'stretch':
                if displacement >= 0:  # 拉伸
                    stiffness = self.stiffness_stretch_tension
                else:  # 压缩
                    stiffness = self.stiffness_stretch_compression
            elif spring['type'] == 'shear':
                stiffness = self.stiffness_shear
            else:  # bend
                if displacement >= 0:  # 拉伸
                    stiffness = self.stiffness_bend_tension
                else:  # 压缩
                    stiffness = self.stiffness_bend_compression

            # 弹性力
            force = stiffness * displacement * direction

            # 阻尼力（沿弹簧方向）
            relative_velocity = v2 - v1
            damping_force = self.spring_damping_coeff * np.dot(relative_velocity, direction) * direction

            total_force = force + damping_force

            # 应用到粒子上
            if not self.fixed_points[p1_idx]:
                self.forces[p1_idx] += total_force
            if not self.fixed_points[p2_idx]:
                self.forces[p2_idx] -= total_force

    def integrate(self, dt):
        """
        使用显式欧拉法进行时间积分
        
        Parameters:
        - dt: 时间步长
        """
        # 计算所有力
        self.reset_forces()
        self.compute_spring_forces()
        
        # 更新速度和位置
        for i in range(self.nx):
            for j in range(self.ny):
                if not self.fixed_points[i, j]:
                    # 加速度 = 力 / 质量
                    acceleration = self.forces[i, j] / self.mass_per_particle
                    
                    # 更新速度 (带阻尼)
                    self.velocities[i, j] += acceleration * dt
                    self.velocities[i, j] *= self.damping
                    
                    # 更新位置
                    self.positions[i, j] += self.velocities[i, j] * dt
    
        # 防止布料自相交
        self.handle_self_collision(min_distance=self.min_distance)

    def simulate(self, steps, dt):
        """
        运行仿真
        
        Parameters:
        - steps: 仿真步数
        - dt: 时间步长
        """
        for _ in range(steps):
            self.integrate(dt*self.suofang_fact)
    
    def get_vertices(self):
        """获取所有粒子的位置用于可视化"""
        return self.positions.reshape(-1, 3)
    
    def get_faces(self):
        """获取三角面片索引用于可视化"""
        faces = []
        for i in range(self.nx - 1):
            for j in range(self.ny - 1):
                # 每个四边形单元划分为两个三角形
                # 第一个三角形
                faces.append([
                    i * self.ny + j,
                    i * self.ny + (j + 1),
                    (i + 1) * self.ny + j
                ])
                # 第二个三角形
                faces.append([
                    (i + 1) * self.ny + j,
                    i * self.ny + (j + 1),
                    (i + 1) * self.ny + (j + 1)
                ])
        return np.array(faces)
    
    def handle_self_collision(self, min_distance=0.002):
        """
        优化版自碰撞检测：
        1. 使用空间哈希加速
        2. 先判定，后计算
        """
        cell_size = min_distance * self.suofang_fact
        grid = {}

        # ---------- 构建空间哈希 ----------
        for i in range(self.nx):
            for j in range(self.ny):
                key = tuple((self.positions[i, j] // cell_size).astype(int))
                if key not in grid:
                    grid[key] = []
                grid[key].append((i, j))

        # ---------- 检测碰撞 ----------
        for key, particles in grid.items():
            # 邻居格子（包含当前 + 周围 26 个）
            neighbors = [(dx, dy, dz) for dx in [-1, 0, 1]
                                        for dy in [-1, 0, 1]
                                        for dz in [-1, 0, 1]]

            for dx, dy, dz in neighbors:
                neighbor_key = (key[0] + dx, key[1] + dy, key[2] + dz)
                if neighbor_key not in grid:
                    continue
                for (i1, j1) in particles:
                    for (i2, j2) in grid[neighbor_key]:
                        if (i1, j1) >= (i2, j2):
                            continue  # 避免重复

                        p1 = self.positions[i1, j1]
                        p2 = self.positions[i2, j2]
                        delta = p2 - p1
                        dist = np.linalg.norm(delta)

                        # ---------- 快速剔除 ----------
                        if dist >= min_distance or dist < 1e-6:
                            continue
                        print("碰撞了")
                        n = delta / dist
                        m1 = m2 = self.mass_per_particle
                        v1 = self.velocities[i1, j1]
                        v2 = self.velocities[i2, j2]

                        # ---------- 1. 位置修正 ----------
                        correction = (min_distance - dist) * n
                        if not self.fixed_points[i1, j1]:
                            self.positions[i1, j1] -= correction * 0.5
                        if not self.fixed_points[i2, j2]:
                            self.positions[i2, j2] += correction * 0.5

                        # ---------- 2. 动量守恒 ----------
                        # ---------- 2. 动量守恒（带损失和弹性） ----------
                        v1n = np.dot(v1, n)
                        v2n = np.dot(v2, n)
                        rel_vel = v1n - v2n

                        if rel_vel < 0:  # 只有相向时才处理
                            # 冲量（含恢复系数）
                            j = -(1 + self.collision_restitution) * rel_vel / (1/m1 + 1/m2)

                            # 动量损失系数：整体缩小冲量
                            j *= self.momentum_loss  

                            impulse = j * n

                            if not self.fixed_points[i1, j1]:
                                self.velocities[i1, j1] += impulse / m1
                            if not self.fixed_points[i2, j2]:
                                self.velocities[i2, j2] -= impulse / m2

def create_cloth_demo():
    """
    创建一个布料仿真实例
    """
    # 创建布料仿真器 (1m x 1m, 20x20粒子)
    cloth = SoftClothSimulator(0.5, 0.5, MESH_LENGTH)
    
    # 固定布料的中心点
    # cloth.fix_point(10, 10)
    cloth.fix_point(5, 5)
    
    # 可视化设置
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置标题为英文以避免字体问题
    ax.set_title('Soft Cloth Simulation')
    
    # 初始化图形元素
    vertices = cloth.get_vertices()
    faces = cloth.get_faces()
    
    # 创建3D表面
    surf = ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=faces,
        color='lightblue',
        alpha=0.8,
        edgecolor='none'
    )
    
    # 设置坐标轴
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-0.5, 0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    def update(frame):
        # 每帧执行仿真步骤
        cloth.simulate(SUB_STEP, DT)  # 每帧执行5步，时间步长0.001秒
        
        # 更新表面
        vertices = cloth.get_vertices()
        
        # 清除旧图形并重新绘制
        ax.clear()
        
        # 重新绘制
        ax.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=faces,
            color='lightblue',
            alpha=0.8,
            edgecolor='none'
        )
        
        # 保持视角一致
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(-0.5, 0.5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Soft Cloth Simulation')
        
        return surf,
    
    # 创建动画
    ani = FuncAnimation(fig, update, frames=300, interval=5, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return cloth, ani

# 运行演示
if __name__ == "__main__":
    print("Starting soft cloth simulation...")
    cloth_sim, animation = create_cloth_demo()
    print("Simulation completed")