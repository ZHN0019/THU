import os
# 在其他导入之前设置 CUDA 路径
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ctypes
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvrtc64_120_0.dll', winmode=0)
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvrtc-builtins64_126.dll', winmode=0)

# 设置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号


# 如果可用可用GPU
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

GPU_AVAILABLE = False

# 仿真参数
MESH_LENGTH = 0.05
SUB_STEP = 5
DT = 0.001

# 布料物理参数
DEFAULT_STIFFNESS_STRETCH_TENSION = 1000.0
DEFAULT_STIFFNESS_SHEAR = 800.0
DEFAULT_STIFFNESS_BEND_TENSION = 250.0
DEFAULT_COMPRESSION_TO_TENSION_RATIO = 0.1

# 阻尼参数
# DEFAULT_DAMPING = 0.997
DEFAULT_DAMPING = 0.999
DEFAULT_SPRING_DAMPING_COEFF = 0.6

# 重力参数
GRAVITY = np.array([0, 0, -9.81])

class SoftClothSimulator:
    def __init__(self, Init_R, width, height, mesh_length,
                 stiffness_stretch_tension=DEFAULT_STIFFNESS_STRETCH_TENSION,
                 stiffness_bend_tension=DEFAULT_STIFFNESS_BEND_TENSION,
                 stiffness_shear=DEFAULT_STIFFNESS_SHEAR,
                 compression_to_tension_ratio=DEFAULT_COMPRESSION_TO_TENSION_RATIO,
                 damping=DEFAULT_DAMPING,
                 spring_damping_coeff=DEFAULT_SPRING_DAMPING_COEFF,
                 substep=SUB_STEP,
                 use_gpu=True):
        self.nx = int(width / mesh_length)
        self.ny = int(height / mesh_length)
        self.width = width
        self.height = height
        self.suofang_fact = mesh_length / 0.05
        self.substep = substep
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np

        # 物理参数
        self.mass_per_particle = 0.08 * self.suofang_fact
        self.stiffness_stretch_tension = stiffness_stretch_tension * self.suofang_fact
        self.stiffness_stretch_compression = stiffness_stretch_tension * compression_to_tension_ratio * self.suofang_fact
        self.stiffness_bend_tension = stiffness_bend_tension * self.suofang_fact
        self.stiffness_bend_compression = stiffness_bend_tension * compression_to_tension_ratio * self.suofang_fact
        self.stiffness_shear = stiffness_shear * self.suofang_fact
        self.compression_to_tension_ratio = compression_to_tension_ratio

        # 阻尼
        self.damping = damping
        self.spring_damping_coeff = spring_damping_coeff * self.suofang_fact

        # 重力
        self.gravity = self.xp.array(GRAVITY)

        # 初始化粒子
        self.positions = self.xp.zeros((self.nx, self.ny, 3))
        self.velocities = self.xp.zeros((self.nx, self.ny, 3))
        self.forces = self.xp.zeros((self.nx, self.ny, 3))
        dx = mesh_length
        dy = mesh_length
        for i in range(self.nx):
            for j in range(self.ny):
                self.positions[i, j] = self.xp.array([i * dx, j * dy, 0.0])
        # for i in range(self.nx):
        #     for j in range(self.ny):
        #         self.positions[i, j] = self.xp.array([
        #             i * dx * np.sin(j  * 2 * np.pi / self.ny),
        #             i * dx * np.cos(j  * 2 * np.pi / self.ny),
        #               0.0])

        # 弹簧
        self._init_springs()
        self._preallocate_arrays()

        # 固定点
        self.fixed_points = self.xp.zeros((self.nx, self.ny), dtype=bool)

        # 驱动点弹簧

        self.initial_positions = self.positions.copy()  # 保存每个粒子的初始位置
        self.driven_point_pos = self.xp.array([0.5, 0.5, 0.5])
        self.driven_point_stiffness_tension = 5000.0 * self.suofang_fact
        self.driven_point_stiffness_compression = 50.0 * self.suofang_fact
        self.driven_point_func = None  # 可通过函数动态更新驱动点位置
        # 驱动点自然长度数组 (每个粒子一个)
        # 驱动点初始位置
        self.driven_point_pos = self.xp.array([width*0.5+Init_R, height*0.5, 0])

        # 驱动点自然长度数组（仿真开始时计算）
        self.driven_rest_lengths = self.xp.linalg.norm(
            self.positions - self.driven_point_pos,
            axis=2
        )

        # 时间计数
        self.time = 0.0

    def fix_point(self, i, j):
        self.fixed_points[i, j] = True

    def reset_forces(self):
        self.forces.fill(0.0)
        gravity_force = self.gravity * self.mass_per_particle
        self.forces += gravity_force

    def _init_springs(self):
        self.springs = []
        # 结构弹簧
        for i in range(self.nx):
            for j in range(self.ny):
                if i < self.nx - 1:
                    self.springs.append({'type':'stretch','p1':(i,j),'p2':(i+1,j),'rest_length':float(self.xp.linalg.norm(self.positions[i,j]-self.positions[i+1,j]))})
                if j < self.ny -1:
                    self.springs.append({'type':'stretch','p1':(i,j),'p2':(i,j+1),'rest_length':float(self.xp.linalg.norm(self.positions[i,j]-self.positions[i,j+1]))})
        # 剪切弹簧
        for i in range(self.nx-1):
            for j in range(self.ny-1):
                self.springs.append({'type':'shear','p1':(i,j),'p2':(i+1,j+1),'rest_length':float(self.xp.linalg.norm(self.positions[i,j]-self.positions[i+1,j+1]))})
                self.springs.append({'type':'shear','p1':(i+1,j),'p2':(i,j+1),'rest_length':float(self.xp.linalg.norm(self.positions[i+1,j]-self.positions[i,j+1]))})
        # 弯曲弹簧
        for i in range(self.nx):
            for j in range(self.ny):
                if i < self.nx -2:
                    self.springs.append({'type':'bend','p1':(i,j),'p2':(i+2,j),'rest_length':float(self.xp.linalg.norm(self.positions[i,j]-self.positions[i+2,j]))})
                if j < self.ny -2:
                    self.springs.append({'type':'bend','p1':(i,j),'p2':(i,j+2),'rest_length':float(self.xp.linalg.norm(self.positions[i,j]-self.positions[i,j+2]))})

    def _preallocate_arrays(self):
        self.p1_indices = []
        self.p2_indices = []
        self.spring_types = []
        self.rest_lengths = []
        for spring in self.springs:
            self.p1_indices.append(spring['p1'])
            self.p2_indices.append(spring['p2'])
            self.spring_types.append(spring['type'])
            self.rest_lengths.append(spring['rest_length'])
        self.p1_indices = self.xp.array(self.p1_indices)
        self.p2_indices = self.xp.array(self.p2_indices)
        self.rest_lengths = self.xp.array(self.rest_lengths)

    def compute_spring_forces_vectorized(self):
        p1_pos = self.positions[self.p1_indices[:,0], self.p1_indices[:,1]]
        p2_pos = self.positions[self.p2_indices[:,0], self.p2_indices[:,1]]
        p1_vel = self.velocities[self.p1_indices[:,0], self.p1_indices[:,1]]
        p2_vel = self.velocities[self.p2_indices[:,0], self.p2_indices[:,1]]
        delta = p2_pos - p1_pos
        dist = self.xp.linalg.norm(delta, axis=1)
        dist = self.xp.where(dist==0,1e-10,dist)
        direction = delta / dist[:,self.xp.newaxis]
        displacement = dist - self.rest_lengths
        stiffness = self.xp.zeros(len(self.springs))
        stretch_mask = self.xp.array([t=='stretch' for t in self.spring_types])
        tension_mask = displacement>=0
        compression_mask = displacement<0
        stiffness = self.xp.where(stretch_mask & tension_mask, self.stiffness_stretch_tension, stiffness)
        stiffness = self.xp.where(stretch_mask & compression_mask, self.stiffness_stretch_compression, stiffness)
        shear_mask = self.xp.array([t=='shear' for t in self.spring_types])
        stiffness = self.xp.where(shear_mask, self.stiffness_shear, stiffness)
        bend_mask = self.xp.array([t=='bend' for t in self.spring_types])
        stiffness = self.xp.where(bend_mask & tension_mask, self.stiffness_bend_tension, stiffness)
        stiffness = self.xp.where(bend_mask & compression_mask, self.stiffness_bend_compression, stiffness)
        force_magnitude = stiffness * displacement
        elastic_force = direction * force_magnitude[:,self.xp.newaxis]
        relative_velocity = p2_vel - p1_vel
        velocity_dot = self.xp.sum(relative_velocity * direction, axis=1)
        damping_force = (self.spring_damping_coeff * velocity_dot[:,self.xp.newaxis] * direction)
        total_force = elastic_force + damping_force
        for i,(p1_idx,p2_idx) in enumerate(zip(self.p1_indices,self.p2_indices)):
            if not self.fixed_points[p1_idx[0],p1_idx[1]]:
                self.forces[p1_idx[0],p1_idx[1]] += total_force[i]
            if not self.fixed_points[p2_idx[0],p2_idx[1]]:
                self.forces[p2_idx[0],p2_idx[1]] -= total_force[i]

    def compute_driven_point_forces(self):
        # 更新驱动点位置
        if self.driven_point_func is not None:
            # 确保驱动点位置与当前使用的计算库一致 (CPU/GPU)
            driven_point_np = self.driven_point_func(self.time)
            self.driven_point_pos = self.xp.array(driven_point_np)  # 转换为正确的类型

        # 当前粒子到驱动点的向量
        delta = self.positions - self.driven_point_pos

        # 当前长度
        current_length = self.xp.linalg.norm(delta, axis=2)
        current_length = self.xp.where(current_length == 0, 1e-10, current_length)

        # 单位方向
        direction = delta / current_length[:, :, self.xp.newaxis]

        # 重新计算每个粒子到驱动点的"自然长度"
        # 自然长度 = 驱动点当前位置 与 初始粒子位置之间的距离
        self.driven_rest_lengths = self.xp.linalg.norm(self.initial_positions - self.driven_point_pos, axis=2)

        # 位移 = 当前长度 - 自然长度
        displacement = current_length - self.driven_rest_lengths

        # 各向异性刚度
        tension_mask = displacement >= 0
        stiffness = self.xp.where(
            tension_mask,
            self.driven_point_stiffness_tension,
            self.driven_point_stiffness_compression
        )

        # 弹簧力（拉向驱动点）
        force_magnitude = -stiffness * displacement
        elastic_force = direction * force_magnitude[:, :, self.xp.newaxis]

        # 阻尼力
        relative_velocity = self.velocities  # 假设驱动点速度=0
        velocity_dot = self.xp.sum(relative_velocity * direction, axis=2)
        damping_force = -self.spring_damping_coeff * velocity_dot[:, :, self.xp.newaxis] * direction

        total_force = elastic_force + damping_force

        # 应用到非固定粒子
        mask_non_fixed = ~self.fixed_points
        self.forces[mask_non_fixed] += total_force[mask_non_fixed]

    def integrate(self, dt):
        self.reset_forces()
        self.compute_spring_forces_vectorized()
        self.compute_driven_point_forces()
        non_fixed_mask = ~self.fixed_points
        if self.xp.any(non_fixed_mask):
            acceleration = self.forces[non_fixed_mask] / self.mass_per_particle
            self.velocities[non_fixed_mask] += acceleration * dt
            self.velocities[non_fixed_mask] *= self.damping
            self.positions[non_fixed_mask] += self.velocities[non_fixed_mask] * dt
        self.time += dt

    def simulate(self, steps, dt):
        for _ in range(steps):
            self.integrate(dt * self.suofang_fact)

    def get_vertices(self):
        if self.use_gpu:
            return cp.asnumpy(self.positions).reshape(-1,3)
        else:
            return self.positions.reshape(-1,3)

    def get_faces(self):
        faces = []
        for i in range(self.nx-1):
            for j in range(self.ny-1):
                faces.append([i*self.ny+j,i*self.ny+(j+1),(i+1)*self.ny+j])
                faces.append([(i+1)*self.ny+j,i*self.ny+(j+1),(i+1)*self.ny+(j+1)])
        return np.array(faces)

# ---------------- 创建演示 ----------------
def create_cloth_demo():
    width, height = 0.8, 0.8
    n = 4.5   # 2 圈/ s
    r = 0.08 # 半径
    cloth = SoftClothSimulator(r, width, height, MESH_LENGTH,use_gpu=True)

    # 固定点
    # cloth.fix_point(5,5)

    # 设置驱动点随时间上下运动
    import math

    def driven_point_func(t):
        ddt = 0.02
        if t > ddt:
            a = np.array([width*0.5+r*math.cos(4*math.pi*n*(t-ddt)),
                        height*0.5+r*math.sin(4*math.pi*n*(t-ddt)),
                        0.0])
        else:
            a = np.array([width*0.5+r, height*0.5, 0.0])
        return a
    # def driven_point_func(t):
    #     a = np.array([0.5,0.5,0.0])
    #     print(a)
    #     return a
    cloth.driven_point_func = driven_point_func
    # cloth.fix_point(10,10)

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111,projection='3d')
    ax.set_title('Soft Cloth Simulation')

    vertices = cloth.get_vertices()
    faces = cloth.get_faces()
    surf = ax.plot_trisurf(vertices[:,0],vertices[:,1],vertices[:,2],
                           triangles=faces,color='lightblue',alpha=0.8,edgecolor='none')

    ax.set_xlim(-width,width)
    ax.set_ylim(-height,height)
    ax.set_zlim(-0.5,0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    def update(frame):
        cloth.simulate(SUB_STEP, DT)
        vertices = cloth.get_vertices()
        ax.clear()
        ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2],
                        triangles=faces, color='lightblue', alpha=0.8, edgecolor='none')
        
        # 将 CuPy 数组转换为 NumPy 数组再传递给 Matplotlib
        if cloth.use_gpu:
            driven_point_np = cp.asnumpy(cloth.driven_point_pos)
        else:
            driven_point_np = cloth.driven_point_pos
        
        ax.scatter(driven_point_np[0], driven_point_np[1], driven_point_np[2],
                color='red', s=80, label="Driven Point")
        ax.set_xlim(-width*0.5, width*1.5)
        ax.set_ylim(-height*0.5, height*1.5)
        ax.set_zlim(-0.5, 0.5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Soft Cloth Simulation \t Sim time: {:.3f} s'.format(cloth.time))
        return surf,

    ani = FuncAnimation(fig,update,frames=300,interval=5,blit=False,repeat=True)
    plt.tight_layout()
    plt.show()
    return cloth,ani
# 在文件末尾添加新的演示函数
def create_offline_cloth_demo():
    width, height = 0.4, 0.4
    n = 2.5   # 2.5 圈/ s
    r = 0.08  # 半径
    cloth = SoftClothSimulator(r, width, height, MESH_LENGTH, use_gpu=True)
    
    # 设置驱动点随时间上下运动
    import math
    
    def driven_point_func(t):
        ddt = 1
        if t > ddt:
            a = np.array([width*0.5+r*math.cos(2*math.pi*n*(t-ddt)),
                        height*0.5+r*math.sin(2*math.pi*n*(t-ddt)),
                        0.0])
        else:
            a = np.array([width*0.5+r, height*0.5, 0.0])
        return a
    
    cloth.driven_point_func = driven_point_func
    
    # 离线仿真参数
    simulation_time = 10.0  # 仿真10秒
    dt = DT
    steps = int(simulation_time / dt)
    
    # 存储关键帧数据
    keyframe_interval = max(1, steps // 300)  # 存储约300帧
    positions_history = []
    time_points = []
    
    print(f"开始离线仿真，总步数: {steps}")
    import time
    start_time = time.time()
    
    # 运行仿真并记录关键帧
    for step in range(steps):
        cloth.simulate(1, dt)  # 每步仿真
        
        # 记录关键帧
        if step % keyframe_interval == 0:
            # 只存储CPU上的副本，避免频繁数据传输
            if cloth.use_gpu:
                positions_copy = cp.asnumpy(cloth.positions)
            else:
                positions_copy = np.copy(cloth.positions)
            positions_history.append(positions_copy)
            time_points.append(cloth.time)
            
            # 显示进度
            if step % (steps // 20) == 0:
                progress = step / steps * 100
                print(f"仿真进度: {progress:.1f}%")
    
    sim_time = time.time() - start_time
    print(f"仿真完成，用时: {sim_time:.2f}秒")
    print(f"记录了 {len(positions_history)} 帧数据")
    
    # 绘制结果
    print("开始绘制结果...")
    start_time = time.time()
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Soft Cloth Simulation - Offline Rendering')
    
    faces = cloth.get_faces()
    
    def animate_frame(frame_idx):
        ax.clear()
        positions = positions_history[frame_idx]
        vertices = positions.reshape(-1, 3)
        
        ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2],
                       triangles=faces, color='lightblue', alpha=0.8, edgecolor='none')
        
        # 获取当前驱动点位置
        current_time = time_points[frame_idx]
        driven_pos = driven_point_func(current_time)
        ax.scatter(driven_pos[0], driven_pos[1], driven_pos[2],
                  color='red', s=80, label="Driven Point")
        
        ax.set_xlim(-width*0.5, width*1.5)
        ax.set_ylim(-height*0.5, height*1.5)
        ax.set_zlim(-0.5, 0.5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Soft Cloth Simulation (t = {time_points[frame_idx]:.3f} s)')
        
        return ax
    
    # 创建动画
    ani = FuncAnimation(fig, animate_frame, frames=len(positions_history),
                       interval=50, blit=False, repeat=True)
    
    render_time = time.time() - start_time
    print(f"绘制完成，用时: {render_time:.2f}秒")
    
    plt.tight_layout()
    plt.show()
    
    return cloth, ani, positions_history, time_points

def simulate_without_visualization():
    width, height = 0.4, 0.4
    r = 0.08
    cloth = SoftClothSimulator(r, width, height, MESH_LENGTH, use_gpu=True)
    
    # 设置驱动函数
    import math
    def driven_point_func(t):
        ddt = 1
        if t > ddt:
            a = np.array([width*0.5+r*math.cos(2*math.pi*2.5*(t-ddt)),
                        height*0.5+r*math.sin(2*math.pi*2.5*(t-ddt)),
                        0.0])
        else:
            a = np.array([width*0.5+r, height*0.5, 0.0])
        return a
    
    cloth.driven_point_func = driven_point_func
    
    # 运行完整仿真
    simulation_time = 10.0
    dt = DT
    steps = int(simulation_time / dt)
    
    import time
    start_time = time.time()
    
    # 批量仿真以减少GPU内核启动开销
    batch_size = 100  # 每批仿真步数
    
    for step in range(0, steps, batch_size):
        actual_batch_size = min(batch_size, steps - step)
        cloth.simulate(actual_batch_size, dt)
        
        # 显式清理GPU缓存（如果使用GPU）
        if cloth.use_gpu:
            cp.get_default_memory_pool().free_all_blocks()
        
        if step % (steps // 10) < batch_size:
            progress = min(step + actual_batch_size, steps) / steps * 100
            print(f"进度: {progress:.1f}%")
    
    sim_time = time.time() - start_time
    print(f"仿真完成，用时: {sim_time:.2f}秒")
    
    # 最后获取结果进行可视化
    final_vertices = cloth.get_vertices()
    faces = cloth.get_faces()
    
    # 绘制最终结果
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(final_vertices[:,0], final_vertices[:,1], final_vertices[:,2],
                   triangles=faces, color='lightblue', alpha=0.8)
    plt.title('Final Cloth State')
    plt.show()
    
    return cloth# ---------------- 运行演示 ----------------
if __name__=="__main__":

    print(f"Starting soft cloth simulation... {'(GPU Accelerated)' if GPU_AVAILABLE else '(CPU Only)'}")
    
    # 使用在线仿真模式
    cloth_sim, animation = create_cloth_demo()
    print("Simulation completed")

    # # 使用离线仿真模式
    # cloth_sim, animation, history, times = simulate_without_visualization()
    # print("离线仿真完成")