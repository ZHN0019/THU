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

# 导入vispy相关模块
from vispy import app, scene
from vispy.visuals import MeshVisual
from vispy.visuals.transforms import STTransform
from vispy.geometry import create_sphere
import vispy.visuals.transforms as vtf

# 如果可用可用GPU
try:
    import cupy as cp
    from cupy import cuda

    GPU_AVAILABLE = True
except ImportError:
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
DEFAULT_DAMPING = 0.99
DEFAULT_SPRING_DAMPING_COEFF = 0.3

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
        self.mass_per_particle = 0.01 * self.suofang_fact
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

        # 弹簧
        self._init_springs()
        self._preallocate_arrays()

        # 固定点
        self.fixed_points = self.xp.zeros((self.nx, self.ny), dtype=bool)

        # 驱动点弹簧
        self.initial_positions = self.positions.copy()  # 保存每个粒子的初始位置
        self.driven_point_stiffness_tension = 500.0 * self.suofang_fact
        self.driven_point_stiffness_compression = 50.0 * self.suofang_fact
        self.driven_point_func = None  # 可通过函数动态更新驱动点位置
        # 驱动点初始位置
        self.driven_point_pos = self.xp.array([width * 0.5 + Init_R, height * 0.5, 0])

        # 驱动点自然长度数组（仿真开始时计算）
        # 正确的实现 - 基于初始位置计算
        self.driven_rest_lengths = self.xp.linalg.norm(
            self.initial_positions - self.driven_point_pos,
            axis=2
        )

        # 时间计数
        self.time = 0.0

        # 编译CUDA kernel
        if self.use_gpu:
            self._compile_kernels()

    def fix_point(self, i, j):
        self.fixed_points[i, j] = True

    def _init_springs(self):
        self.springs = []
        # 结构弹簧
        for i in range(self.nx):
            for j in range(self.ny):
                if i < self.nx - 1:
                    self.springs.append({'type': 'stretch', 'p1': (i, j), 'p2': (i + 1, j), 'rest_length': float(
                        self.xp.linalg.norm(self.positions[i, j] - self.positions[i + 1, j]))})
                if j < self.ny - 1:
                    self.springs.append({'type': 'stretch', 'p1': (i, j), 'p2': (i, j + 1), 'rest_length': float(
                        self.xp.linalg.norm(self.positions[i, j] - self.positions[i, j + 1]))})
        # 剪切弹簧
        for i in range(self.nx - 1):
            for j in range(self.ny - 1):
                self.springs.append({'type': 'shear', 'p1': (i, j), 'p2': (i + 1, j + 1), 'rest_length': float(
                    self.xp.linalg.norm(self.positions[i, j] - self.positions[i + 1, j + 1]))})
                self.springs.append({'type': 'shear', 'p1': (i + 1, j), 'p2': (i, j + 1), 'rest_length': float(
                    self.xp.linalg.norm(self.positions[i + 1, j] - self.positions[i, j + 1]))})
        # 弯曲弹簧
        for i in range(self.nx):
            for j in range(self.ny):
                if i < self.nx - 2:
                    self.springs.append({'type': 'bend', 'p1': (i, j), 'p2': (i + 2, j), 'rest_length': float(
                        self.xp.linalg.norm(self.positions[i, j] - self.positions[i + 2, j]))})
                if j < self.ny - 2:
                    self.springs.append({'type': 'bend', 'p1': (i, j), 'p2': (i, j + 2), 'rest_length': float(
                        self.xp.linalg.norm(self.positions[i, j] - self.positions[i, j + 2]))})

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

    def _compile_kernels(self):
        # CUDA kernel for computing spring forces
        self.spring_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void compute_spring_forces(
            const double* positions,
            const double* velocities,
            const bool* fixed_points,
            double* forces,
            const int* p1_indices,
            const int* p2_indices,
            const double* rest_lengths,
            const int* spring_types,
            const int num_springs,
            const int nx,
            const int ny,
            const double stiffness_stretch_tension,
            const double stiffness_stretch_compression,
            const double stiffness_shear,
            const double stiffness_bend_tension,
            const double stiffness_bend_compression,
            const double spring_damping_coeff,
            const double mass_per_particle
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_springs) return;

            int p1_x = p1_indices[idx * 2];
            int p1_y = p1_indices[idx * 2 + 1];
            int p2_x = p2_indices[idx * 2];
            int p2_y = p2_indices[idx * 2 + 1];

            int p1_idx = (p1_x * ny + p1_y) * 3;
            int p2_idx = (p2_x * ny + p2_y) * 3;

            // 获取粒子位置
            double p1_pos[3] = {positions[p1_idx], positions[p1_idx+1], positions[p1_idx+2]};
            double p2_pos[3] = {positions[p2_idx], positions[p2_idx+1], positions[p2_idx+2]};

            // 计算向量和距离
            double delta[3] = {p2_pos[0] - p1_pos[0], p2_pos[1] - p1_pos[1], p2_pos[2] - p1_pos[2]};
            double dist = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
            if (dist < 1e-10) dist = 1e-10;

            double direction[3] = {delta[0]/dist, delta[1]/dist, delta[2]/dist};

            // 计算位移
            double displacement = dist - rest_lengths[idx];

            // 根据弹簧类型和位移确定刚度
            double stiffness;
            if (spring_types[idx] == 0) { // stretch
                stiffness = (displacement >= 0) ? stiffness_stretch_tension : stiffness_stretch_compression;
            } else if (spring_types[idx] == 1) { // shear
                stiffness = stiffness_shear;
            } else { // bend
                stiffness = (displacement >= 0) ? stiffness_bend_tension : stiffness_bend_compression;
            }

            // 计算弹性力
            double force_magnitude = stiffness * displacement;
            double elastic_force[3] = {direction[0] * force_magnitude, direction[1] * force_magnitude, direction[2] * force_magnitude};

            // 获取粒子速度
            double p1_vel[3] = {velocities[p1_idx], velocities[p1_idx+1], velocities[p1_idx+2]};
            double p2_vel[3] = {velocities[p2_idx], velocities[p2_idx+1], velocities[p2_idx+2]};

            // 计算相对速度和阻尼力
            double relative_velocity[3] = {p2_vel[0] - p1_vel[0], p2_vel[1] - p1_vel[1], p2_vel[2] - p1_vel[2]};
            double velocity_dot = relative_velocity[0]*direction[0] + relative_velocity[1]*direction[1] + relative_velocity[2]*direction[2];
            double damping_force[3] = {spring_damping_coeff * velocity_dot * direction[0],
                                       spring_damping_coeff * velocity_dot * direction[1],
                                       spring_damping_coeff * velocity_dot * direction[2]};

            // 总力
            double total_force[3] = {elastic_force[0] + damping_force[0],
                                     elastic_force[1] + damping_force[1],
                                     elastic_force[2] + damping_force[2]};

            // 应用力到粒子
            if (!fixed_points[p1_x * ny + p1_y]) {
                atomicAdd(&forces[p1_idx], total_force[0]);
                atomicAdd(&forces[p1_idx+1], total_force[1]);
                atomicAdd(&forces[p1_idx+2], total_force[2]);
            }
            if (!fixed_points[p2_x * ny + p2_y]) {
                atomicAdd(&forces[p2_idx], -total_force[0]);
                atomicAdd(&forces[p2_idx+1], -total_force[1]);
                atomicAdd(&forces[p2_idx+2], -total_force[2]);
            }
        }
        ''', 'compute_spring_forces')

        # CUDA kernel for integration
        self.integration_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void integrate(
            double* positions,
            double* velocities,
            double* forces,
            const bool* fixed_points,
            const double mass_per_particle,
            const double damping,
            const double dt,
            const int nx,
            const int ny,
            const double gravity_x,
            const double gravity_y,
            const double gravity_z
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_particles = nx * ny;
            if (idx >= total_particles) return;

            int i = idx / ny;
            int j = idx % ny;

            if (fixed_points[idx]) return;

            int pos_idx = idx * 3;

            // 添加重力
            forces[pos_idx] += gravity_x * mass_per_particle;
            forces[pos_idx+1] += gravity_y * mass_per_particle;
            forces[pos_idx+2] += gravity_z * mass_per_particle;

            // 计算加速度
            double acceleration[3] = {forces[pos_idx] / mass_per_particle,
                                      forces[pos_idx+1] / mass_per_particle,
                                      forces[pos_idx+2] / mass_per_particle};

            // 更新速度和位置
            velocities[pos_idx] = (velocities[pos_idx] + acceleration[0] * dt) * damping;
            velocities[pos_idx+1] = (velocities[pos_idx+1] + acceleration[1] * dt) * damping;
            velocities[pos_idx+2] = (velocities[pos_idx+2] + acceleration[2] * dt) * damping;

            positions[pos_idx] += velocities[pos_idx] * dt;
            positions[pos_idx+1] += velocities[pos_idx+1] * dt;
            positions[pos_idx+2] += velocities[pos_idx+2] * dt;

            // 重置力
            forces[pos_idx] = 0.0;
            forces[pos_idx+1] = 0.0;
            forces[pos_idx+2] = 0.0;
        }
        ''', 'integrate')

        # CUDA kernel for driven point forces
        self.driven_point_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void compute_driven_point_forces(
            const double* positions,
            const double* velocities,
            const bool* fixed_points,
            double* forces,
            const double* initial_positions,
            const double driven_point_x,
            const double driven_point_y,
            const double driven_point_z,
            const double* driven_rest_lengths,
            const double driven_point_stiffness_tension,
            const double driven_point_stiffness_compression,
            const double spring_damping_coeff,
            const int nx,
            const int ny
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_particles = nx * ny;
            if (idx >= total_particles) return;

            int i = idx / ny;
            int j = idx % ny;

            if (fixed_points[idx]) return;

            int pos_idx = idx * 3;

            // 当前粒子到驱动点的向量
            double delta[3] = {positions[pos_idx] - driven_point_x,
                               positions[pos_idx+1] - driven_point_y,
                               positions[pos_idx+2] - driven_point_z};

            // 当前长度
            double current_length = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
            if (current_length < 1e-10) current_length = 1e-10;

            // 单位方向
            double direction[3] = {delta[0]/current_length, delta[1]/current_length, delta[2]/current_length};

            // 位移 = 当前长度 - 自然长度
            double displacement = current_length - driven_rest_lengths[idx];

            // 各向异性刚度
            double stiffness = (displacement >= 0) ? driven_point_stiffness_tension : driven_point_stiffness_compression;

            // 弹簧力（拉向驱动点）
            double force_magnitude = -stiffness * displacement;
            double elastic_force[3] = {direction[0] * force_magnitude,
                                       direction[1] * force_magnitude,
                                       direction[2] * force_magnitude};

            // 阻尼力
            double velocity_dot = velocities[pos_idx]*direction[0] +
                                  velocities[pos_idx+1]*direction[1] +
                                  velocities[pos_idx+2]*direction[2];
            double damping_force[3] = {-spring_damping_coeff * velocity_dot * direction[0],
                                       -spring_damping_coeff * velocity_dot * direction[1],
                                       -spring_damping_coeff * velocity_dot * direction[2]};

            // 应用到非固定粒子
            forces[pos_idx] += elastic_force[0] + damping_force[0];
            forces[pos_idx+1] += elastic_force[1] + damping_force[1];
            forces[pos_idx+2] += elastic_force[2] + damping_force[2];
        }
        ''', 'compute_driven_point_forces')

    def compute_spring_forces_vectorized(self):
        if not self.use_gpu:
            # CPU版本保持不变
            p1_pos = self.positions[self.p1_indices[:, 0], self.p1_indices[:, 1]]
            p2_pos = self.positions[self.p2_indices[:, 0], self.p2_indices[:, 1]]
            p1_vel = self.velocities[self.p1_indices[:, 0], self.p1_indices[:, 1]]
            p2_vel = self.velocities[self.p2_indices[:, 0], self.p2_indices[:, 1]]
            delta = p2_pos - p1_pos
            dist = self.xp.linalg.norm(delta, axis=1)
            dist = self.xp.where(dist == 0, 1e-10, dist)
            direction = delta / dist[:, self.xp.newaxis]
            displacement = dist - self.rest_lengths
            stiffness = self.xp.zeros(len(self.springs))
            stretch_mask = self.xp.array([t == 'stretch' for t in self.spring_types])
            tension_mask = displacement >= 0
            compression_mask = displacement < 0
            stiffness = self.xp.where(stretch_mask & tension_mask, self.stiffness_stretch_tension, stiffness)
            stiffness = self.xp.where(stretch_mask & compression_mask, self.stiffness_stretch_compression, stiffness)
            shear_mask = self.xp.array([t == 'shear' for t in self.spring_types])
            stiffness = self.xp.where(shear_mask, self.stiffness_shear, stiffness)
            bend_mask = self.xp.array([t == 'bend' for t in self.spring_types])
            stiffness = self.xp.where(bend_mask & tension_mask, self.stiffness_bend_tension, stiffness)
            stiffness = self.xp.where(bend_mask & compression_mask, self.stiffness_bend_compression, stiffness)
            force_magnitude = stiffness * displacement
            elastic_force = direction * force_magnitude[:, self.xp.newaxis]
            relative_velocity = p2_vel - p1_vel
            velocity_dot = self.xp.sum(relative_velocity * direction, axis=1)
            damping_force = (self.spring_damping_coeff * velocity_dot[:, self.xp.newaxis] * direction)
            total_force = elastic_force + damping_force
            for i, (p1_idx, p2_idx) in enumerate(zip(self.p1_indices, self.p2_indices)):
                if not self.fixed_points[p1_idx[0], p1_idx[1]]:
                    self.forces[p1_idx[0], p1_idx[1]] += total_force[i]
                if not self.fixed_points[p2_idx[0], p2_idx[1]]:
                    self.forces[p2_idx[0], p2_idx[1]] -= total_force[i]
        else:
            # GPU版本使用CUDA kernel
            # 准备数据
            num_springs = len(self.springs)
            spring_types_map = {'stretch': 0, 'shear': 1, 'bend': 2}
            spring_types_array = self.xp.array([spring_types_map[t] for t in self.spring_types], dtype=self.xp.int32)

            p1_indices_flat = self.xp.array([[idx[0], idx[1]] for idx in self.p1_indices],
                                            dtype=self.xp.int32).flatten()
            p2_indices_flat = self.xp.array([[idx[0], idx[1]] for idx in self.p2_indices],
                                            dtype=self.xp.int32).flatten()

            # 调用CUDA kernel
            block_size = 256
            grid_size = (num_springs + block_size - 1) // block_size

            self.spring_kernel(
                (grid_size,), (block_size,),
                (
                    self.positions,
                    self.velocities,
                    self.fixed_points,
                    self.forces,
                    p1_indices_flat,
                    p2_indices_flat,
                    self.rest_lengths,
                    spring_types_array,
                    num_springs,
                    self.nx,
                    self.ny,
                    self.stiffness_stretch_tension,
                    self.stiffness_stretch_compression,
                    self.stiffness_shear,
                    self.stiffness_bend_tension,
                    self.stiffness_bend_compression,
                    self.spring_damping_coeff,
                    self.mass_per_particle
                )
            )

    def compute_driven_point_forces(self):
        # 更新驱动点位置
        if self.driven_point_func is not None:
            # 确保驱动点位置与当前使用的计算库一致 (CPU/GPU)
            driven_point_np = self.driven_point_func(self.time)
            self.driven_point_pos = self.xp.array(driven_point_np)  # 转换为正确的类型

        if not self.use_gpu:
            # CPU版本保持不变
            # 当前粒子到驱动点的向量
            delta = self.positions - self.driven_point_pos

            # 当前长度
            current_length = self.xp.linalg.norm(delta, axis=2)
            current_length = self.xp.where(current_length == 0, 1e-10, current_length)

            # 单位方向
            direction = delta / current_length[:, :, self.xp.newaxis]

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
        else:
            # GPU版本使用CUDA kernel
            # 在GPU版本中，我们也需要重新计算自然长度以匹配CPU版本的行为
            self.driven_rest_lengths = self.xp.linalg.norm(
                self.initial_positions - self.driven_point_pos,
                axis=2
            )

            block_size = 256
            num_particles = self.nx * self.ny
            grid_size = (num_particles + block_size - 1) // block_size

            self.driven_point_kernel(
                (grid_size,), (block_size,),
                (
                    self.positions,
                    self.velocities,
                    self.fixed_points,
                    self.forces,
                    self.initial_positions,
                    float(self.driven_point_pos[0]),
                    float(self.driven_point_pos[1]),
                    float(self.driven_point_pos[2]),
                    self.driven_rest_lengths,
                    self.driven_point_stiffness_tension,
                    self.driven_point_stiffness_compression,
                    self.spring_damping_coeff,
                    self.nx,
                    self.ny
                )
            )

    def integrate(self, dt):
        if not self.use_gpu:
            # CPU版本保持不变
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
        else:
            # GPU版本使用CUDA kernel
            # 首先重置力并添加重力
            self.forces.fill(0.0)
            block_size = 256
            num_particles = self.nx * self.ny
            grid_size = (num_particles + block_size - 1) // block_size

            self.integration_kernel(
                (grid_size,), (block_size,),
                (
                    self.positions,
                    self.velocities,
                    self.forces,
                    self.fixed_points,
                    self.mass_per_particle,
                    self.damping,
                    dt,
                    self.nx,
                    self.ny,
                    float(self.gravity[0]),
                    float(self.gravity[1]),
                    float(self.gravity[2])
                )
            )

            # 计算弹簧力
            self.compute_spring_forces_vectorized()

            # 计算驱动点力
            self.compute_driven_point_forces()

            self.time += dt

    def reset_forces(self):
        self.forces.fill(0.0)
        gravity_force = self.gravity * self.mass_per_particle
        self.forces += gravity_force

    def simulate(self, steps, dt):
        for _ in range(steps):
            self.integrate(dt * self.suofang_fact)

    def get_vertices(self):
        if self.use_gpu:
            return cp.asnumpy(self.positions).reshape(-1, 3)
        else:
            return self.positions.reshape(-1, 3)

    def get_faces(self):
        faces = []
        for i in range(self.nx - 1):
            for j in range(self.ny - 1):
                faces.append([i * self.ny + j, i * self.ny + (j + 1), (i + 1) * self.ny + j])
                faces.append([(i + 1) * self.ny + j, i * self.ny + (j + 1), (i + 1) * self.ny + (j + 1)])
        return np.array(faces)


# ---------------- 创建演示 ----------------
def create_gpu_cloth_demo():
    width, height = 0.4, 0.4
    n = 2.5  # 2 圈/ s
    r = 0.08  # 半径
    cloth = SoftClothSimulator(r, width, height, MESH_LENGTH, use_gpu=True)

    # 设置驱动点随时间上下运动
    import math

    def driven_point_func(t):
        ddt = 1
        if t > ddt:
            a = np.array([width * 0.5 + r * math.cos(2 * math.pi * n * (t - ddt)),
                          height * 0.5 + r * math.sin(2 * math.pi * n * (t - ddt)),
                          0.0])
        else:
            a = np.array([width * 0.5 + r, height * 0.5, 0.0])
        return a

    cloth.driven_point_func = driven_point_func

    # 创建vispy canvas
    canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 60
    view.camera.distance = 2

    # 获取初始顶点和面
    vertices = cloth.get_vertices()
    faces = cloth.get_faces()

    # 创建网格对象
    mesh = scene.visuals.Mesh(vertices=vertices, faces=faces, color=(0.5, 0.7, 1.0, 0.8))
    view.add(mesh)

    # 添加驱动点
    driven_point_marker = scene.visuals.Markers()
    driven_point_marker.set_data(np.array([[width * 0.5 + r, height * 0.5, 0.0]]),
                                 face_color='red', size=10)
    view.add(driven_point_marker)

    # 添加坐标轴
    axis = scene.visuals.XYZAxis(parent=view.scene)

    def update(ev):
        # 仿真几步
        cloth.simulate(SUB_STEP, DT)

        # 更新网格顶点
        vertices = cloth.get_vertices()
        mesh.set_data(vertices=vertices, faces=faces)

        # 更新驱动点位置
        if cloth.driven_point_func is not None:
            driven_pos = cloth.driven_point_func(cloth.time)
            driven_point_marker.set_data(np.array([driven_pos]), face_color='red', size=10)

        canvas.title = f'Soft Cloth Simulation (GPU) - Time: {cloth.time:.2f}s'

    # 设置定时器更新
    timer = app.Timer(interval=50, connect=update, start=True)  # 增加间隔以提高稳定性

    # 设置视图范围
    view.camera.set_range(x=[-0.2, 0.6], y=[-0.2, 0.6], z=[-0.5, 0.5])

    return cloth, canvas, timer


# 主程序
if __name__ == "__main__":


    print(
        f"Starting GPU-accelerated soft cloth simulation... {'(GPU Available)' if GPU_AVAILABLE else '(Falling back to CPU)'}")

    # 使用GPU加速的vispy渲染模式
    cloth_sim, canvas, timer = create_gpu_cloth_demo()

    # 启动vispy事件循环
    app.run()