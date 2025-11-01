import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import math
import warnings

# ---------------------------
# 辅助函数：三维重心坐标
def barycentric_coords_3d(A, P1, P2, P3):
    """
    输入：
        A : 内部点 np.array([x,y,z])
        P1,P2,P3 : 三角形顶点 np.array([x,y,z])
    返回：
        l : np.array([l1,l2,l3]) 重心坐标
    """
    # 将三角形投影到任意平面计算重心坐标，常用 x-y 平面
    # 更稳健的方法是用向量代数
    v0 = P2 - P1
    v1 = P3 - P1
    v2 = A - P1

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    l2 = (d11 * d20 - d01 * d21) / denom
    l3 = (d00 * d21 - d01 * d20) / denom
    l1 = 1 - l2 - l3
    return np.array([l1, l2, l3])

# ---------------------------
# 1) 允许形变、最小范数 / 最小能量解
def vertex_velocities_min_norm_3d(P1, P2, P3, A, vA):
    """
    输入：
        P1,P2,P3 : 三角形顶点 np.array([x,y,z])
        A : 内部点 np.array([x,y,z])
        vA : 内部点速度 np.array([vx,vy,vz])
    返回：
        v1,v2,v3 : 三顶点速度 np.array([vx,vy,vz])
    """
    lambdas = barycentric_coords_3d(A, P1, P2, P3)
    D = np.sum(lambdas**2)
    v1 = (lambdas[0] / D) * vA
    v2 = (lambdas[1] / D) * vA
    v3 = (lambdas[2] / D) * vA
    return v1, v2, v3

# ---------------------------
# 2) 刚体运动（保持三角形刚性），最小能量解 = 平移
def vertex_velocities_rigid_3d(P1, P2, P3, A, vA):
    """
    刚体最小能量解：纯平移，所有顶点速度等于 vA
    """
    v1 = vA.copy()
    v2 = vA.copy()
    v3 = vA.copy()
    return v1, v2, v3

# ---------------------------
# Example usage
# if __name__ == "__main__":
#     # P1 = np.array([0.0, 0.0])
#     # P2 = np.array([1.0, 0.0])
#     # P3 = np.array([0.0, 1.0])
#     # A  = np.array([0.3, 0.2])
#     # vA = np.array([1.0, 0.5])
#     R = 3
#     P1 = np.array([0.0, R,0])
#     P2 = np.array([R*np.sin(np.deg2rad(60)), -R*np.cos(np.deg2rad(60)),0])
#     P3 = np.array([-R*np.sin(np.deg2rad(60)), -R*np.cos(np.deg2rad(60)),0])
#     A  = np.array([2.0, 0.0,0])
#     vA = np.array([1.0, 0.5,5])
#
#     v1_min, v2_min, v3_min = vertex_velocities_min_norm_3d(P1, P2, P3, A, vA)
#     v1_rig, v2_rig, v3_rig = vertex_velocities_rigid_3d(P1, P2, P3, A, vA)
#
#     print("Min-norm (allow deformation):")
#     print(v1_min, v2_min, v3_min,np.linalg.norm(v1_min)+np.linalg.norm(v2_min)+np.linalg.norm(v3_min))
#     print("Rigid motion:")
#     print(v1_rig, v2_rig, v3_rig,np.linalg.norm(v1_rig)+np.linalg.norm(v2_rig)+np.linalg.norm(v3_rig))


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import math
import warnings

# ---------- 辅助：点到三角形最近点 ----------
def point_triangle_distance(p, a, b, c):
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    d00 = np.dot(ab, ab)
    d01 = np.dot(ab, ac)
    d11 = np.dot(ac, ac)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return a.copy(), np.linalg.norm(p - a), (1.0, 0.0, 0.0)
    v = (d11 * d1 - d01 * d2) / denom
    w = (d00 * d2 - d01 * d1) / denom
    u = 1.0 - v - w
    if u >= 0 and v >= 0 and w >= 0:
        q = u * a + v * b + w * c
        return q, np.linalg.norm(p - q), (u, v, w)
    def seg_closest(pa, pb):
        r = pb - pa
        rr = np.dot(r, r)
        if rr < 1e-12:
            return pa.copy(), np.linalg.norm(p - pa)
        t = np.dot(p - pa, r) / rr
        t = max(0.0, min(1.0, t))
        q = pa + t * r
        return q, np.linalg.norm(p - q)
    cand = [seg_closest(a, b), seg_closest(b, c), seg_closest(c, a)]
    q, dist = min(cand, key=lambda x: x[1])
    # 最小二乘估算重心坐标
    v0 = b - a
    v1 = c - a
    v2 = q - a
    d00 = np.dot(v0, v0); d01 = np.dot(v0, v1); d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0); d21 = np.dot(v2, v1)
    denom2 = d00 * d11 - d01 * d01
    if abs(denom2) < 1e-12:
        return q, dist, (1.0, 0.0, 0.0)
    v = (d11 * d20 - d01 * d21) / denom2
    w = (d00 * d21 - d01 * d20) / denom2
    u = 1.0 - v - w
    return q, dist, (u, v, w)

# ====================== SoftClothSimulator（按物理量纲缩放） ======================
class SoftClothSimulator:
    def __init__(self, nodes, springs, faces=None,
                 # 下面的刚度/阻尼/质量参数都被解释为“物理量级”参数（不随 mesh 盲目线性缩放）
                 stiffness_stretch_tension=1000.0,  # per-spring axial stiffness (N/m equivalent in discrete model)
                 stiffness_bend_tension=50.0,
                 stiffness_shear=800.0,
                 compression_to_tension_ratio=0.1,
                 damping=0.995,  # velocity global damping factor (dimensionless)
                 spring_damping_coeff=0.5,  # damping coefficient (in same units as spring forces per velocity)
                 meshlength=0.1,
                 surface_density=1.0,  # mass per unit area (kg / m^2) -- choose to match default mass
                 thickness=0.001,  # cloth thickness (m) used if needed (not strictly necessary)
                 momentum_loss=0.4,
                 collision_restitution=0.3,
                 drive_influence_ratio=1,
                 # 新增参数
                 drive_velocity_transfer_ratio=0.05,  # 降低速度传递比例
                 vertex_velocity_method='min_norm'):
        """
        关键缩放思路：
          - mass_per_particle ~ surface_density * meshlength^2
          - treat the provided stiffness_* as per-spring stiffness (do NOT multiply again by mesh scale)
          - critical dt ~ C * sqrt(mass_per_particle / k_max) -> 若用户 dt 太大，则自动进行子步细分
        """
        # 设置默认的驱动点投影方法
        self._compute_initial_projection = self._compute_initial_projection_barycentric
        self.meshlength = float(meshlength)
        self.positions = np.array(nodes, dtype=float)
        self.velocities = np.zeros_like(self.positions)
        self.forces = np.zeros_like(self.positions)
        self.springs = springs
        self.faces = np.array(faces) if faces is not None else None
        self.num_particles = len(self.positions)
        self.fixed_points = np.zeros(self.num_particles, dtype=bool)

        # 初始网格（用于计算驱动点初始投影位置）
        self.initial_positions = self.positions.copy()

        # 驱动点
        self.DrivePoints = []  # 每个元素为 dict {'func', 'pos', 'prev_pos', 'collision_info', 'initial_proj'}
        self.drive_influence_ratio = drive_influence_ratio

        # 物理量（按上面说明）
        self.stiffness_stretch_tension = float(stiffness_stretch_tension)
        self.stiffness_stretch_compression = float(stiffness_stretch_tension * compression_to_tension_ratio)
        self.stiffness_bend_tension = float(stiffness_bend_tension)
        self.stiffness_bend_compression = float(stiffness_bend_tension * compression_to_tension_ratio)
        self.stiffness_shear_tension = float(stiffness_shear)
        self.stiffness_shear_compression = float(stiffness_shear * compression_to_tension_ratio)
        self.damping = float(damping)
        # spring_damping_coeff 在质点质量、时间尺度不同的时候也应相应调整；将其解释为“每米长度下的阻尼系数”，
        # 在离散化后使用时乘以 meshlength（使其与质量/速度量纲一致的经验式处理）
        self.spring_damping_coeff_base = float(spring_damping_coeff)
        self.spring_damping_coeff = self.spring_damping_coeff_base * self.meshlength

        # 质量按面积缩放（2D cloth）
        # 让默认 meshlength = 0.1 时，mass_per_particle 与原先代码接近：原来为 0.01
        # 设定 surface_density（kg/m^2）为外部参数，默认 surface_density = 1.0，
        # 因此 mass_per_particle = surface_density * meshlength^2.
        self.surface_density = float(surface_density)
        self.mass_per_particle = self.surface_density * (self.meshlength ** 2)

        self.gravity = np.array([0, 0, -9.81])
        self.min_distance = self.meshlength  # 碰撞判阈按网格长度
        self.momentum_loss = momentum_loss
        self.collision_restitution = collision_restitution

        # 防穿透
        self.contact_guard_time = meshlength
        self.contact_frames_remaining = np.zeros(self.num_particles, dtype=int)

        # 新增参数
        self.drive_velocity_transfer_ratio = drive_velocity_transfer_ratio
        self.vertex_velocity_method = vertex_velocity_method

    # ---------- 固定点 ----------
    def fix_point(self, idx):
        self.fixed_points[idx] = True

    # ---------- 驱动点 ----------
    def add_drive_point(self, func):
        pos = np.array(func(0.0))
        self.DrivePoints.append({'func': func, 'pos': pos.copy(), 'prev_pos': pos.copy(),
                                 'collision_info': None, 'initial_proj': None})

    def update_drive_points(self, t):
        for dp in self.DrivePoints:
            dp['prev_pos'] = dp['pos'].copy()
            dp['pos'] = np.array(dp['func'](t))

    # 分配驱动点与网格点的连接
    # 碰撞点
    def _compute_initial_projection_barycentric(self, dp):
        """
        原始方法：通过重心坐标计算驱动点在初始网格中的对应位置
        """
        f_idx, bary = dp['collision_info']
        if f_idx is None or bary is None:
            return None

        # 碰撞点在初始网格的位置（用 initial_positions）
        q0 = (bary[0] * self.initial_positions[f_idx[0]] +
              bary[1] * self.initial_positions[f_idx[1]] +
              bary[2] * self.initial_positions[f_idx[2]])
        return q0
    # 直接映射
    def _compute_initial_projection_xy_projection(self, dp):
        """
        新方法：通过XY平面投影计算驱动点在初始网格中的对应位置
        """
        f_idx, bary = dp['collision_info']
        if f_idx is None or bary is None:
            return None

        return np.array([dp['pos'][0], dp['pos'][1], 0])

    def set_drive_projection_method(self, method='barycentric'):
        """
        设置驱动点初始位置计算方法

        Parameters:
        method: str, 'barycentric' 或 'xy_projection'
        """
        if method == 'barycentric':
            self._compute_initial_projection = self._compute_initial_projection_barycentric
        elif method == 'xy_projection':
            self._compute_initial_projection = self._compute_initial_projection_xy_projection
        else:
            raise ValueError("Method must be 'barycentric' or 'xy_projection'")

    # ---------- 力计算 ----------
    def reset_forces(self):
        self.forces.fill(0.0)
        for i in range(self.num_particles):
            self.forces[i] += self.gravity * self.mass_per_particle

    def compute_spring_forces(self):
        # 使用类内全局刚度值，根据 spring['type'] 选择
        for spring in self.springs:
            i, j = spring['p1'], spring['p2']
            p1, p2 = self.positions[i], self.positions[j]
            v1, v2 = self.velocities[i], self.velocities[j]
            delta = p2 - p1
            dist = np.linalg.norm(delta)
            dist = np.where(dist == 0, 1e-10, dist)
            direction = delta / dist
            displacement = dist - spring['rest_length']
            # print(displacement)
            if spring['type'] == 'stretch':
                stiffness = self.stiffness_stretch_tension if displacement >= 0 else self.stiffness_stretch_compression
            elif spring['type'] == 'shear':
                stiffness = self.stiffness_shear_tension if displacement >= 0 else self.stiffness_shear_compression
            else:
                stiffness = self.stiffness_bend_tension if displacement >= 0 else self.stiffness_bend_compression
            force = stiffness * displacement * direction
            damping_force = self.spring_damping_coeff * np.dot(v2 - v1, direction) * direction
            total_force = force + damping_force
            self.forces[i] += total_force
            self.forces[j] -= total_force

    # 添加一个新的辅助函数用于线段与球体的相交检测
    def ray_sphere_intersect(self, origin, direction, sphere_center, sphere_radius):
        """
        检测从origin出发，沿direction方向的射线是否与球体相交
        返回最近的交点参数t（如果有的话）
        """
        oc = origin - sphere_center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - sphere_radius * sphere_radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None  # 没有交点

        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2.0 * a)
        t2 = (-b + sqrt_discriminant) / (2.0 * a)

        # 返回最近的正交点
        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            return None  # 交点在射线反方向

    # ---------- 碰撞 ----------
    def resolve_contacts(self, dt_sub):
        """
        完全非弹性碰撞处理（改进版）。
        优化速度传递机制，防止布料缩成一团
        """
        for dp in self.DrivePoints:
            dp_pos = dp['pos']
            dp_prev = dp['prev_pos']
            dp_vel = (dp_pos - dp_prev) / max(dt_sub, 1e-12)

            # ---------------- 寻找距离最近的碰撞点 ----------------
            closest_dist = 1e12
            closest_idx = None
            collision_time = 1.0  # 碰撞发生的时间比例（0-1之间）

            for i in range(self.num_particles):
                if self.fixed_points[i]:
                    continue

                particle_pos = self.positions[i]
                particle_vel = self.velocities[i]

                # 计算相对位置和速度
                rel_pos = particle_pos - dp_prev
                rel_vel = particle_vel - dp_vel

                # 使用连续碰撞检测
                if np.linalg.norm(rel_vel) > 1e-12:
                    direction = rel_vel * dt_sub
                    intersection_t = self.ray_sphere_intersect(rel_pos, direction, np.array([0, 0, 0]),
                                                               self.min_distance)

                    if intersection_t is not None and 0 <= intersection_t <= 1:
                        # 发生碰撞，检查是否是最近的碰撞
                        if intersection_t < closest_dist:
                            closest_dist = intersection_t
                            closest_idx = i
                            collision_time = intersection_t
                else:
                    # 使用静态检测作为备选
                    delta = particle_pos - dp_pos
                    dist = np.linalg.norm(delta)
                    if dist < self.min_distance and dist > 1e-8 and dist < closest_dist:
                        closest_dist = dist
                        closest_idx = i

            # ---------------- 处理最近碰撞点 ----------------
            if closest_idx is not None:
                # 计算碰撞点位置
                collision_dp_pos = dp_prev + collision_time * (dp_pos - dp_prev)
                collision_particle_pos = self.positions[closest_idx] + collision_time * dt_sub * self.velocities[
                    closest_idx]

                n = (collision_particle_pos - collision_dp_pos)
                dist = np.linalg.norm(n)
                if dist > 1e-12:
                    n = n / dist
                    v_rel = np.dot(self.velocities[closest_idx] - dp_vel, n)

                    # --- 速度修正（完全非弹性） ---
                    if v_rel < 0:
                        # 正确做法：将粒子速度的法向分量替换为驱动点的法向速度（一次操作）
                        vel_n = np.dot(self.velocities[closest_idx], n)
                        self.velocities[closest_idx] += (dp_vel - vel_n) * n

                    # --- 位置修正 ---
                    # 将粒子推出到最小距离位置
                    if dist < self.min_distance:
                        penetration = self.min_distance - dist
                        self.positions[closest_idx] += n * penetration * 1.01

                    # 更新驱动弹簧碰撞信息（基于当前三角形）
                    min_tri_dist = 1e12
                    best_tri = None
                    best_bary = None
                    if self.faces is not None:
                        for f_idx in self.faces:
                            if closest_idx not in f_idx:
                                continue
                            a, b, c = self.positions[f_idx[0]], self.positions[f_idx[1]], self.positions[f_idx[2]]
                            q, d, bary = point_triangle_distance(collision_dp_pos, a, b, c)
                            if d < min_tri_dist:
                                min_tri_dist = d
                                best_tri = f_idx
                                best_bary = bary
                    dp['collision_info'] = (best_tri, best_bary)

                    # 根据碰撞信息设置三角形顶点速度（局部化处理）
                    if best_tri is not None:
                        # 获取三角形三个顶点的索引
                        idx1, idx2, idx3 = best_tri

                        # 计算顶点速度
                        P1, P2, P3 = self.positions[idx1], self.positions[idx2], self.positions[idx3]

                        # 根据选择的方法计算顶点速度
                        if self.vertex_velocity_method == 'rigid':
                            v1, v2, v3 = vertex_velocities_rigid_3d(P1, P2, P3, collision_dp_pos,
                                                                    dp_vel * self.drive_velocity_transfer_ratio)
                        else:  # 默认使用最小范数方法
                            v1, v2, v3 = vertex_velocities_min_norm_3d(P1, P2, P3, collision_dp_pos,
                                                                       dp_vel * self.drive_velocity_transfer_ratio)

                        # 应用速度到顶点（仅当顶点不是固定点时）
                        # 添加衰减因子，使速度传递更加局部化
                        decay_factor = 0.3  # 衰减因子，防止速度扩散

                        if not self.fixed_points[idx1]:
                            self.velocities[idx1] += v1 * decay_factor
                        if not self.fixed_points[idx2]:
                            self.velocities[idx2] += v2 * decay_factor
                        if not self.fixed_points[idx3]:
                            self.velocities[idx3] += v3 * decay_factor

                        # 添加额外的阻尼，防止过度运动
                        self.velocities[idx1] *= 0.9
                        self.velocities[idx2] *= 0.9
                        self.velocities[idx3] *= 0.9
                else:
                    dp['collision_info'] = None
            else:
                dp['collision_info'] = None

    # ---------- 辅助：计算临界时间步 ----------
    def critical_timestep(self):
        # 经验系数 C，保守一点取 0.2
        C = 0.2
        # 找到当前模型中最大的弹簧刚度（最危险的）
        k_candidates = [self.stiffness_stretch_tension, self.stiffness_shear_tension, self.stiffness_bend_tension]
        k_max = max(k_candidates)
        if k_max <= 0:
            return 1e-6
        dt_crit = C * math.sqrt(self.mass_per_particle / k_max)
        return dt_crit

    # 处理穿透问题
    def prevent_penetration(self, dt_sub):
        """
        防止布料穿透驱动点（改进版）
        """
        for dp in self.DrivePoints:
            dp_pos = dp['pos']
            dp_prev = dp['prev_pos']
            dp_vel = (dp_pos - dp_prev) / max(dt_sub, 1e-12)

            # 检查是否有粒子穿透驱动点
            for i in range(self.num_particles):
                if self.fixed_points[i]:
                    continue

                particle_pos = self.positions[i]
                particle_vel = self.velocities[i]

                # 计算相对位置和速度
                rel_pos = particle_pos - dp_prev  # 粒子相对于驱动点初始位置
                rel_vel = particle_vel - dp_vel  # 相对速度

                # 使用连续碰撞检测检查在dt_sub时间内是否发生碰撞
                if np.linalg.norm(rel_vel) > 1e-12:  # 如果有相对运动
                    # 检查粒子路径是否与驱动点球体相交
                    direction = rel_vel * dt_sub
                    intersection_t = self.ray_sphere_intersect(rel_pos, direction, np.array([0, 0, 0]),
                                                               self.min_distance)

                    # 如果在[0,1]范围内有交点，则发生碰撞
                    if intersection_t is not None and 0 <= intersection_t <= 1:
                        # 发生碰撞，将粒子定位到碰撞点
                        collision_point = dp_prev + intersection_t * dt_sub * dp_vel + \
                                          rel_pos + intersection_t * direction
                        # 计算法线方向
                        n = (particle_pos - collision_point)
                        dist = np.linalg.norm(n)
                        if dist > 1e-12:
                            n = n / dist
                            # 将粒子推出到最小距离位置
                            self.positions[i] = collision_point + n * self.min_distance * 1.01
                        continue  # 处理下一个粒子

                # 如果没有检测到连续碰撞，使用原始的静态检测方法
                delta = particle_pos - dp_pos
                dist = np.linalg.norm(delta)

                # 如果粒子距离驱动点太近，将其推出
                if dist < self.min_distance and dist > 1e-12:
                    n = delta / dist  # 从驱动点指向粒子的单位向量
                    # 将粒子移动到最小距离位置
                    self.positions[i] = dp_pos + n * self.min_distance * 1.01

    # 在 integrate 方法中调用防穿透方法
    # ---------- 积分（显式，带自适应子步） ----------
    def integrate(self, dt, t_global=0.0):
        """
        dt: 用户传入的大步；内部根据临界 dt 划分若干子步执行显式积分。
        """
        # 更新驱动点（基于全局时间）
        self.update_drive_points(t_global)

        # 计算临界子步
        dt_crit = self.critical_timestep()
        if dt_crit <= 0:
            dt_crit = 1e-6
        # 允许的最大子步：如果用户 dt 比临界值大，就细分
        n_sub = max(1, int(math.ceil(dt / dt_crit)))
        dt_sub = dt / n_sub

        # 如果分子步太多，警告但还是执行
        if n_sub > 200:
            warnings.warn(f"Very many substeps ({n_sub}) for stability. dt={dt}, dt_crit={dt_crit:.3e}")

        for _ in range(n_sub):
            # 在每个子步内，按显式欧拉做力计算 -> v, pos 更新，并做碰撞修正
            self.reset_forces()
            self.compute_spring_forces()

            # 碰撞处理（会修改 velocities 并更新驱动碰撞信息）
            self.resolve_contacts(dt_sub)

            acceleration = self.forces / max(self.mass_per_particle, 1e-12)

            # 更新速度（排除固定点）
            movable = ~self.fixed_points
            self.velocities[movable] += acceleration[movable] * dt_sub
            # 全局速度阻尼（无量纲）
            self.velocities[movable] *= self.damping * 0.99  # 增加一些额外阻尼

            # 更新位置
            self.positions[movable] += self.velocities[movable] * dt_sub

            # 防止穿透
            self.prevent_penetration(dt_sub)

    def simulate(self, steps, dt, t0=0.0):
        t = t0
        for _ in range(steps):
            # integrate 接受 dt 作为一个“宏步”，内部会自动细分
            self.integrate(dt, t)
            t += dt

    def get_vertices(self):
        return self.positions

# ====================== 构建 10x10 网格（同原版） ======================
def build_10x10_cloth(mesh_length=0.1):
    nx = ny = 10
    nodes = []
    for i in range(nx):
        for j in range(ny):
            nodes.append([i*mesh_length, j*mesh_length, 0.0])
    nodes = np.array(nodes)
    springs = []
    for i in range(nx):
        for j in range(ny):
            idx = i*ny+j
            if i < nx-1:
                springs.append({'type':'stretch','p1':idx,'p2':(i+1)*ny+j,'rest_length':mesh_length})
            if j < ny-1:
                springs.append({'type':'stretch','p1':idx,'p2':i*ny+(j+1),'rest_length':mesh_length})
            if i < nx-1 and j < ny-1:
                springs.append({'type':'shear','p1':idx,'p2':(i+1)*ny+(j+1),'rest_length':mesh_length*math.sqrt(2)})
                springs.append({'type':'shear','p1':(i+1)*ny+j,'p2':i*ny+(j+1),'rest_length':mesh_length*math.sqrt(2)})
            if i < nx-2:
                springs.append({'type':'bend','p1':idx,'p2':(i+2)*ny+j,'rest_length':mesh_length*2})
            if j < ny-2:
                springs.append({'type':'bend','p1':idx,'p2':i*ny+(j+2),'rest_length':mesh_length*2})
    faces=[]
    for i in range(nx-1):
        for j in range(ny-1):
            idx = i*ny+j
            faces.append([idx, idx+1, idx+ny])
            faces.append([idx+ny, idx+1, idx+ny+1])
    return nodes, springs, faces

# ====================== 驱动函数 ======================
def drive_circle(t, center=np.array([0.45,0.45,-0.05]), radius=0.05, omega=2*math.pi):
    # 注意：这里使用的 t 单位是秒
    # t = 0
    return np.array([center[0]+radius*math.cos(omega*t*10),
                     center[1]+radius*math.sin(omega*t*10),
                     center[2]])

# ====================== 演示 ======================
def demo_single_3d(mesh_length=0.1):

    nx = ny = 10
    dt = mesh_length / 100

    nodes, springs, faces = build_10x10_cloth(mesh_length=mesh_length)
    # 选择 surface_density 使得当 mesh_length=0.1 时 mass_per_particle 约等于旧值 0.01
    # 旧 mass = 0.01 at mesh_length=0.1 => surface_density = 0.01 / (0.1^2) = 1.0
    cloth = SoftClothSimulator(nodes, springs, faces,
                               stiffness_stretch_tension=1000.0,
                               stiffness_bend_tension=50.0,
                               stiffness_shear=800.0,
                               meshlength=mesh_length,
                               surface_density=1.0,
                               spring_damping_coeff=0.5,
                               damping=0.995)

    cloth.add_drive_point(lambda t: drive_circle(t,center=np.array([nx*mesh_length*0.5,nx*mesh_length*0.5,-mesh_length*0.5])))
    # cloth.fix_point(55)

    def idx(i,j): return i*ny+j
    tris = []
    for i in range(nx-1):
        for j in range(ny-1):
            tris.append([idx(i,j), idx(i+1,j), idx(i,j+1)])
            tris.append([idx(i+1,j), idx(i+1,j+1), idx(i,j+1)])
    tris = np.array(tris)

    fig = plt.figure(figsize=(8,6))
    ax3d = fig.add_subplot(111, projection='3d')

    # 预计算并打印临界时间步以便观察
    dt_crit = cloth.critical_timestep()
    print(f"[Info] mesh_length={mesh_length:.4f}, mass_per_particle={cloth.mass_per_particle:.6e}, dt_crit≈{dt_crit:.6e}")

    def update(frame):
        t0 = frame*0.005
        cloth.simulate(5, dt, t0=t0)
        ax3d.clear()
        verts = cloth.get_vertices()
        ax3d.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles=tris,
                          color='lightblue', alpha=0.9, edgecolor='none')
        for dp in cloth.DrivePoints:
            ax3d.scatter(dp['pos'][0], dp['pos'][1], dp['pos'][2], color='red', s=60)
        ax3d.set_xlim(-2*mesh_length, (nx+2)*mesh_length)
        ax3d.set_ylim(-2*mesh_length, (ny+2)*mesh_length)
        ax3d.set_zlim(-(ny+1)*mesh_length,ny*mesh_length)
        ax3d.set_title(f"Cloth Simulation (meshlength={mesh_length})")

    ani = FuncAnimation(fig, update, frames=400, interval=40, blit=False)
    plt.show()

if __name__ == "__main__":
    # 试试两种网格尺度：0.1（原始），以及 1.0（大网格）
    demo_single_3d(mesh_length=1)
    # demo_single_3d(mesh_length=1.0, dt=0.01)  # 若要测试更大的网格，请调整 dt（或让自动子步处理）
