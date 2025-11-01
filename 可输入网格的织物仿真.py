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
                 stiffness_stretch_tension=1000.0,   # per-spring axial stiffness (N/m equivalent in discrete model)
                 stiffness_bend_tension=50.0,
                 stiffness_shear=800.0,
                 compression_to_tension_ratio=0.1,
                 damping=0.995,                      # velocity global damping factor (dimensionless)
                 spring_damping_coeff=0.5,           # damping coefficient (in same units as spring forces per velocity)
                 meshlength=0.1,
                 surface_density=1.0,                # mass per unit area (kg / m^2) -- choose to match default mass
                 thickness=0.001,                    # cloth thickness (m) used if needed (not strictly necessary)
                 momentum_loss=0.4,
                 collision_restitution=0.3,
                 drive_influence_ratio=1):
        """
        关键缩放思路：
          - mass_per_particle ~ surface_density * meshlength^2
          - treat the provided stiffness_* as per-spring stiffness (do NOT multiply again by mesh scale)
          - critical dt ~ C * sqrt(mass_per_particle / k_max) -> 若用户 dt 太大，则自动进行子步细分
        """
        # 设置默认的驱动点投影方法
        self._compute_initial_projection = self._compute_initial_projection_xy_projection
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

        self.gravity = np.array([0,0,-9.81])
        self.min_distance = self.meshlength  # 碰撞判阈按网格长度
        self.momentum_loss = momentum_loss
        self.collision_restitution = collision_restitution

        # 防穿透
        self.contact_guard_time = meshlength
        self.contact_frames_remaining = np.zeros(self.num_particles, dtype=int)

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

    # ---------- 驱动弹簧力，仅在碰撞时 ----------
    def compute_drive_spring_forces(self, dt_sub):
        """
        驱动点碰撞时产生的弹簧力（修正版）。
        这里把驱动点速度按 dt_sub 计算，确保阻尼项单位一致（m/s）。
        """
        for dp in self.DrivePoints:
            if dp['collision_info'] is None:
                continue  # 没有碰撞就跳过

            # 使用封装的方法计算初始投影位置
            q0 = self._compute_initial_projection(dp)
            if q0 is None:
                dp['initial_proj'] = None
                continue

            dp['initial_proj'] = q0

            # 驱动点在子步内的速度（用 dp.pos - dp.prev_pos），与子步时间尺度匹配
            dp_vel = (dp['pos'] - dp['prev_pos']) / max(dt_sub, 1e-12)

            for i in range(self.num_particles):
                p0 = self.initial_positions[i]
                # 弹簧原始长度（从驱动相交点到该粒子的初始距离）
                rest_len = np.linalg.norm(q0 - p0)
                if rest_len < 1e-12:
                    continue

                # 每增加一个 meshlength 衰减一半
                decay = 0.5 ** (rest_len / self.meshlength)
                # decay = 1

                delta = self.positions[i] - dp['pos']
                dist = np.linalg.norm(delta)
                if dist < 1e-12:
                    continue
                direction = delta / dist
                displacement = dist - rest_len
                # print(displacement)
                stiffness = self.stiffness_stretch_tension if displacement >= 0 else self.stiffness_stretch_compression
                # print(stiffness)

                # 弹簧拉力（带衰减）
                force = stiffness * displacement * direction * decay
                print(force)

                # 驱动弹簧阻尼：相对速度（现在 dp_vel 已是 m/s 量级）
                rel_vel = self.velocities[i] - dp_vel
                damping_force = self.spring_damping_coeff * np.dot(rel_vel, direction) * direction * decay
                # print(damping_force)

                self.forces[i] += force + damping_force


    # ---------- 碰撞 ----------
    def resolve_contacts(self, dt_sub):
        """
        完全非弹性碰撞处理（修正版）。
        修复：不重复地把粒子的法向速度设置为驱动点法向速度（删除重复加成）。
        """
        for dp in self.DrivePoints:
            dp_pos = dp['pos']
            dp_prev = dp['prev_pos']
            dp_vel = (dp_pos - dp_prev) / max(dt_sub, 1e-12)

            # ---------------- 寻找距离最近的碰撞点 ----------------
            closest_dist = 1e12
            closest_idx = None
            for i in range(self.num_particles):
                if self.fixed_points[i]:
                    continue
                delta = self.positions[i] - dp_pos
                dist = np.linalg.norm(delta)
                if dist < self.min_distance and dist > 1e-8 and dist < closest_dist:
                    closest_dist = dist
                    closest_idx = i

            # ---------------- 处理最近碰撞点 ----------------
            if closest_idx is not None and closest_dist < 1e12:
                n = (self.positions[closest_idx] - dp_pos) / closest_dist
                v_rel = np.dot(self.velocities[closest_idx] - dp_vel, n)

                # --- 速度修正（完全非弹性） ---
                if v_rel < 0:
                    # 正确做法：将粒子速度的法向分量替换为驱动点的法向速度（一次操作）
                    # 计算当前粒子法向分量
                    vel_n = np.dot(self.velocities[closest_idx], n)
                    # 用驱动点的法向速度替换之
                    self.velocities[closest_idx] += (dp_vel - vel_n) * n
                    # （注意：不要再额外加 dp_vel * n，否则会重复加成）

                # --- 位置修正（关键） ---
                penetration = self.min_distance - closest_dist
                if penetration > 0:
                    self.positions[closest_idx] += n * penetration

                # 更新驱动弹簧碰撞信息（基于当前三角形）
                min_tri_dist = 1e12
                best_tri = None
                best_bary = None
                if self.faces is not None:
                    for f_idx in self.faces:
                        if closest_idx not in f_idx:
                            continue
                        a, b, c = self.positions[f_idx[0]], self.positions[f_idx[1]], self.positions[f_idx[2]]
                        q, d, bary = point_triangle_distance(dp_pos, a, b, c)
                        if d < min_tri_dist:
                            min_tri_dist = d
                            best_tri = f_idx
                            best_bary = bary
                dp['collision_info'] = (best_tri, best_bary)
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
            # 为了让驱动阻尼量纲正确，在 compute_drive_spring_forces 内计算的 rel_vel
            # 使用已有 dp['pos'] 和 dp['prev_pos']，但那两个是基于全大步更新的，精确上应该按子步更新。
            # 为简洁，这里把驱动阻尼项的尺度修正为除以 dt_sub：
            # 我们先记录 forces_before, 然后在 compute_drive_spring_forces 里用 rel_vel raw，
            # 但最终涌入的 damping_force 需要除以 dt_sub — 下面在调用后修正：
            # self.compute_drive_spring_forces()  # 注意内部产生的阻尼项有 dt_scale 问题
            self.compute_drive_spring_forces(dt_sub)
            # 修正驱动阻尼贡献：由于 compute_drive_spring_forces 中 rel_vel 使用了 (dp.pos - dp.prev_pos)（未除以 dt_sub）
            # 并乘以 spring_damping_coeff（已经乘了 meshlength），我们这里用一个简单修正：
            # （更严谨的做法是把驱动点速度随子步同步更新；这里折中：把驱动阻尼按 dt_sub 调整）
            # 直接在 forces 中对每项不做逐项区分，采用近似修正：forces *= 1 (skip) — 因为我们在 compute_spring_forces 已经有阻尼项。
            # （如果你需要更精确，可把驱动点速度按子步内线性插值并在每个子步前 update_drive_points_sub）
            acceleration = self.forces / max(self.mass_per_particle, 1e-12)

            # 更新速度（排除固定点）
            movable = ~self.fixed_points
            self.velocities[movable] += acceleration[movable] * dt_sub
            # 全局速度阻尼（无量纲）
            self.velocities[movable] *= self.damping

            # 碰撞处理（会修改 velocities 并更新驱动碰撞信息）
            self.resolve_contacts(dt_sub)

            # 更新位置
            self.positions[movable] += self.velocities[movable] * dt_sub

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
