import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

# ====================== SoftClothSimulator ======================
class SoftClothSimulator:
    def __init__(self, nodes, springs, faces=None,
                 stiffness_stretch_tension=1000.0,
                 stiffness_bend_tension=50.0,
                 stiffness_shear=800.0,
                 compression_to_tension_ratio=0.1,
                 damping=0.995,
                 spring_damping_coeff=0.5,
                 meshlength=0.1,
                 surface_density=1.0,
                 momentum_loss=0.4,
                 collision_restitution=0.3,
                 drive_influence_ratio=1):
        self.meshlength = float(meshlength)
        self.positions = np.array(nodes, dtype=float)
        self.velocities = np.zeros_like(self.positions)
        self.forces = np.zeros_like(self.positions)
        self.springs = springs
        self.faces = np.array(faces) if faces is not None else None
        self.num_particles = len(self.positions)
        self.fixed_points = np.zeros(self.num_particles, dtype=bool)
        self.initial_positions = self.positions.copy()
        self.DrivePoints = []
        self.drive_influence_ratio = drive_influence_ratio

        # 物理参数
        self.stiffness_stretch_tension = float(stiffness_stretch_tension)
        self.stiffness_stretch_compression = stiffness_stretch_tension * compression_to_tension_ratio
        self.stiffness_bend_tension = float(stiffness_bend_tension)
        self.stiffness_bend_compression = self.stiffness_bend_tension * compression_to_tension_ratio
        self.stiffness_shear = float(stiffness_shear)
        self.damping = float(damping)
        self.spring_damping_coeff_base = float(spring_damping_coeff)
        self.spring_damping_coeff = self.spring_damping_coeff_base * self.meshlength
        self.surface_density = float(surface_density)
        self.mass_per_particle = self.surface_density * (self.meshlength ** 2)
        self.gravity = np.array([0, 0, -9.81])
        self.min_distance = self.meshlength
        self.momentum_loss = momentum_loss
        self.collision_restitution = collision_restitution

        # 驱动点投影方法（重心法）
        self._compute_initial_projection = self._compute_initial_projection_barycentric

    def fix_point(self, idx):
        self.fixed_points[idx] = True

    def add_drive_point(self, func):
        pos = np.array(func(0.0))
        self.DrivePoints.append({'func': func, 'pos': pos.copy(), 'prev_pos': pos.copy(),
                                 'collision_info': None, 'initial_proj': None})

    def update_drive_points(self, t):
        for dp in self.DrivePoints:
            dp['prev_pos'] = dp['pos'].copy()
            dp['pos'] = np.array(dp['func'](t))

    def _compute_initial_projection_barycentric(self, dp):
        f_idx, bary = dp['collision_info']
        if f_idx is None or bary is None:
            return None
        q0 = (bary[0] * self.initial_positions[f_idx[0]] +
              bary[1] * self.initial_positions[f_idx[1]] +
              bary[2] * self.initial_positions[f_idx[2]])
        return q0

    def reset_forces(self):
        self.forces.fill(0.0)
        for i in range(self.num_particles):
            self.forces[i] += self.gravity * self.mass_per_particle

    def compute_spring_forces(self):
        for spring in self.springs:
            i, j = spring['p1'], spring['p2']
            p1, p2 = self.positions[i], self.positions[j]
            v1, v2 = self.velocities[i], self.velocities[j]
            delta = p2 - p1
            dist = np.linalg.norm(delta)
            if dist < 1e-12:
                continue
            direction = delta / dist
            displacement = dist - spring['rest_length']
            if spring['type'] == 'stretch':
                stiffness = self.stiffness_stretch_tension if displacement >= 0 else self.stiffness_stretch_compression
            elif spring['type'] == 'shear':
                stiffness = self.stiffness_shear
            else:
                stiffness = self.stiffness_bend_tension if displacement >= 0 else self.stiffness_bend_compression
            force = stiffness * displacement * direction
            damping_force = self.spring_damping_coeff * np.dot(v2 - v1, direction) * direction
            total_force = force + damping_force
            self.forces[i] += total_force
            self.forces[j] -= total_force

    def compute_drive_spring_forces(self, dt_sub):
        for dp in self.DrivePoints:
            if dp['collision_info'] is None:
                continue
            q0 = self._compute_initial_projection(dp)
            if q0 is None:
                continue
            dp['initial_proj'] = q0
            for i in range(self.num_particles):
                p0 = self.initial_positions[i]
                rest_len = np.linalg.norm(q0 - p0)
                if rest_len < 1e-12:
                    continue
                # 改进衰减：以布料边长或平均网格直径为参考
                decay = np.exp(-rest_len / (5*self.meshlength))
                delta = self.positions[i] - dp['pos']
                dist = np.linalg.norm(delta)
                if dist < 1e-12:
                    continue
                direction = delta / dist
                displacement = dist - rest_len
                stiffness = self.stiffness_stretch_tension if displacement >= 0 else self.stiffness_stretch_compression
                force = stiffness * displacement * direction * decay
                # 驱动阻尼按子步 dt_sub 修正
                dp_vel_sub = (dp['pos'] - dp['prev_pos']) / dt_sub
                rel_vel = self.velocities[i] - dp_vel_sub
                damping_force = self.spring_damping_coeff * np.dot(rel_vel, direction) * direction * decay
                self.forces[i] += force + damping_force

    def resolve_contacts(self, dt_sub):
        for dp in self.DrivePoints:
            dp_pos = dp['pos']
            dp_prev = dp['prev_pos']
            dp_vel = (dp_pos - dp_prev) / dt_sub
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
            if closest_idx is not None:
                n = (self.positions[closest_idx] - dp_pos) / closest_dist
                v_rel = np.dot(self.velocities[closest_idx] - dp_vel, n)
                if v_rel < 0:
                    self.velocities[closest_idx] -= v_rel * n
                    self.velocities[closest_idx] += dp_vel * n
                penetration = self.min_distance - closest_dist
                if penetration > 0:
                    self.positions[closest_idx] += n * penetration
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

    def critical_timestep(self):
        C = 0.2
        k_candidates = [self.stiffness_stretch_tension, self.stiffness_shear, self.stiffness_bend_tension]
        k_max = max(k_candidates)
        if k_max <= 0:
            return 1e-6
        dt_crit = C * math.sqrt(self.mass_per_particle / k_max)
        return dt_crit

    def integrate(self, dt, t_global=0.0):
        self.update_drive_points(t_global)
        dt_crit = self.critical_timestep()
        if dt_crit <= 0:
            dt_crit = 1e-6
        n_sub = max(1, int(math.ceil(dt / dt_crit)))
        dt_sub = dt / n_sub
        if n_sub > 200:
            warnings.warn(f"Very many substeps ({n_sub}) for stability. dt={dt}, dt_crit={dt_crit:.3e}")
        for _ in range(n_sub):
            self.reset_forces()
            self.compute_spring_forces()
            self.compute_drive_spring_forces(dt_sub)
            acceleration = self.forces / max(self.mass_per_particle, 1e-12)
            movable = ~self.fixed_points
            self.velocities[movable] += acceleration[movable] * dt_sub
            self.velocities[movable] *= self.damping
            self.resolve_contacts(dt_sub)
            self.positions[movable] += self.velocities[movable] * dt_sub

    def simulate(self, steps, dt, t0=0.0):
        t = t0
        for _ in range(steps):
            self.integrate(dt, t)
            t += dt

    def get_vertices(self):
        return self.positions

# ====================== 网格构建 ======================
def build_10x10_cloth(mesh_length=0.1):
    nx = ny = 10
    nodes = [[i*mesh_length,j*mesh_length,0.0] for i in range(nx) for j in range(ny)]
    nodes = np.array(nodes)
    springs = []
    for i in range(nx):
        for j in range(ny):
            idx = i*ny+j
            if i<nx-1: springs.append({'type':'stretch','p1':idx,'p2':(i+1)*ny+j,'rest_length':mesh_length})
            if j<ny-1: springs.append({'type':'stretch','p1':idx,'p2':i*ny+(j+1),'rest_length':mesh_length})
            if i<nx-1 and j<ny-1:
                springs.append({'type':'shear','p1':idx,'p2':(i+1)*ny+(j+1),'rest_length':mesh_length*math.sqrt(2)})
                springs.append({'type':'shear','p1':(i+1)*ny+j,'p2':i*ny+(j+1),'rest_length':mesh_length*math.sqrt(2)})
            if i<nx-2: springs.append({'type':'bend','p1':idx,'p2':(i+2)*ny+j,'rest_length':2*mesh_length})
            if j<ny-2: springs.append({'type':'bend','p1':idx,'p2':i*ny+(j+2),'rest_length':2*mesh_length})
    faces=[]
    for i in range(nx-1):
        for j in range(ny-1):
            idx=i*ny+j
            faces.append([idx, idx+1, idx+ny])
            faces.append([idx+ny, idx+1, idx+ny+1])
    return nodes, springs, faces

# ====================== 驱动函数 ======================
def drive_circle(t, center=np.array([0.45,0.45,-0.05]), radius=0.05, omega=2*math.pi):
    return np.array([center[0]+radius*math.cos(omega*t*10),
                     center[1]+radius*math.sin(omega*t*10),
                     center[2]])

# ====================== 演示 ======================
def demo_single_3d(mesh_length=0.1):
    nx=ny=10
    dt=mesh_length/100
    nodes, springs, faces = build_10x10_cloth(mesh_length)
    cloth = SoftClothSimulator(nodes, springs, faces,
                               stiffness_stretch_tension=1000.0,
                               stiffness_bend_tension=50.0,
                               stiffness_shear=800.0,
                               meshlength=mesh_length,
                               surface_density=1.0,
                               spring_damping_coeff=0.5,
                               damping=0.995)
    cloth.add_drive_point(lambda t: drive_circle(t, center=np.array([nx*mesh_length*0.5,nx*mesh_length*0.5,-mesh_length*0.5])))
    tris=[]
    for i in range(nx-1):
        for j in range(ny-1):
            tris.append([i*ny+j,(i+1)*ny+j,i*ny+(j+1)])
            tris.append([(i+1)*ny+j,(i+1)*ny+(j+1),i*ny+(j+1)])
    tris=np.array(tris)
    fig = plt.figure(figsize=(8,6))
    ax3d = fig.add_subplot(111,projection='3d')
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
        ax3d.set_xlim(-2*mesh_length,(nx+2)*mesh_length)
        ax3d.set_ylim(-2*mesh_length,(ny+2)*mesh_length)
        ax3d.set_zlim(-(ny + 1) * mesh_length, 2 * mesh_length)
        ax3d.set_box_aspect([1, 1, 0.5])
        ax3d.set_title(f"Cloth Simulation t={t0:.3f}s")
        return []

    anim = FuncAnimation(fig, update, frames=200, interval=30, blit=False)
    plt.show()


if __name__ == "__main__":
    demo_single_3d(mesh_length=0.05)
