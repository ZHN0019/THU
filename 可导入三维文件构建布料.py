import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import tkinter as tk
from tkinter import simpledialog, filedialog
import math

# 尝试导入CAD文件处理库
try:
    import trimesh

    HAS_TRIMESH = True
    try:
        from scipy.spatial import ConvexHull

        HAS_CONVEXHULL = True
    except ImportError:
        HAS_CONVEXHULL = False
except ImportError:
    HAS_TRIMESH = False
    HAS_CONVEXHULL = False


class ClothMeshGenerator:
    """
    柔性布料网格生成器
    支持通过绘制或导入3D文件生成布料网格
    """

    def __init__(self):
        self.curve_points = None
        self.valid_cells = None
        self.x_points = None
        self.y_points = None
        self.mesh_length = None

    def create_from_drawing(self, canvas_size_mm, mesh_size_mm):
        """
        通过鼠标绘制创建布料轮廓

        Args:
            canvas_size_mm (tuple): 画板大小 (width, height) 单位:mm
            mesh_size_mm (float): 网格大小 单位:mm
        """
        # 创建绘图界面并获取用户绘制的封闭曲线
        app = CurveDrawingApp(canvas_size_mm)
        plt.show(block=True)

        # 获取封闭曲线点
        self.curve_points = app.get_curve_points()

        if len(self.curve_points) < 3:
            raise ValueError("No valid closed curve drawn")

        self.mesh_length = mesh_size_mm
        return self

    def create_from_3d_file(self, filename, mesh_size_mm, plane="XY"):
        """
        通过3D文件创建布料轮廓

        Args:
            filename (str): 3D文件路径
            mesh_size_mm (float): 网格大小 单位:mm
            plane (str): 投影平面，可选"XY", "YZ", "XZ"
        """
        if not HAS_TRIMESH:
            raise ImportError("缺少trimesh库，无法处理CAD文件")

        # 检查文件是否存在
        import os
        if not os.path.exists(filename):
            raise FileNotFoundError(f"文件不存在: {filename}")

        try:
            # 使用trimesh加载文件
            mesh = trimesh.load(filename)

            # 如果是多个几何体，合并它们
            if isinstance(mesh, list):
                mesh = trimesh.util.concatenate(mesh)

            # 根据指定平面获取对应的坐标索引
            plane_indices = {
                "XY": [0, 1],  # X, Y
                "YZ": [1, 2],  # Y, Z
                "XZ": [0, 2]  # X, Z
            }

            if plane not in plane_indices:
                raise ValueError("无效的投影平面，可选: XY, YZ, XZ")

            indices = plane_indices[plane]

            # 获取顶点的指定平面坐标
            vertices_2d = mesh.vertices[:, indices]

            # 使用不同的凸包计算方法
            if HAS_CONVEXHULL:
                # 使用scipy的ConvexHull
                from scipy.spatial import ConvexHull
                hull = ConvexHull(vertices_2d)
                hull_points = vertices_2d[hull.vertices]
            else:
                # 使用边界框近似
                try:
                    # 获取边界
                    bounds_2d = mesh.bounds[:, indices]
                    # 使用边界框的角点作为近似轮廓
                    min_x, min_y = bounds_2d[0]
                    max_x, max_y = bounds_2d[1]
                    hull_points = np.array([
                        [min_x, min_y],
                        [max_x, min_y],
                        [max_x, max_y],
                        [min_x, max_y]
                    ])
                except:
                    # 最简单的方法：直接使用所有顶点
                    hull_points = vertices_2d[::max(1, len(vertices_2d) // 100)]

            # 确保点按顺序排列形成封闭轮廓
            if len(hull_points) > 2:
                # 计算中心点
                center = np.mean(hull_points, axis=0)

                # 按角度排序确保点按顺序排列
                angles = np.arctan2(hull_points[:, 1] - center[1], hull_points[:, 0] - center[0])
                sorted_indices = np.argsort(angles)
                sorted_points = hull_points[sorted_indices]

                # 添加第一个点以闭合轮廓
                sorted_points = np.vstack([sorted_points, sorted_points[0]])

                self.curve_points = sorted_points
            else:
                raise ValueError("无法从3D模型提取有效轮廓")

        except Exception as e:
            raise RuntimeError(f"加载文件时出错: {e}")

        self.mesh_length = mesh_size_mm
        return self

    def generate_mesh(self):
        """
        生成网格
        """
        if self.curve_points is None:
            raise ValueError("尚未设置轮廓点，请先调用create_from_drawing或create_from_3d_file方法")

        # 获取最小外接矩形
        min_x, max_x, min_y, max_y = self._get_bounding_rectangle(self.curve_points)

        # 创建网格
        self.x_points, self.y_points = self._create_grid(min_x, max_x, min_y, max_y, self.mesh_length)

        # 检查每个网格单元
        self.valid_cells = self._check_grid_cell_vertices(self.x_points, self.y_points, self.curve_points)

        info = self.get_mesh_info()
        print(f"总网格数: {info['total_grids']}")
        print(f"有效网格数: {info['valid_grids']}")
        print(f"有效比例: {info['valid_ratio']:.2%}")


        return self

    def show(self):
        """
        可视化结果
        """
        if self.curve_points is None or self.valid_cells is None:
            raise ValueError("尚未生成网格，请先调用generate_mesh方法")

        self._visualize_results(self.curve_points, self.x_points, self.y_points, self.valid_cells, self.mesh_length)
        return self

    def get_mesh_info(self):
        """
        获取网格信息
        """
        if self.x_points is None or self.y_points is None or self.valid_cells is None:
            raise ValueError("尚未生成网格，请先调用generate_mesh方法")

        total_grids = (len(self.x_points) - 1) * (len(self.y_points) - 1)
        valid_grids = len(self.valid_cells)

        return {
            "total_grids": total_grids,
            "valid_grids": valid_grids,
            "valid_ratio": valid_grids / total_grids if total_grids > 0 else 0
        }

    def _point_in_polygon(self, point, polygon):
        """
        判断点是否在多边形内（射线法）
        """
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _get_bounding_rectangle(self, points):
        """
        获取封闭曲线的最小外接矩形
        """
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        return min_x, max_x, min_y, max_y

    def _create_grid(self, min_x, max_x, min_y, max_y, mesh_length):
        """
        创建均匀网格
        """
        x_points = np.arange(min_x, max_x + mesh_length, mesh_length)
        y_points = np.arange(min_y, max_y + mesh_length, mesh_length)

        return x_points, y_points

    def _check_grid_cell_vertices(self, x_points, y_points, curve_points):
        """
        检查每个网格单元是否有至少两个顶点在封闭曲线内
        """
        valid_cells = []

        for i in range(len(x_points) - 1):
            for j in range(len(y_points) - 1):
                # 获取网格的四个顶点
                vertices = [
                    (x_points[i], y_points[j]),
                    (x_points[i + 1], y_points[j]),
                    (x_points[i + 1], y_points[j + 1]),
                    (x_points[i], y_points[j + 1])
                ]

                # 统计在曲线内的顶点数
                inside_count = 0
                for vertex in vertices:
                    if self._point_in_polygon(vertex, curve_points):
                        inside_count += 1

                # 如果至少有两个顶点在曲线内，则标记为有效网格
                if inside_count >= 2:
                    valid_cells.append((i, j, vertices))

        return valid_cells

    def _visualize_results(self, curve_points, x_points, y_points, valid_cells, mesh_length):
        """
        可视化结果
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # 绘制原始封闭曲线
        curve_x = [p[0] for p in curve_points]
        curve_y = [p[1] for p in curve_points]
        ax.plot(curve_x, curve_y, 'b-', linewidth=2, label='Closed Curve')
        ax.fill(curve_x, curve_y, alpha=0.2, color='lightblue')

        # 绘制所有网格点
        for x in x_points:
            for y in y_points:
                ax.plot(x, y, 'k.', markersize=2)

        # 绘制有效的网格单元
        for i, j, vertices in valid_cells:
            # 绘制网格的边线（四条边）
            vertices_closed = vertices + [vertices[0]]  # 闭合多边形
            grid_x = [v[0] for v in vertices_closed]
            grid_y = [v[1] for v in vertices_closed]
            ax.plot(grid_x, grid_y, 'g-', linewidth=1.5, alpha=0.7)

            # 绘制对角线
            # 第一条对角线
            ax.plot([vertices[0][0], vertices[2][0]],
                    [vertices[0][1], vertices[2][1]],
                    'r--', linewidth=1, alpha=0.7)
            # 第二条对角线
            ax.plot([vertices[1][0], vertices[3][0]],
                    [vertices[1][1], vertices[3][1]],
                    'r--', linewidth=1, alpha=0.7)

            # 绘制网格点
            for vertex in vertices:
                ax.plot(vertex[0], vertex[1], 'go', markersize=4)

        ax.set_xlim(min(x_points) - mesh_length, max(x_points) + mesh_length)
        ax.set_ylim(min(y_points) - mesh_length, max(y_points) + mesh_length)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Grid Division Result (Mesh Size: {mesh_length}mm)')
        ax.legend()

        plt.tight_layout()
        plt.show()


class CurveDrawingApp:
    def __init__(self, canvas_size_mm):
        self.points = []
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_title('Draw a closed curve, double click to finish')
        # 将mm转换为matplotlib单位（假设1单位=1mm）
        canvas_width, canvas_height = canvas_size_mm
        self.ax.set_xlim(0, canvas_width)
        self.ax.set_ylim(0, canvas_height)
        self.ax.grid(True)
        self.line, = self.ax.plot([], [], 'b-', linewidth=2)
        self.poly_line = None
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_double = self.fig.canvas.mpl_connect('button_press_event', self.on_double_click)
        self.drawing = True
        self.is_double_click = False
        self.click_times = []

    def on_click(self, event):
        if not self.drawing or event.inaxes != self.ax:
            return

        self.points.append([event.xdata, event.ydata])
        self.update_plot()

    def on_release(self, event):
        pass

    def on_double_click(self, event):
        if event.dblclick and self.drawing and event.inaxes == self.ax and len(self.points) >= 2:
            self.drawing = False
            self.complete_curve()
            self.fig.canvas.mpl_disconnect(self.cid_press)
            self.fig.canvas.mpl_disconnect(self.cid_release)
            self.fig.canvas.mpl_disconnect(self.cid_double)

    def update_plot(self):
        if len(self.points) > 0:
            x = [p[0] for p in self.points]
            y = [p[1] for p in self.points]
            self.line.set_data(x, y)
            self.fig.canvas.draw()

    def complete_curve(self):
        # 连接首尾形成封闭曲线
        if len(self.points) >= 2:
            self.points.append(self.points[0])  # 连接首尾
            self.update_plot()

    def get_curve_points(self):
        return np.array(self.points)


# 使用示例
if __name__ == "__main__":
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # # 方法1: 通过绘制创建
    # try:
    #     generator = ClothMeshGenerator()
    #     generator.create_from_drawing(canvas_size_mm=(200, 200), mesh_size_mm=10)
    #     generator.generate_mesh()
    #     generator.show()
    #     info = generator.get_mesh_info()
    # except Exception as e:
    #     print(f"Error: {e}")

    # 方法2: 通过3D文件创建
    try:
        generator = ClothMeshGenerator()
        generator.create_from_3d_file(filename="Handkerchief.stl", mesh_size_mm=5, plane="XZ")
        generator.generate_mesh()
        generator.show()
    except Exception as e:
        print(f"Error: {e}")