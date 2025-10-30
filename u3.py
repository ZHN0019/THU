#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
机器人双臂控制系统架构
支持末端位姿控制、关节角度读取等功能
"""

from __future__ import annotations
import numpy as np
import mujoco
import mujoco.viewer
import time
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Slerp
from xboxer import XboxController
from threading import Thread, Lock
import threading
import mink
import logging
logging.getLogger('loop_rate_limiters').setLevel(logging.CRITICAL)

# ─────────────── 默认参数 ───────────────
XML_PATH = "urdf/1029scene.xml"
MOCAP_LEFT = "target_left"
MOCAP_RIGHT = "target_right"
SITE_LEFT = "attachment_site_Left"
SITE_RIGHT = "attachment_site_Right"
POS_L = np.array([0.30, 0.255, 1.885])
POS_R = np.array([0.30, -0.255, 1.885])
QUAT_ID = np.array([1.0, 0.0, 0.0, 0.0])
DEFAULT_TIMESTEP = 1e-2  # 2500 Hz
INTERFACE_FRAMERATE = 10    # 可视化界面的动画刷新率
CACU_REPEAT = 4    # 迭代次数

REF_LEFT = {
    "Joint1^Left": 0.7749, "Joint2^Left": 0.1603, "Joint3^Left": 0.0, "Joint3_2^Left": 1.0457, "Joint4^Left": -1.0457,
    "Joint5_1_1^Left": 0.0205, "Joint5_2_2^Left": 0.0205, "Joint5_2_3^Left": -0.0205, "Joint6^Left": 0.0205,
    "Joint5_2_1^Left": 0.1663, "Joint5_1_2^Left": 0.1663, "Joint5_1_3^Left": -0.1663,
    "Joint7^Left": 0.1624,
    "thumb_1_left": -0.72, "thumb_2_left": -0.72, "thumb_3_left": -0.877,
    "index_1_left": 0.0, "index_2_left": 0.0,
    "middle_1_left": -1.5, "middle_2_left": -1.5,
    "ring_1_left": -1.5, "ring_2_left": -1.5,
    "little_1_left": -1.5, "little_2_left": -1.5,
}
REF_RIGHT = {
    "Joint1^Right": 0.7746, "Joint2^Right": 0.1603, "Joint3^Right": 0.0, "Joint3_2^Right": 1.0461, "Joint4^Right": 1.0461, 
    "Joint5_1_1^Right": 0.0205, "Joint5_2_2^Right": -0.0205, "Joint5_2_3^Right": 0.0205, "Joint6^Right": 0.0205,
    "Joint5_2_1^Right": 0.1663,"Joint5_1_2^Right": 0.1663, "Joint5_1_3^Right": -0.1663,
    "Joint7^Right": 0.1624,
    "thumb_1_right": -1.42, "thumb_2_right": -0.525, "thumb_3_right": -1.28,
    "index_1_right": 0.0, "index_2_right": 0.0,
    "middle_1_right": -1.24, "middle_2_right": -1.57,
    "ring_1_right": -1.5, "ring_2_right": -1.5,
    "little_1_right": -1.5, "little_2_right": -1.5,
}

# 添加到u0.py中，放在类定义之前
class PreferredPoseTask(mink.Task):
    def __init__(self, model, names, q_ref, w):
        self.idx = [model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
        self.qr = np.asarray(q_ref, float)
        super().__init__(cost=np.asarray(w, float))

    def num_errors(self): return len(self.idx)
    def compute_error(self, cfg): return cfg.q[self.idx] - self.qr
    def compute_jacobian(self, cfg):
        J = np.zeros((len(self.idx), cfg.q.size))
        J[np.arange(len(self.idx)), self.idx] = 1.0
        return J

class CoupleTask(mink.Task):
    """ q_i + sign*q_j ≈ 0 """
    def __init__(self, model, j1, j2, sign=1.0, cost=25.0):
        self.q1 = model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, j1)]
        self.q2 = model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, j2)]
        self.sign = sign
        super().__init__(cost=np.array([cost], float))

    def num_errors(self): return 1
    def compute_error(self, cfg):
        return np.array([cfg.q[self.q1] + self.sign * cfg.q[self.q2]])

    def compute_jacobian(self, cfg):
        J = np.zeros((1, cfg.q.size))
        J[0, self.q1] = 1
        J[0, self.q2] = self.sign
        return J

class MinimalJointMovementTask(mink.Task):
    """
    最小关节运动任务（支持不同关节不同权重）
    """
    def __init__(self, model, joint_names, costs=None):
        """
        初始化最小关节运动任务
        
        Args:
            model: MuJoCo模型
            joint_names: 关节名称列表
            costs: 各关节的权重系数列表，如果为None则使用默认值
        """
        self.model = model
        # 获取关节索引
        self.joint_indices = [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in joint_names
        ]
        # 存储上一时刻的关节角度
        self.last_q = None
        
        # 设置权重
        if costs is None:   # 默认权重
            n_arm_joints = 7
            costs = [0.2] * n_arm_joints + [0.01] * (len(joint_names) - n_arm_joints)
        
        super().__init__(cost=np.asarray(costs))

    def num_errors(self):
        return len(self.joint_indices)

    def compute_error(self, cfg):
        if self.last_q is None:
            self.last_q = cfg.q[self.joint_indices].copy()
            return np.zeros(len(self.joint_indices))
        else:
            delta = cfg.q[self.joint_indices] - self.last_q
            self.last_q = cfg.q[self.joint_indices].copy()
            return delta

    def compute_jacobian(self, cfg):
        J = np.zeros((len(self.joint_indices), cfg.q.size))
        for i, idx in enumerate(self.joint_indices):
            J[i, idx] = 1.0
        return J

    def reset_reference(self, q_current):
        if self.last_q is not None:
            self.last_q = q_current[self.joint_indices].copy()

class JointLimitTask(mink.Task):
    """关节限制任务，防止关节超出限制范围"""
    def __init__(self, model, joint_names, cost=100.0):
        self.model = model
        self.joint_indices = []
        self.lower_limits = []
        self.upper_limits = []
        
        for name in joint_names:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            # 获取关节的控制范围
            if model.jnt_limited[jnt_id] == 1:  # 如果关节有限制
                self.joint_indices.append(model.jnt_qposadr[jnt_id])
                self.lower_limits.append(model.jnt_range[jnt_id, 0])
                self.upper_limits.append(model.jnt_range[jnt_id, 1])
            else:
                # 如果没有定义限制，使用默认值
                self.joint_indices.append(model.jnt_qposadr[jnt_id])
                self.lower_limits.append(-np.inf)
                self.upper_limits.append(np.inf)
        
        self.lower_limits = np.array(self.lower_limits)
        self.upper_limits = np.array(self.upper_limits)
        # 权重数量应该与错误数量一致，即关节数量的2倍（上下限）
        super().__init__(cost=np.array([cost] * (len(joint_names) * 2)))

    def num_errors(self):
        # 每个关节上下限各一个约束
        return len(self.joint_indices) * 2

    def compute_error(self, cfg):
        errors = np.zeros(self.num_errors())
        for i, idx in enumerate(self.joint_indices):
            q = cfg.q[idx]
            # 下限约束 (负值表示违反约束)
            if self.lower_limits[i] != -np.inf and q < self.lower_limits[i]:
                errors[i*2] = q - self.lower_limits[i]
            # 上限约束 (正值表示违反约束)
            if self.upper_limits[i] != np.inf and q > self.upper_limits[i]:
                errors[i*2+1] = q - self.upper_limits[i]
        return errors

    def compute_jacobian(self, cfg):
        J = np.zeros((self.num_errors(), cfg.q.size))
        for i, idx in enumerate(self.joint_indices):
            J[i*2, idx] = 1     # 下限约束雅可比
            J[i*2+1, idx] = 1   # 上限约束雅可比
        return J

class RobotController:
    """ 机器人控制器主类,   提供统一的API接口用于控制双臂机器人 """
    # 初始化控制器
    def __init__(self, xml_path=None, timestep=None, framerate=None, interpolation_density=5):
        """
        初始化控制器
        
        Args:
            xml_path: URDF/MJCF模型路径
            timestep: 仿真时间步长
            framerate: 显示刷新率
            interpolation_density: 插值密度（每单位距离 1m 的插值点数），默认50
        """
        self.xml_path = xml_path or XML_PATH
        self.timestep = timestep or DEFAULT_TIMESTEP
        self.interpolation_points = interpolation_density  # 插值密度参数

        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        # 增加约束和关节的最大数量（预防性措施）

        self.model.opt.timestep = self.timestep
        self.data = mujoco.MjData(self.model)

        # 初始化状态
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # 使用u8.py的DualIK结构
        self.cfg = mink.Configuration(self.model)
        self.cfg.data = self.data
        self.max_vel = 10
        self.cfg.q[:] = self.data.qpos

        # 末端 到达 任务权重：3 平移 + 2 旋转，FrameTask内部会为每个自由度分配默认权重
        self.tL = mink.FrameTask(SITE_LEFT, "site", 3, 1)       # 左臂任务（让左臂标记点SITE_LEFT  去接近控制点）
        self.tR = mink.FrameTask(SITE_RIGHT, "site", 3, 1)      # 右臂任务（让右臂标记点SITE_RIGHT 去接近控制点）
        self.tL._base_cost = self.tL.cost.copy()
        self.tR._base_cost = self.tR.cost.copy()

        # 参考姿态任务

        """ [
        'Joint1^Left', 'Joint2^Left', 'Joint3^Left', 'Joint3_2^Left', 'Joint5_1_1^Left', 'Joint5_2_1^Left', 'Joint7^Left', 
        'thumb_1_left', 'thumb_2_left', 'thumb_3_left', 'index_1_left', 'index_2_left', 'middle_1_left', 'middle_2_left', 'ring_1_left', 'ring_2_left', 'little_1_left', 'little_2_left', 
        'Joint1^Right', 'Joint2^Right', 'Joint3^Right', 'Joint3_2^Right', 'Joint5_1_1^Right', 'Joint5_2_1^Right', 'Joint7^Right', 
        'thumb_1_right', 'thumb_2_right', 'thumb_3_right', 'index_1_right', 'index_2_right', 'middle_1_right', 'middle_2_right', 'ring_1_right', 'ring_2_right', 'little_1_right', 'little_2_right'
        ]"""

        
        self.pref = PreferredPoseTask(
            self.model, 
            list(REF_LEFT) + list(REF_RIGHT), 
            list(REF_LEFT.values()) + list(REF_RIGHT.values()), 
            #    抬臂    展背      扭肩    曲肘         翻腕    翻腕        扭腕       这里是维持参考姿态的权重，越高越不容易动
            ([  0.02,   0.2,     0.1,   0.5,     0.02,   0.02,   0.02] + [25.0] * (len(REF_LEFT) - 7)) * 2)


        # 耦合约束，关节联动
        COUPLE_EQ = [
            ("Joint3_2^Left", "Joint4^Left", 1.0),
            ("Joint5_1_2^Left", "Joint5_1_3^Left", 1.0),
            ("Joint5_1_2^Left", "Joint5_2_1^Left", 1.0),
            ("Joint5_1_1^Left", "Joint5_2_2^Left", -1.0),
            ("Joint5_1_1^Left", "Joint5_2_3^Left", 1.0),
            ("Joint5_1_1^Left", "Joint6^Left", -1.0),
            ("Joint3_2^Right", "Joint4^Right", -1.0),
            ("Joint5_1_2^Right", "Joint5_1_3^Right", 1.0),
            ("Joint5_1_2^Right", "Joint5_2_1^Right", 1.0),
            ("Joint5_1_1^Right", "Joint5_2_2^Right", 1.0),
            ("Joint5_1_1^Right", "Joint5_2_3^Right", -1.0),
            ("Joint5_1_1^Right", "Joint6^Right", -1.0),
        ]
        self.cpl_tasks = [CoupleTask(self.model, a, b, s) for a, b, s in COUPLE_EQ]
        # 最小关节运动任务
        self.min_movement_task = MinimalJointMovementTask(
            self.model, 
            list(REF_LEFT)[:7] + list(REF_RIGHT)[:7], 
            costs=[  2,   2,     1,   5,     2.5,   2.5,   2] *2)  
        # 创建关节限制任务
        self.joint_limit_task = JointLimitTask(
            self.model, 
            list(REF_LEFT) + list(REF_RIGHT), 
            cost=25.0)
        # 将其放在任务列表的前面以提高优先级
        # self.tasks = [self.tL, self.tR, self.pref, self.min_movement_task, self.joint_limit_task] + self.cpl_tasks
        self.tasks = [self.tL, self.tR, self.min_movement_task, self.joint_limit_task] + self.cpl_tasks

        # 初始化mocap位姿
        self.mocap_left_id = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, MOCAP_LEFT)]
        self.mocap_right_id = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, MOCAP_RIGHT)]

        # 记录关节名称索引
        self.joints = list(REF_LEFT)[:7] + list(REF_RIGHT)[:7]
        self.jidx = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in self.joints
        ]

        # 初始化viewer
        self.current_pose_left = None   # 跟踪左臂当前位姿
        self.current_pose_right = None  # 跟踪右臂当前位姿
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data,
            show_left_ui=True, show_right_ui=True
        )
    # 停止仿真可视化
    def stop_simulation(self):
        """
        停止仿真可视化
        """
        # 停止同步线程
        self.sync_running = False
        if self.sync_thread is not None:
            self.sync_thread.join(timeout=1.0)
        
        # 停止viewer
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    # 设置末端执行器位姿
    def set_end_effector_pose(self, left_pose=None, right_pose=None, interpolate=True):
        """
        设置末端执行器位姿
        
        Args:
            left_pose: 左臂末端位姿 (position, quaternion) 或 mink.SE3 对象
            right_pose: 右臂末端位姿 (position, quaternion) 或 mink.SE3 对象
            interpolate: 是否使用插值移动到目标位姿
        """
        try:
            # print(self.get_joint_angles())
            # 获取当前位姿（如果尚未初始化）
            if self.current_pose_left is None:
                self.current_pose_left = mink.SE3.from_mocap_name(self.model, self.data, MOCAP_LEFT)
            if self.current_pose_right is None:
                self.current_pose_right = mink.SE3.from_mocap_name(self.model, self.data, MOCAP_RIGHT)
            self.reset_minimal_movement_reference()
            # 处理左臂位姿
            target_left_pose = None
            if left_pose is not None:
                if isinstance(left_pose, mink.SE3):
                    target_left_pose = left_pose
                else:
                    pos, quat = left_pose
                    pos = np.array(pos, dtype=np.float64).reshape(3,)
                    quat = np.array(quat, dtype=np.float64).reshape(4,)
                    if len(quat) == 4:
                        target_left_pose = mink.SE3(np.concatenate([quat, pos]))
                    else:
                        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
                        target_left_pose = mink.SE3(np.concatenate([quat_wxyz, pos]))
            else:
                target_left_pose = self.current_pose_left

            # 处理右臂位姿
            target_right_pose = None
            if right_pose is not None:
                if isinstance(right_pose, mink.SE3):
                    target_right_pose = right_pose
                else:
                    pos, quat = right_pose
                    pos = np.array(pos, dtype=np.float64).reshape(3,)
                    quat = np.array(quat, dtype=np.float64).reshape(4,)
                    if len(quat) == 4:
                        target_right_pose = mink.SE3(np.concatenate([quat, pos]))
                    else:
                        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
                        target_right_pose = mink.SE3(np.concatenate([quat_wxyz, pos]))
            else:
                target_right_pose = self.current_pose_right

            # 执行移动（带插值或不带插值）
            if interpolate:
                self._move_with_interpolation(self.current_pose_left, self.current_pose_right, 
                                            target_left_pose, target_right_pose)
            else:
                # 直接设置目标位姿
                self.tL.set_target(target_left_pose)
                self.tR.set_target(target_right_pose)
                
                # 更新mocap位置
                self.data.mocap_pos[self.mocap_left_id] = target_left_pose.translation()
                self.data.mocap_quat[self.mocap_left_id] = target_left_pose.wxyz_xyz[0:4]
                self.data.mocap_pos[self.mocap_right_id] = target_right_pose.translation()
                self.data.mocap_quat[self.mocap_right_id] = target_right_pose.wxyz_xyz[0:4]
                
                # 执行IK求解
                self._solve_ik()
                
                # 更新当前位姿
                self.current_pose_left = target_left_pose
                self.current_pose_right = target_right_pose
                
            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()

        except Exception as e:
            print(f"Error in set_end_effector_pose: {e}")
            import traceback
            traceback.print_exc()
    def _move_with_interpolation(self, current_left_pose, current_right_pose, 
                            target_left_pose, target_right_pose):
        """
        使用插值平滑移动到目标位姿
        
        Args:
            current_left_pose: 当前左臂位姿
            current_right_pose: 当前右臂位姿
            target_left_pose: 目标左臂位姿
            target_right_pose: 目标右臂位姿
        """
        # 获取当前和目标位置
        current_left_pos = current_left_pose.translation()
        current_left_quat = current_left_pose.wxyz_xyz[0:4]
        target_left_pos = target_left_pose.translation()
        target_left_quat = target_left_pose.wxyz_xyz[0:4]
        
        current_right_pos = current_right_pose.translation()
        current_right_quat = current_right_pose.wxyz_xyz[0:4]
        target_right_pos = target_right_pose.translation()
        target_right_quat = target_right_pose.wxyz_xyz[0:4]
        
        # 计算位置差值距离
        left_pos_diff = np.linalg.norm(target_left_pos - current_left_pos)
        right_pos_diff = np.linalg.norm(target_right_pos - current_right_pos)
        
        # 根据差值密度计算插值点数
        # 假设 self.interpolation_points 是插值密度参数（单位距离的插值点数）
        max_diff = max(left_pos_diff, right_pos_diff)
        if max_diff <= 1e-6:  # 如果距离非常小，直接移动到目标位置
            # 直接设置目标位姿
            self.tL.set_target(target_left_pose)
            self.tR.set_target(target_right_pose)
            
            # 更新mocap位置
            self.data.mocap_pos[self.mocap_left_id] = target_left_pos
            self.data.mocap_quat[self.mocap_left_id] = target_left_quat
            self.data.mocap_pos[self.mocap_right_id] = target_right_pos
            self.data.mocap_quat[self.mocap_right_id] = target_right_quat
            
            # 执行IK求解
            self._solve_ik()
            
            # 更新当前位姿
            self.current_pose_left = target_left_pose
            self.current_pose_right = target_right_pose
            return
        
        # 根据插值密度计算实际插值点数
        actual_points = max(int(max_diff * self.interpolation_points), 2)  # 至少2个点
        
        # 使用四元数球面线性插值处理旋转
        from scipy.spatial.transform import Rotation as R
        from scipy.spatial.transform import Slerp
        
        # 创建Slerp插值对象
        times = [0, 1]
        rotations_left = R.from_quat([
            [current_left_quat[1], current_left_quat[2], current_left_quat[3], current_left_quat[0]],
            [target_left_quat[1], target_left_quat[2], target_left_quat[3], target_left_quat[0]]
        ])
        slerp_left = Slerp(times, rotations_left)
        
        rotations_right = R.from_quat([
            [current_right_quat[1], current_right_quat[2], current_right_quat[3], current_right_quat[0]],
            [target_right_quat[1], target_right_quat[2], target_right_quat[3], target_right_quat[0]]
        ])
        slerp_right = Slerp(times, rotations_right)
        
        # 生成插值点
        for s in np.linspace(0, np.pi, actual_points):
            t = 0.5 - 0.5 * np.cos(s)  # S型插值
            
            # 线性插值位置
            interp_left_pos = (1 - t) * current_left_pos + t * target_left_pos
            interp_right_pos = (1 - t) * current_right_pos + t * target_right_pos
            
            # 球面线性插值旋转
            interp_left_rot = slerp_left([t])[0]
            interp_right_rot = slerp_right([t])[0]
            
            # 转换四元数格式 (xyzw -> wxyz)
            interp_left_quat_xyzw = interp_left_rot.as_quat()
            interp_left_quat = np.array([interp_left_quat_xyzw[3], interp_left_quat_xyzw[0], 
                                    interp_left_quat_xyzw[1], interp_left_quat_xyzw[2]])
            
            interp_right_quat_xyzw = interp_right_rot.as_quat()
            interp_right_quat = np.array([interp_right_quat_xyzw[3], interp_right_quat_xyzw[0], 
                                        interp_right_quat_xyzw[1], interp_right_quat_xyzw[2]])
            
            # 创建中间位姿
            intermediate_left_pose = mink.SE3(np.concatenate([interp_left_quat, interp_left_pos]))
            intermediate_right_pose = mink.SE3(np.concatenate([interp_right_quat, interp_right_pos]))
            
            # 设置目标位姿
            self.tL.set_target(intermediate_left_pose)
            self.tR.set_target(intermediate_right_pose)
            
            # 更新mocap位置
            self.data.mocap_pos[self.mocap_left_id] = interp_left_pos
            self.data.mocap_quat[self.mocap_left_id] = interp_left_quat
            self.data.mocap_pos[self.mocap_right_id] = interp_right_pos
            self.data.mocap_quat[self.mocap_right_id] = interp_right_quat
            
            # 执行IK求解
            self._solve_ik()
            
                        # 更新仿真并同步显示
            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()
            
            # 控制插值速度
            time.sleep(0.01)
        
        # 更新当前位姿
        self.current_pose_left = target_left_pose
        self.current_pose_right = target_right_pose
    def set_interpolation_points(self, points):
        """
        设置插值密度（每单位距离的插值点数）
        
        Args:
            points: 插值密度，例如0.05表示每0.05单位距离进行一次插值
        """
        self.interpolation_points = points
    def _solve_ik(self):
        """使用自适应重复次数执行IK求解，越接近目标点重复次数越多"""
        try:
            dt = self.timestep
            
            # 计算当前位姿与目标位姿的距离，用于确定重复次数
            repeat = self._calculate_adaptive_repeat()
            
            # 调用_step_ik执行IK求解
            self._step_ik(dt, repeat=repeat)
        except Exception as e:
            print(f"Error in _solve_ik: {e}")
            import traceback
            traceback.print_exc()
    def _step_ik(self, dt, repeat=15):
        """执行IK求解步骤（增加错误处理和并发控制）"""
        for _ in range(repeat):
            try:
                vel = mink.solve_ik(
                    self.cfg, self.tasks, dt, solver="daqp", damping=1e-3)
                max_vel = self.max_vel
                v_norm = np.linalg.norm(vel, np.inf)
                if v_norm > max_vel:
                    vel *= max_vel / v_norm
                if vel is None:
                    vel = 0.0
                self.cfg.integrate_inplace(vel, dt)
                
                self.data.qpos[:] = self.cfg.q
                self.data.qvel[:] = vel
                mujoco.mj_forward(self.model, self.data)
                    
            except Exception as e:
                if "nefc under-allocation" in str(e):
                    print("检测到约束不足，正在重新分配...")
                    # 保存当前状态
                    current_qpos = self.data.qpos.copy()
                    current_qvel = self.data.qvel.copy()
                    
                    # 重新创建模型和数据，增加约束数量
                    self.model.njmax = int(self.model.njmax * 1.5)
                    self.model.nconmax = int(self.model.nconmax * 1.5)
                    new_data = mujoco.MjData(self.model)
                    
                    # 恢复状态
                    new_data.qpos[:] = current_qpos
                    new_data.qvel[:] = current_qvel
                    self.data = new_data
                    self.cfg.data = self.data
                    
                    # 重试
                    vel = mink.solve_ik(
                        self.cfg, self.tasks, dt, solver="daqp", damping=1e-3)
                    max_vel = self.max_vel
                    v_norm = np.linalg.norm(vel, np.inf)
                    if v_norm > max_vel:
                        vel *= max_vel / v_norm
                    if vel is None:
                        vel = 0.0
                    self.cfg.integrate_inplace(vel, dt)
                    
                    with self.sync_lock:
                        self.data.qpos[:] = self.cfg.q
                        self.data.qvel[:] = vel
                        mujoco.mj_forward(self.model, self.data)
                else:
                    raise e  # 重新抛出其他异常
    def _calculate_adaptive_repeat(self):
        """
        根据当前位姿与目标位姿的距离计算自适应重复次数
        距离越远，重复次数越少；距离越近，重复次数越多（控制越精细）
        同时根据距离调整任务权重
        """
        # 通过MuJoCo原生方法获取当前左右臂末端执行器位姿
        left_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, SITE_LEFT)
        right_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, SITE_RIGHT)
        
        # 获取当前位置
        current_left_pos = self.data.site_xpos[left_site_id].copy()
        current_right_pos = self.data.site_xpos[right_site_id].copy()
        
        # 获取目标位置（从mocap获取）
        target_left_pos = self.data.mocap_pos[self.mocap_left_id].copy()
        target_right_pos = self.data.mocap_pos[self.mocap_right_id].copy()
        
        # 计算位置误差
        left_pos_error = np.linalg.norm(current_left_pos - target_left_pos)
        right_pos_error = np.linalg.norm(current_right_pos - target_right_pos)
        
        # 简化处理，只使用位置误差来确定重复次数
        # 如果需要更精确的控制，可以添加旋转误差计算
        total_error = left_pos_error + right_pos_error
        
        # 定义参数
        min_repeat = CACU_REPEAT  # 最小重复次数（原始值）
        max_repeat = CACU_REPEAT * 2  # 最大重复次数
        threshold = 0.1  # 距离阈值，小于这个值时开始增加重复次数
        
        # 根据误差计算重复次数（反比关系）
        if total_error > threshold:
            repeat = min_repeat
        else:
            # 越接近目标点，重复次数越多
            repeat = min_repeat + (max_repeat - min_repeat) * (1 - total_error/threshold)
            repeat = int(max(min_repeat, min(repeat, max_repeat)))
        
        # 动态调整任务权重：越远离目标点，任务权重越大
        # 基础权重值
        base_left_trans_weight = self.tL._base_cost[0]  # 假设平移权重相同
        base_left_rot_weight = self.tL._base_cost[3]    # 假设旋转权重相同
        
        base_right_trans_weight = self.tR._base_cost[0]
        base_right_rot_weight = self.tR._base_cost[3]
        
        # 根据距离调整权重（距离越远权重越大）
        # 这里使用指数函数使远处权重增长更快
        distance_factor = min(total_error / threshold, 1.0)  # 归一化距离因子
        
        # 调整权重：越远离目标权重越大，有助于快速移动
        adjusted_left_trans_weight = base_left_trans_weight * (1 + 2 * distance_factor)
        adjusted_left_rot_weight = base_left_rot_weight * (1 + distance_factor)
        
        adjusted_right_trans_weight = base_right_trans_weight * (1 + 2 * distance_factor)
        adjusted_right_rot_weight = base_right_rot_weight * (1 + distance_factor)
        
        # 应用新的权重
        self.tL.cost[:3] = adjusted_left_trans_weight
        self.tL.cost[3:6] = adjusted_left_rot_weight
        self.tR.cost[:3] = adjusted_right_trans_weight
        self.tR.cost[3:6] = adjusted_right_rot_weight
        
        return repeat
    # 获取指定关节的角度
    def get_joint_angles(self, joint_names=None):
        """
        获取指定关节的角度
        
        Args:
            joint_names: 关节名称列表，如果为None则返回所有关节角度
            
        Returns:
            dict: 关节名称到角度的映射
        """
        if joint_names is None:
            joint_names = self.joints

        joint_angles = {}
        for joint_name in joint_names:
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qpos_idx = self.model.jnt_qposadr[jid]
                joint_angles[joint_name] = self.data.qpos[qpos_idx]
            except:
                joint_angles[joint_name] = None  # 关节不存在

        return joint_angles
    # 更改末端的任务权重（姿态优先或位置优先）
    def set_end_effector_weights(self, left_translation_weight=3.0, left_rotation_weight=1.0,
                            right_translation_weight=3.0, right_rotation_weight=1.0):
        """
        设置末端执行器任务的权重
        
        Args:
            left_translation_weight: 左臂平移权重
            left_rotation_weight: 左臂旋转权重
            right_translation_weight: 右臂平移权重
            right_rotation_weight: 右臂旋转权重
        """
        # 设置左臂权重
        self.tL.cost[:3] = left_translation_weight
        self.tL.cost[3:6] = left_rotation_weight
        
        # 设置右臂权重
        self.tR.cost[:3] = right_translation_weight
        self.tR.cost[3:6] = right_rotation_weight
    # 开始仿真可视化
    def start_simulation(self, framerate=None):
        """
        开始仿真可视化
        
        Args:
            framerate: 显示刷新率，默认与仿真频率一致
        """
        if self.viewer is not None:
            return  # 已经启动

        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data,
            show_left_ui=True, show_right_ui=True
        )

        freq = framerate or int(1 / self.timestep)
    # 停止仿真可视化
    def stop_simulation(self):
        """
        停止仿真可视化
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    def reset_minimal_movement_reference(self):
        """
        重置最小关节运动任务的参考角度
        在开始新动作前调用此方法
        """
        if hasattr(self, 'min_movement_task'):
            self.min_movement_task.reset_reference(self.cfg.q)
    # 初始化到参考姿态
    def initialize_to_reference_pose(self, left_pose=REF_LEFT, right_pose=REF_RIGHT):
        """
        平滑移动到参考姿态（从当前位置开始）
        """
        print("正在初始化到参考姿态...")
        
        # 获取当前关节角度作为起始位置
        current_qpos = self.data.qpos.copy()
        
        for s in np.linspace(0, np.pi, 50):
            t = 0.5 - 0.5 * np.cos(s)  # S型插值
            
            # 对于每个左侧关节
            for jn in left_pose:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                idx = self.model.jnt_qposadr[jid]
                # 从当前位置插值到参考角度
                current_angle = current_qpos[idx]
                target_angle = left_pose[jn]
                self.data.qpos[idx] = (1 - t) * current_angle + t * target_angle
                
            # 对于每个右侧关节
            for jn in right_pose:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                idx = self.model.jnt_qposadr[jid]
                # 从当前位置插值到参考角度
                current_angle = current_qpos[idx]
                target_angle = right_pose[jn]
                self.data.qpos[idx] = (1 - t) * current_angle + t * target_angle

            # 更新仿真并同步显示
            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)  # 控制速度
            
        print("初始化完成")

from world import CoordinateTransformer
def demo_xbox_control_with_visualization():
    """
    使用Xbox手柄控制机器人双臂的演示，并可视化目标位姿
    左摇杆控制左臂位置，右摇杆控制左臂姿态
    """
    try:
        # 创建控制器实例
        robot_controller = RobotController()
        
        # 初始化到参考姿态
        robot_controller.initialize_to_reference_pose()
        
        # 获取初始位姿
        initial_left_pose = mink.SE3.from_mocap_name(robot_controller.model, robot_controller.data, MOCAP_LEFT)
        initial_right_pose = mink.SE3.from_mocap_name(robot_controller.model, robot_controller.data, MOCAP_RIGHT)
        
        # 初始化Xbox控制器
        try:
            xbox_controller = XboxController(long_press_threshold=1.0)
        except Exception as e:
            print(f"无法初始化Xbox控制器: {e}")
            print("请确保已连接Xbox控制器并正确安装pygame")
            return
        
        # 初始化坐标变换器
        coordinate_transformer = CoordinateTransformer()
        
        print("Xbox手柄控制演示开始:")
        print("左摇杆: 控制左臂位置 (X/Y方向)")
        print("右摇杆: 控制左臂姿态 (绕X/Y轴旋转)")
        print("扳机键: 控制Z方向移动 (LT下降, RT上升)")
        print("按B键退出")
        print("按A键重置位置")
        print("按LB键切换左右手")
        print("按RB键执行任务")
        print("按X键切换到坐标跟踪模式")
        
        # 控制参数
        position_scale = 0.1  # 降低位置控制灵敏度，避免移动过快
        rotation_scale = 0.1   # 姿态控制灵敏度
        
        # 初始化左右臂的当前位置和姿态
        current_left_pos = initial_left_pose.translation().copy()
        current_right_pos = initial_right_pose.translation().copy()
        last_left_pos = current_left_pos
        last_right_pos = current_right_pos
        
        # 获取初始四元数 (wxyz格式)
        initial_left_quat = initial_left_pose.wxyz_xyz[0:4].copy()  # wxyz格式
        current_left_quat = initial_left_quat.copy()
        initial_right_quat = initial_right_pose.wxyz_xyz[0:4].copy()  # wxyz格式
        current_right_quat = initial_right_quat.copy()
        
        # 当前控制的手臂标志 (False=左臂, True=右臂)
        control_right_arm = False
        
        # 坐标跟踪模式标志
        position_tracking_mode = False
        
        running = True
        
        print("\r\r\r\r\r\r\r\r")
        
        try:
            while running and robot_controller.viewer is not None and robot_controller.viewer.is_running():
                # 注意：XboxController在后台线程中自动更新状态，无需手动调用update()
                
                # 检查是否按下B键退出
                if xbox_controller.namedbutton_states["B"][0]:
                    print("检测到B键按下，退出控制")
                    running = False
                    break
                # 检查是否按下A键重置位置
                if xbox_controller.namedbutton_states["A"][0]:
                    print("检测到A键按下，重置位置")
                    xbox_controller.reset_button_flags()
                    
                    current_left_pos = initial_left_pose.translation().copy()
                    current_left_quat = initial_left_quat.copy()
                    current_right_pos = initial_right_pose.translation().copy()
                    current_right_quat = initial_right_quat.copy()
                    
                    # 创建目标位姿
                    target_left_pose = mink.SE3(np.concatenate([current_left_quat, current_left_pos]))
                    target_right_pose = mink.SE3(np.concatenate([current_right_quat, current_right_pos]))
                    
                    # 重要：更新当前位姿状态，防止闪动
                    robot_controller.current_pose_left = target_left_pose
                    robot_controller.current_pose_right = target_right_pose
                    
                    # 重置到参考姿态
                    robot_controller.initialize_to_reference_pose()
                    
                    # 更新mocap位置
                    robot_controller.data.mocap_pos[robot_controller.mocap_left_id] = target_left_pose.translation()
                    robot_controller.data.mocap_quat[robot_controller.mocap_left_id] = target_left_pose.wxyz_xyz[0:4]
                    robot_controller.data.mocap_pos[robot_controller.mocap_right_id] = target_right_pose.translation()
                    robot_controller.data.mocap_quat[robot_controller.mocap_right_id] = target_right_pose.wxyz_xyz[0:4]
                    
                    # 同步显示
                    mujoco.mj_forward(robot_controller.model, robot_controller.data)
                    robot_controller.viewer.sync()
                    
                    time.sleep(0.3)  # 防止重复触发
                    continue
                
                # 检查是否按下X键切换坐标跟踪模式
                position_tracking_mode = xbox_controller.namedbutton_states["X"][1]
                # 检查是否按下LB键切换控制手臂
                control_right_arm = xbox_controller.namedbutton_states["LB"][1]
                
                # # 检查是否按下RB键切换 追踪/摊平
                # if xbox_controller.namedbutton_states["RB"][1]:
                #     robot_controller.set_end_effector_weights(10,1,10,1)    # 位置优先，指尖追踪
                #     print("切换追踪/摊平")
                # else:
                #     robot_controller.set_end_effector_weights(10,1,10,1)    # 姿态优先，手摊平
                #     print("切换追踪/摊平")
                
                # 如果处于坐标跟踪模式
                if position_tracking_mode:
                    # 当 all_correct 为 True 时，使用获取到的新位置，姿态保持不变
                    tracked_position = coordinate_transformer.get_current_position()
                    # 当 all_correct 为 False 时，位置和姿态都保持不变
                    if coordinate_transformer.all_correct:
                        # 将跟踪的位置应用到当前控制的手臂
                        if control_right_arm:
                            # 更新右臂位置，保持姿态不变
                            current_right_pos[:] = [i / 1000. for i in tracked_position]
                        else:
                            # 更新左臂位置，保持姿态不变
                            current_left_pos[:] = [i / 1000. for i in tracked_position]
                    else:
                        current_left_pos[:] = last_left_pos
                        current_right_pos[:] = last_right_pos
                    
                    # 如果 all_correct 为 False，则位置和姿态都保持不变（不需要特殊处理）
                else:
                    # 手动控制模式
                    # 根据当前控制的手臂选择参数
                    if control_right_arm:
                        # 控制右臂
                        current_pos = current_right_pos
                        current_quat = current_right_quat
                    else:
                        # 控制左臂
                        current_pos = current_left_pos
                        current_quat = current_left_quat
                    
                    # 获取摇杆输入
                    left_stick = xbox_controller.left_stick    # 用于位置控制 (x, y)
                    right_stick = xbox_controller.right_stick  # 用于姿态控制 (x, y)
                    # 使用扳机键控制Z轴移动
                    up_stick = xbox_controller.triggers[1] - xbox_controller.triggers[0]  # RT - LT
                    
                    # 添加死区处理，避免微小摇杆移动
                    left_stick = [0.0 if abs(x) < 0.1 else x for x in left_stick]
                    right_stick = [0.0 if abs(x) < 0.1 else x for x in right_stick]
                    up_stick = 0.0 if abs(up_stick) < 0.1 else up_stick
                    
                    # 位置变化量
                    delta_pos = np.array([
                        left_stick[1] * position_scale,   # X方向
                        -left_stick[0] * position_scale,  # Y方向 (注意符号，可能需要调整)
                        up_stick * position_scale         # Z方向 (RT上升, LT下降)
                    ])
                    
                    # 应用位置变化
                    if np.any(np.abs(delta_pos) > 0):
                        current_pos += delta_pos
                    
                    # 姿态控制 - 使用更简单的方法
                    if np.any(np.abs(right_stick) > 0):
                        # 将当前四元数转换为旋转矩阵，应用增量旋转，再转回四元数
                        from scipy.spatial.transform import Rotation as R
                        
                        # 当前旋转 (wxyz -> xyzw)
                        current_rot = R.from_quat([current_quat[1], current_quat[2], 
                                                current_quat[3], current_quat[0]])  # wxyz -> xyzw
                        
                        # 增量旋转 (绕Y轴和X轴)
                        delta_rot_y = right_stick[1] * rotation_scale  # 右摇杆左右 -> 绕Y轴
                        delta_rot_x = right_stick[0] * rotation_scale  # 右摇杆上下 -> 绕X轴
                        
                        # 创建增量旋转 (使用与u0.py相同的顺序)
                        incremental_rot = R.from_euler('yx', [delta_rot_y, delta_rot_x])
                        
                        # 应用旋转 (顺序也需与u0.py保持一致)
                        new_rot = incremental_rot * current_rot
                        
                        # 转回四元数 (xyzw -> wxyz)
                        new_quat_xyzw = new_rot.as_quat()  # 返回 xyzw
                        current_quat[:] = np.array([new_quat_xyzw[3], new_quat_xyzw[0], 
                                                  new_quat_xyzw[1], new_quat_xyzw[2]])  # xyzw -> wxyz
                    
                    # 更新对应手臂的位置和姿态
                    if control_right_arm:
                        current_right_pos[:] = current_pos
                        current_right_quat[:] = current_quat
                    else:
                        current_left_pos[:] = current_pos
                        current_left_quat[:] = current_quat
                
                # 创建目标位姿
                target_left_pose = mink.SE3(np.concatenate([current_left_quat, current_left_pos]))
                target_right_pose = mink.SE3(np.concatenate([current_right_quat, current_right_pos]))
                last_left_pos = current_left_pos
                last_right_pos = current_right_pos
                
                print(target_left_pose,target_right_pose)
                
                # 设置双臂目标位姿
                robot_controller.set_end_effector_pose(
                    left_pose=target_left_pose,
                    right_pose=target_right_pose,
                    interpolate=False
                )
                
                # 控制循环频率
                time.sleep(0.02)  # 50Hz控制频率
                
        except KeyboardInterrupt:
            print("检测到Ctrl+C，退出控制")
        except Exception as e:
            print(f"控制循环中出现错误: {e}")
            import traceback
            traceback.print_exc()
            
        # 停止仿真
        robot_controller.stop_simulation()
        try:
            xbox_controller.stop()
        except:
            pass
        print("Xbox控制演示结束")
        
    except Exception as e:
        print(f"Error in demo_xbox_control: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行带可视化的控制演示
    demo_xbox_control_with_visualization()