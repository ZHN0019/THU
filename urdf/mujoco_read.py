import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("scene.xml")
data  = mujoco.MjData(model)

# 彻底关闭所有动力学项
model.opt.disableflags |= (
    mujoco.mjtDisableBit.mjDSBL_GRAVITY   |
    mujoco.mjtDisableBit.mjDSBL_COLLISION |
    mujoco.mjtDisableBit.mjDSBL_PASSIVE   |
    mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)

# 给关节填一个姿态 → 只正向运动学
data.qpos[:] = np.zeros(model.nq)        # 按自由度数调整
mujoco.mju_forward(model, data)          # 无积分、零抖动

print("末端位置：", data.site("ee_site").xpos)
