import mujoco
from mujoco import viewer

# 加载模型
model = mujoco.MjModel.from_xml_path('urdf/1029scene.xml')
data = mujoco.MjData(model)

# 创建查看器
viewer.launch(model, data)