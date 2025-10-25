import serial.tools.list_ports

# 列出所有可用的串口
ports = serial.tools.list_ports.comports()
for port in ports:
    print(port.device)  # 输出如 /dev/ttyUSB0

# # 检查某个串口是否被占用
# try:
#     ser = serial.Serial(port.device)  # 尝试打开
#     print(f"{ser.name} 已成功打开！")
#     ser.close()
# except serial.SerialException as e:
#     print(f"无法打开串口: {e}")