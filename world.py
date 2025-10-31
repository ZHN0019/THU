#file:e:\- 项目 -\清华人形Rob\NOKOV动作捕捉系统\红外场景测试\world.py
import socket
import json
import struct
import threading
import time
import numpy as np
import math

class CoordinateTransformer:
    def __init__(self, server_ip="192.168.1.114", server_port=8080):
        """
        初始化坐标变换器
        
        Args:
            server_ip (str): 服务器IP地址，默认为"192.168.1.114"
            server_port (int): 服务器端口，默认为8080
        """
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = None
        self.connected = False
        self.running = False
        self.data_thread = None
        self.all_correct = True
        
        # 参考坐标系A在世界坐标系中的原点位置
        self.reference_origin_in_world = [-160, 0, 1485]
        
        # 存储最新的数据和转换结果
        self.latest_data = None
        self.last_valid_position = [0,0,0]
        self.current_world_position = None
        
        self.Handkerchief_Poss = [np.array([0,0,0]) for _ in range(5)]
        self.Handkerchief_Quats = [np.array([0,0,0,0]) for _ in range(5)]
        self.Orighin_Poss = [[0,0,0] for _ in range(5)]
        self.Origin_Quats = [[0,0,0,0] for _ in range(5)]
        
        # 添加锁保护共享数据
        self.data_lock = threading.Lock()
        
        # 启动后台数据接收和处理线程
        self.start_background_processing()
        
    def start_background_processing(self):
        """
        启动后台处理线程
        """
        self.running = True
        self.process_thread = threading.Thread(target=self._connection_manager, daemon=True)
        self.process_thread.start()
        
    def _connection_manager(self):
        """
        管理连接状态的线程
        """
        while self.running:
            if not self.connected:
                self._attempt_connection()
                
            # 短暂休眠避免过度占用CPU
            time.sleep(0.1)
    
    def _attempt_connection(self):
        """
        尝试连接到数据服务器
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.server_ip, self.server_port))
            self.connected = True
            print(f"成功连接到 {self.server_ip}:{self.server_port}")
            self.all_correct = True
            # 启动数据接收线程
            self.data_thread = threading.Thread(target=self._data_receive_loop, daemon=True)
            self.data_thread.start()
            
        except Exception as e:
            self.all_correct = False
            # 连接失败，稍后重试
            time.sleep(1)
    
    def _data_receive_loop(self):
        """
        专门的数据接收线程，模仿data_receiver.py的设计
        """
        while self.running and self.connected:
            try:
                # 接收数据
                data = self._receive_data()
                if data:
                    # 更新最新数据
                    with self.data_lock:
                        self.latest_data = data
                    # 实时处理坐标转换
                    self.all_correct = True
                    self._transform_coordinates()
                else:
                    self.connected = False
                    break
                    
            except Exception as e:
                if self.running:
                    self.all_correct = False
                    print(f"数据接收错误: {e}")
                self.connected = False
                break
                
        # 清理连接
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
    
    def _receive_data(self):
        """
        从套接字接收并解析数据
        
        Returns:
            dict: 解析后的数据字典，如果出错则返回None
        """
        if not self.connected or not self.socket:
            self.all_correct = False
            return None
            
        try:
            # 接收4字节的数据长度
            length_data = self._recv_all(4)
            if not length_data:
                self.all_correct = False
                return None
                
            # 解析数据长度
            data_length = struct.unpack('!I', length_data)[0]
            
            # 接收指定长度的数据
            json_data = self._recv_all(data_length)
            if not json_data:
                self.all_correct = False
                return None
                
            # 解析JSON数据
            data = json.loads(json_data.decode('utf-8'))
            self.all_correct = True
            return data
            
        except Exception as e:
            self.all_correct = False
            return None
    
    def _recv_all(self, length):
        """
        确保接收到指定长度的数据
        
        Args:
            length (int): 需要接收的数据长度
            
        Returns:
            bytes: 接收到的数据，如果出错则返回None
        """
        data = b''
        while len(data) < length:
            try:
                chunk = self.socket.recv(length - len(data))
                if not chunk:
                    self.all_correct = False
                    return None
                data += chunk
            except:
                self.all_correct = False
                return None
        self.all_correct = True
        return data

    def _transform_coordinates(self):
        """
        将未识别标记点1的位置转换为世界坐标系坐标
        """
        with self.data_lock:
            if not self.latest_data:
                self.all_correct = False
                return
            
            dia_ref_A_list = self.latest_data.get("rigid_bodies", [])
            
            if len(dia_ref_A_list) < 2:
                # 未找到参考坐标系与手绢，保持上一次的位置
                self.all_correct = False
                return
            
            dia_ref_A_markers = dia_ref_A_list[0].get("markers", [])   # 参考坐标系
            target_marker = dia_ref_A_list[1]   # 参手绢坐标系
            target_marker_loc = target_marker.get("position", [])
            target_marker_rot = target_marker.get("rotation", [])
            
            if len(dia_ref_A_markers) < 2:
                self.all_correct = False
                return
                
            dia_ref_A_loc = dia_ref_A_markers[1].get("position", [])    # 参考坐标系原点是第二个点
            if dia_ref_A_loc[0] == 9999999.0 or target_marker_loc[0] == 9999999.0:
                self.all_correct = False
                return
            dia_ref_A_rot = dia_ref_A_markers[1].get("rotation", [])
                
            # 未识别标记点1在参考坐标系A中的位置
            target_marker_locs = [target_marker_loc] + [loc["position"] for loc in target_marker.get("markers", [])]
            target_marker_rots = [target_marker_rot] + [target_marker_rot for loc in target_marker.get("markers", [])]
            if len(target_marker_locs[0]) < 3 or len(dia_ref_A_loc) < 3:
                # 坐标不完整，保持上一次的位置
                self.all_correct = False
                return
            
            # 创建坐标系A的原点（在世界坐标系中）
            origin_A_in_world = np.array(self.reference_origin_in_world)
            world_positions = [np.array([0,0,0]) for _ in range(6)]
            world_rots = [np.array([0,0,0,0]) for _ in range(6)]
            for i in range(len(target_marker_locs)):
                target_marker_loc = target_marker_locs[i]
                # 创建标记点在坐标系B中的位置
                marker_in_B = np.array(target_marker_loc)
                # 创建坐标系A的原点在坐标系B中的位置
                origin_A_in_B = np.array(dia_ref_A_loc)
                # 计算标记点相对于坐标系A原点在坐标系B中的向量
                vector_B = marker_in_B - origin_A_in_B
                # 坐标系B相对于世界坐标系绕Z轴顺时针旋转90度的变换矩阵
                # 或者说逆时针旋转270度(-90度)
                # 这样坐标系B的X轴指向世界坐标系的Y轴正方向
                # 坐标系B的Y轴指向世界坐标系的X轴负方向
                cos_neg90 = 0
                sin_neg90 = -1
                R_B_to_W = np.array([
                    [cos_neg90, -sin_neg90, 0],  # [0, 1, 0]
                    [sin_neg90, cos_neg90, 0],   # [-1, 0, 0]
                    [0, 0, 1]                    # [0, 0, 1]
                ])
                
                # 将向量从坐标系B变换到世界坐标系
                vector_W = R_B_to_W @ vector_B
                
                # 计算标记点在世界坐标系中的最终位置
                # 世界坐标 = 坐标系A原点在世界坐标系中的位置 + 向量在世界坐标系中的表示
                if target_marker_loc[0] != 9999999.0:
                    world_positions[i] = origin_A_in_world + vector_W
                    world_rots[i] = np.array(target_marker_rots[i])
            
            # 更新当前位置并保存为上一次的有效位置
            self.Handkerchief_Poss = world_positions[1:]
            self.Handkerchief_Quats = world_rots[1:]
            self.current_world_position = world_positions[0].tolist()
            self.last_valid_position = self.current_world_position
            self.all_correct = True

    def _transform_coordinates_with_pos(self):
        """
        将未识别标记点1的位置和姿态转换为世界坐标系坐标
        """
        with self.data_lock:
            if not self.latest_data:
                self.all_correct = False
                return
            
            # 提取未识别标记点1的位置和姿态（相对于参考坐标系A）
            unidentified_markers = self.latest_data.get("unidentified_markers", [])
            dia_ref_A_list = self.latest_data.get("rigid_bodies", [])
            
            if len(unidentified_markers) < 1 or len(dia_ref_A_list) < 2:
                # 未找到未识别标记点1，保持上一次的位置
                self.all_correct = False
                return
            
            dia_ref_A = dia_ref_A_list[0]
            dia_ref_A_markers = dia_ref_A.get("markers", [])
            
            if len(dia_ref_A_markers) < 2:
                self.all_correct = False
                return
                
            dia_ref_A_loc = dia_ref_A_markers[1].get("position", [])
            dia_ref_A_rot = dia_ref_A_markers[1].get("rotation", [])
                
            # 未识别标记点1在参考坐标系A中的位置和姿态
            marker_local_pos = unidentified_markers[0]
            # 假设还有姿态数据，这里简化处理
            marker_local_rot = [0, 0, 0]  # 如果有姿态数据应从数据源获取
            
            if len(marker_local_pos) < 3 or len(dia_ref_A_loc) < 3:
                # 坐标不完整，保持上一次的位置
                self.all_correct = False
                return
            
            # 创建坐标系A的原点（在世界坐标系中）
            origin_A_in_world = np.array(self.reference_origin_in_world)
            
            # 创建标记点在坐标系B中的位置
            marker_in_B = np.array(marker_local_pos)
            
            # 创建坐标系A的原点在坐标系B中的位置
            origin_A_in_B = np.array(dia_ref_A_loc)
            
            # 计算标记点相对于坐标系A原点在坐标系B中的向量
            vector_B = marker_in_B - origin_A_in_B
            
            # 坐标系B相对于世界坐标系绕Z轴顺时针旋转90度的变换矩阵
            # 或者说逆时针旋转270度(-90度)
            cos_neg90 = 0
            sin_neg90 = -1
            R_B_to_W = np.array([
                [cos_neg90, -sin_neg90, 0],  # [0, 1, 0]
                [sin_neg90, cos_neg90, 0],   # [-1, 0, 0]
                [0, 0, 1]                    # [0, 0, 1]
            ])
            
            # 将向量从坐标系B变换到世界坐标系
            vector_W = R_B_to_W @ vector_B
            
            # 计算标记点在世界坐标系中的最终位置
            # 世界坐标 = 坐标系A原点在世界坐标系中的位置 + 向量在世界坐标系中的表示
            world_position = origin_A_in_world + vector_W
            
            # 处理姿态变换
            # 将局部姿态转换为世界坐标系中的姿态
            # 这里假设姿态是以欧拉角表示的
            if len(marker_local_rot) >= 3:
                # 局部姿态
                local_rotation = np.array(marker_local_rot)
                
                # 由于只有绕Z轴的旋转，我们可以简单相加
                # 注意：实际应用中可能需要更复杂的四元数或旋转矩阵运算
                world_rotation = local_rotation + [0, 0, -np.pi/2]  # 减去90度
                
                # 保存姿态信息
                self.current_world_rotation = world_rotation.tolist()
            else:
                self.current_world_rotation = None
            
            # 更新当前位置并保存为上一次的有效位置
            self.current_world_position = world_position.tolist()
            self.last_valid_position = self.current_world_position
            self.all_correct = True

    def get_current_pose(self):
        """
        获取当前未识别标记点1在世界坐标系中的位置和姿态
        
        Returns:
            tuple: (position, rotation) 分别包含位置[x, y, z]和姿态[rx, ry, rz]，
                如果从未获取到有效数据则返回(None, None)
        """
        position = self.current_world_position if self.current_world_position else self.last_valid_position
        rotation = getattr(self, 'current_world_rotation', None)
        return (position, rotation)

    def get_current_position(self):
        """
        获取当前未识别标记点1在世界坐标系中的位置
        
        Returns:
            list: 包含[x, y, z]的世界坐标，如果从未获取到有效数据则返回None
        """
        return self.current_world_position if self.current_world_position else self.last_valid_position

# 使用示例
def main():
    # 创建坐标变换器实例，它会在后台自动运行
    transformer = CoordinateTransformer()
    
    # 持续读取坐标数据
    count = 0
    while 1:  # 读取100次示例
        position = transformer.get_current_position()
        if position:
            print(f"世界坐标系位置: X={position[0]:.3f}, Y={position[1]:.3f}, Z={position[2]:.3f}, {transformer.all_correct}")
        else:
            print("尚未获取到有效位置数据")
        time.sleep(0.01)  # 每100ms读取一次
        count += 1

if __name__ == "__main__":
    main()