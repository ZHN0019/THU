# enhanced_data_receiver.py
import socket
import json
import struct
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext

class EnhancedNokovGUIReceiver:
    def __init__(self, root):
        self.root = root
        self.root.title("Nokov动作捕捉数据完整接收器")
        self.root.geometry("1200x800")
        
        # 连接参数
        self.server_ip = tk.StringVar(value="192.168.1.114")
        self.server_port = tk.IntVar(value=8080)
        self.connected = False
        self.socket = None
        self.data_thread = None
        self.running = False
        
        # 数据统计
        self.frame_count = 0
        self.last_frame_time = 0
        self.fps = 0
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        # 顶部连接控制区域
        control_frame = ttk.LabelFrame(self.root, text="连接控制", padding="10")
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(control_frame, text="服务器IP:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.server_ip, width=15).grid(row=0, column=1, padx=5)
        
        ttk.Label(control_frame, text="端口:").grid(row=0, column=2, sticky=tk.W, padx=(10,0))
        ttk.Entry(control_frame, textvariable=self.server_port, width=8).grid(row=0, column=3, padx=5)
        
        self.connect_btn = ttk.Button(control_frame, text="连接", command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=4, padx=10)
        
        self.status_label = ttk.Label(control_frame, text="未连接")
        self.status_label.grid(row=0, column=5, padx=10)
        
        # 统计信息区域
        stats_frame = ttk.LabelFrame(self.root, text="实时统计", padding="10")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=4, state=tk.DISABLED)
        self.stats_text.pack(fill=tk.X)
        
        # 数据显示区域
        data_notebook = ttk.Notebook(self.root)
        data_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 帧数据标签页
        self.frame_frame = ttk.Frame(data_notebook)
        data_notebook.add(self.frame_frame, text="帧数据")
        self.frame_text = scrolledtext.ScrolledText(self.frame_frame, state=tk.DISABLED)
        self.frame_text.pack(fill=tk.BOTH, expand=True)
        
        # 标记集数据标签页
        self.markerset_frame = ttk.Frame(data_notebook)
        data_notebook.add(self.markerset_frame, text="标记集数据")
        self.markerset_text = scrolledtext.ScrolledText(self.markerset_frame, state=tk.DISABLED)
        self.markerset_text.pack(fill=tk.BOTH, expand=True)
        
        # 刚体数据标签页
        self.rigid_body_frame = ttk.Frame(data_notebook)
        data_notebook.add(self.rigid_body_frame, text="刚体数据")
        self.rigid_body_text = scrolledtext.ScrolledText(self.rigid_body_frame, state=tk.DISABLED)
        self.rigid_body_text.pack(fill=tk.BOTH, expand=True)
        
        # 骨骼数据标签页
        self.skeleton_frame = ttk.Frame(data_notebook)
        data_notebook.add(self.skeleton_frame, text="骨骼数据")
        self.skeleton_text = scrolledtext.ScrolledText(self.skeleton_frame, state=tk.DISABLED)
        self.skeleton_text.pack(fill=tk.BOTH, expand=True)
        
        # 未识别标记点标签页
        self.unidentified_frame = ttk.Frame(data_notebook)
        data_notebook.add(self.unidentified_frame, text="未识别标记点")
        self.unidentified_text = scrolledtext.ScrolledText(self.unidentified_frame, state=tk.DISABLED)
        self.unidentified_text.pack(fill=tk.BOTH, expand=True)
        
        # 模拟数据标签页
        self.analog_frame = ttk.Frame(data_notebook)
        data_notebook.add(self.analog_frame, text="模拟数据")
        self.analog_text = scrolledtext.ScrolledText(self.analog_frame, state=tk.DISABLED)
        self.analog_text.pack(fill=tk.BOTH, expand=True)
        
        # 力板数据标签页
        self.force_plate_frame = ttk.Frame(data_notebook)
        data_notebook.add(self.force_plate_frame, text="力板数据")
        self.force_plate_text = scrolledtext.ScrolledText(self.force_plate_frame, state=tk.DISABLED)
        self.force_plate_text.pack(fill=tk.BOTH, expand=True)
        
        # 通知消息标签页
        self.notification_frame = ttk.Frame(data_notebook)
        data_notebook.add(self.notification_frame, text="通知消息")
        self.notification_text = scrolledtext.ScrolledText(self.notification_frame, state=tk.DISABLED)
        self.notification_text.pack(fill=tk.BOTH, expand=True)
        
        # 日志区域
        log_frame = ttk.LabelFrame(self.root, text="系统日志", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def toggle_connection(self):
        if self.connected:
            self.disconnect()
        else:
            self.connect()
            
    def connect(self):
        if self.connected:
            return
            
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            
            ip = self.server_ip.get()
            port = self.server_port.get()
            
            self.log_message(f"正在连接到 {ip}:{port}...")
            self.socket.connect((ip, port))
            self.connected = True
            self.running = True
            
            # 更新界面
            self.connect_btn.config(text="断开")
            self.status_label.config(text=f"已连接到 {ip}:{port}")
            
            # 启动数据接收线程
            self.data_thread = threading.Thread(target=self.data_receive_loop, daemon=True)
            self.data_thread.start()
            
            self.log_message("连接成功!")
            
        except Exception as e:
            self.log_message(f"连接失败: {e}")
            self.disconnect()
            
    def disconnect(self):
        self.running = False
        self.connected = False
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
            
        # 更新界面
        self.connect_btn.config(text="连接")
        self.status_label.config(text="未连接")
        self.log_message("连接已断开")
        
    def data_receive_loop(self):
        while self.running and self.connected:
            try:
                # 接收数据
                data = self.receive_data()
                if data:
                    # 在主线程中更新界面
                    self.root.after(0, self.handle_data, data)
                else:
                    self.root.after(0, self.disconnect)
                    break
                    
            except Exception as e:
                if self.running:
                    self.root.after(0, lambda: self.log_message(f"数据接收错误: {e}"))
                break
                
    def receive_data(self):
        if not self.connected or not self.socket:
            return None
            
        try:
            # 接收4字节的数据长度
            length_data = self._recv_all(4)
            if not length_data:
                return None
                
            # 解析数据长度
            data_length = struct.unpack('!I', length_data)[0]
            
            # 接收指定长度的数据
            json_data = self._recv_all(data_length)
            if not json_data:
                return None
                
            # 解析JSON数据
            data = json.loads(json_data.decode('utf-8'))
            return data
            
        except Exception as e:
            if self.running:
                self.log_message(f"接收数据时出错: {e}")
            self.connected = False
            return None
    
    def _recv_all(self, length):
        """确保接收到指定长度的数据"""
        data = b''
        while len(data) < length:
            chunk = self.socket.recv(length - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def handle_data(self, data):
        # 检查数据类型
        data_type = data.get("type", "frame")
        
        if data_type == "welcome":
            self.log_message(f"服务器消息: {data.get('message')}")
            return
        elif data_type == "notification":
            self.update_notification_display(data)
            return
        elif data_type == "force_plate":
            self.update_force_plate_display(data)
            return
        elif data_type == "analog_channel":
            self.update_analog_display(data)
            return
        
        # 更新统计数据
        self.frame_count += 1
        current_time = time.time()
        if self.last_frame_time > 0:
            self.fps = 1.0 / (current_time - self.last_frame_time)
        else:
            self.fps = 0
        self.last_frame_time = current_time
        
        # 更新各类数据显示
        self.update_stats_display(data)
        self.update_frame_display(data)
        self.update_markerset_display(data)
        self.update_rigid_body_display(data)
        self.update_skeleton_display(data)
        self.update_unidentified_display(data)
        
    def update_stats_display(self, data):
        frame_no = data.get("frame_no", "N/A")
        timestamp = data.get("timestamp", "N/A")
        timecode = data.get("timecode", "N/A")
        
        stats = f"帧号: {frame_no} | 时间戳: {timestamp} | 时间码: {timecode} | FPS: {self.fps:.1f} | 总帧数: {self.frame_count}\n"
        stats += f"标记集: {len(data.get('markersets', []))} | "
        stats += f"刚体: {len(data.get('rigid_bodies', []))} | "
        stats += f"骨骼: {len(data.get('skeletons', []))} | "
        stats += f"未识别标记点: {len(data.get('unidentified_markers', []))} | "
        stats += f"模拟数据: {len(data.get('analog_data', []))}"
        
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats)
        self.stats_text.config(state=tk.DISABLED)
        
    def update_frame_display(self, data):
        display_text = f"帧号: {data.get('frame_no', 'N/A')}\n"
        display_text += f"时间戳: {data.get('timestamp', 'N/A')}\n"
        display_text += f"时间码: {data.get('timecode', 'N/A')}\n"
        
        self.frame_text.config(state=tk.NORMAL)
        self.frame_text.delete(1.0, tk.END)
        self.frame_text.insert(tk.END, display_text)
        self.frame_text.config(state=tk.DISABLED)
        
    def update_markerset_display(self, data):
        markersets = data.get("markersets", [])
        
        display_text = f"标记集总数: {len(markersets)}\n\n"
        for i, ms in enumerate(markersets):
            name = ms.get('name', 'N/A')
            markers = ms.get('markers', [])
            display_text += f"标记集 {i+1} ({name}): {len(markers)} 个标记点\n"
            
            for j, marker in enumerate(markers):
                if len(marker) >= 3:
                    display_text += f"  标记点 {j+1}: x={marker[0]:.3f}, y={marker[1]:.3f}, z={marker[2]:.3f}\n"
            display_text += "\n"
        
        self.markerset_text.config(state=tk.NORMAL)
        self.markerset_text.delete(1.0, tk.END)
        self.markerset_text.insert(tk.END, display_text)
        self.markerset_text.config(state=tk.DISABLED)
        
    def update_rigid_body_display(self, data):
        rigid_bodies = data.get("rigid_bodies", [])
        
        display_text = f"刚体总数: {len(rigid_bodies)}\n\n"
        for i, body in enumerate(rigid_bodies):
            pos = body.get("position", [])
            rot = body.get("rotation", [])
            display_text += f"刚体 {i+1} (ID: {body.get('id', 'N/A')}):\n"
            display_text += f"  位置: x={pos[0] if len(pos) > 0 else 'N/A':.3f}, "
            display_text += f"y={pos[1] if len(pos) > 1 else 'N/A':.3f}, "
            display_text += f"z={pos[2] if len(pos) > 2 else 'N/A':.3f}\n"
            display_text += f"  旋转: qx={rot[0] if len(rot) > 0 else 'N/A':.6f}, "
            display_text += f"qy={rot[1] if len(rot) > 1 else 'N/A':.6f}, "
            display_text += f"qz={rot[2] if len(rot) > 2 else 'N/A':.6f}, "
            display_text += f"qw={rot[3] if len(rot) > 3 else 'N/A':.6f}\n"
            
            markers = body.get("markers", [])
            if markers:
                display_text += f"  标记点数量: {len(markers)}\n"
                for j, marker in enumerate(markers):
                    marker_pos = marker.get("position", [])
                    if len(marker_pos) >= 3:
                        display_text += f"    标记点 {marker.get('id', 'N/A')}: "
                        display_text += f"x={marker_pos[0]:.3f}, y={marker_pos[1]:.3f}, z={marker_pos[2]:.3f}\n"
            display_text += "\n"
        
        self.rigid_body_text.config(state=tk.NORMAL)
        self.rigid_body_text.delete(1.0, tk.END)
        self.rigid_body_text.insert(tk.END, display_text)
        self.rigid_body_text.config(state=tk.DISABLED)
        
    def update_skeleton_display(self, data):
        skeletons = data.get("skeletons", [])
        
        display_text = f"骨骼总数: {len(skeletons)}\n\n"
        for i, sk in enumerate(skeletons):
            segments = sk.get("segments", [])
            display_text += f"骨骼 {i+1}: {len(segments)} 个片段\n"
            
            for j, segment in enumerate(segments):
                seg_id = segment.get('id', 'N/A')
                pos = segment.get("position", [])
                rot = segment.get("rotation", [])
                display_text += f"  片段 {j+1} (ID: {seg_id}):\n"
                display_text += f"    位置: x={pos[0] if len(pos) > 0 else 'N/A':.3f}, "
                display_text += f"y={pos[1] if len(pos) > 1 else 'N/A':.3f}, "
                display_text += f"z={pos[2] if len(pos) > 2 else 'N/A':.3f}\n"
                display_text += f"    旋转: qx={rot[0] if len(rot) > 0 else 'N/A':.6f}, "
                display_text += f"qy={rot[1] if len(rot) > 1 else 'N/A':.6f}, "
                display_text += f"qz={rot[2] if len(rot) > 2 else 'N/A':.6f}, "
                display_text += f"qw={rot[3] if len(rot) > 3 else 'N/A':.6f}\n"
                
                markers = segment.get("markers", [])
                if markers:
                    display_text += f"    标记点数量: {len(markers)}\n"
                    for k, marker in enumerate(markers):
                        marker_pos = marker.get("position", [])
                        if len(marker_pos) >= 3:
                            display_text += f"      标记点 {marker.get('id', 'N/A')}: "
                            display_text += f"x={marker_pos[0]:.3f}, y={marker_pos[1]:.3f}, z={marker_pos[2]:.3f}\n"
            display_text += "\n"
        
        self.skeleton_text.config(state=tk.NORMAL)
        self.skeleton_text.delete(1.0, tk.END)
        self.skeleton_text.insert(tk.END, display_text)
        self.skeleton_text.config(state=tk.DISABLED)
        
    def update_unidentified_display(self, data):
        unidentified = data.get("unidentified_markers", [])
        
        display_text = f"未识别标记点总数: {len(unidentified)}\n\n"
        for i, marker in enumerate(unidentified):
            if len(marker) >= 3:
                display_text += f"标记点 {i+1}: x={marker[0]:.3f}, y={marker[1]:.3f}, z={marker[2]:.3f}\n"
        
        self.unidentified_text.config(state=tk.NORMAL)
        self.unidentified_text.delete(1.0, tk.END)
        self.unidentified_text.insert(tk.END, display_text)
        self.unidentified_text.config(state=tk.DISABLED)
        
    def update_analog_display(self, data):
        if data.get("type") == "analog_channel":
            display_text = f"模拟通道数据:\n"
            display_text += f"  帧号: {data.get('frame', 'N/A')}\n"
            display_text += f"  时间戳: {data.get('timestamp', 'N/A')}\n"
            display_text += f"  通道数: {data.get('channel_count', 'N/A')}\n"
            display_text += f"  子帧数: {data.get('sub_frame', 'N/A')}\n\n"
            
            channel_data = data.get("data", [])
            for i, channel in enumerate(channel_data):
                display_text += f"  通道 {i}: "
                for j, value in enumerate(channel):
                    display_text += f"{value:.4f}"
                    if j < len(channel) - 1:
                        display_text += ", "
                display_text += "\n"
        else:
            analog_data = data.get("analog_data", [])
            display_text = f"模拟数据点数: {len(analog_data)}\n\n"
            for i, value in enumerate(analog_data):
                display_text += f"数据点 {i+1}: {value:.6f}\n"
        
        self.analog_text.config(state=tk.NORMAL)
        self.analog_text.delete(1.0, tk.END)
        self.analog_text.insert(tk.END, display_text)
        self.analog_text.config(state=tk.DISABLED)
        
    def update_force_plate_display(self, data):
        display_text = f"力板数据:\n"
        display_text += f"  帧号: {data.get('frame', 'N/A')}\n\n"
        
        force_plates = data.get("force_plates", [])
        for i, plate in enumerate(force_plates):
            Fxyz = plate.get("Fxyz", [])
            xyz = plate.get("xyz", [])
            Mfree = plate.get("Mfree", 0)
            display_text += f"  力板 {i+1}:\n"
            display_text += f"    力(Fxyz): Fx={Fxyz[0] if len(Fxyz) > 0 else 'N/A':.3f}, "
            display_text += f"Fy={Fxyz[1] if len(Fxyz) > 1 else 'N/A':.3f}, "
            display_text += f"Fz={Fxyz[2] if len(Fxyz) > 2 else 'N/A':.3f}\n"
            display_text += f"    位置(xyz): x={xyz[0] if len(xyz) > 0 else 'N/A':.3f}, "
            display_text += f"y={xyz[1] if len(xyz) > 1 else 'N/A':.3f}, "
            display_text += f"z={xyz[2] if len(xyz) > 2 else 'N/A':.3f}\n"
            display_text += f"    力矩(Mfree): {Mfree:.3f}\n\n"
        
        self.force_plate_text.config(state=tk.NORMAL)
        self.force_plate_text.delete(1.0, tk.END)
        self.force_plate_text.insert(tk.END, display_text)
        self.force_plate_text.config(state=tk.DISABLED)
        
    def update_notification_display(self, data):
        timestamp = time.strftime("%H:%M:%S")
        display_text = f"[{timestamp}] 通知消息:\n"
        display_text += f"  类型: {data.get('notify_type', 'N/A')}\n"
        display_text += f"  值: {data.get('value', 'N/A')}\n"
        display_text += f"  时间戳: {data.get('timestamp', 'N/A')}\n"
        display_text += f"  消息: {data.get('message', 'N/A')}\n"
        display_text += f"  参数1: {data.get('param1', 'N/A')}\n"
        display_text += f"  参数2: {data.get('param2', 'N/A')}\n"
        display_text += f"  参数3: {data.get('param3', 'N/A')}\n"
        display_text += f"  参数4: {data.get('param4', 'N/A')}\n\n"
        
        self.notification_text.config(state=tk.NORMAL)
        self.notification_text.insert(tk.END, display_text)
        self.notification_text.see(tk.END)
        self.notification_text.config(state=tk.DISABLED)
        
    def log_message(self, message):
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, formatted_message)
        self.log_text.see(tk.END)  # 自动滚动到最新消息
        self.log_text.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = EnhancedNokovGUIReceiver(root)
    
    # 处理窗口关闭事件
    def on_closing():
        app.running = False
        app.disconnect()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()