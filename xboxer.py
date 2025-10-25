import pygame
import threading
import time

class XboxController:
    def __init__(self, long_press_threshold = 1):
        """
        初始化Xbox手柄控制器
        """
        # 初始化pygame和手柄
        pygame.init()
        pygame.joystick.init()
        
        # 检查是否有手柄连接
        if pygame.joystick.get_count() == 0:
            raise Exception("未检测到Xbox手柄，请连接手柄后重试")
        
        # 初始化第一个手柄
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        # 按键状态标记字典 {按键索引: [短按标记, 长按标记]}
        self.button_states = {}
        # 按键按下时间记录
        self.button_press_times = {}
        # 长按阈值（秒）
        self.long_press_threshold = long_press_threshold
        
        # 初始化所有按键状态
        for i in range(self.joystick.get_numbuttons()):
            self.button_states[i] = [False, False]  # [短按标记, 长按标记]
            self.button_press_times[i] = 0
        
        # 遥杆轴数
        self.num_axes = self.joystick.get_numaxes()
        # 按键数
        self.num_buttons = self.joystick.get_numbuttons()
        # 帽子开关数
        self.num_hats = self.joystick.get_numhats()
        
        # 运行状态
        self.running = True
        
        # 存储摇杆、按键和帽子开关的状态
        self.left_stick = (0.0, 0.0)
        self.right_stick = (0.0, 0.0)
        self.triggers = (0.0, 0.0)
        self.hat = (0, 0)
        
        # 存储所有按键的短按和长按标志位
        self.button_short_flags = [False] * self.num_buttons
        self.button_long_flags = [False] * self.num_buttons
        self.button_current_states = [False] * self.num_buttons
        
        # 储存按键名称对应的状态
        self.namedbutton_states = {}
        # 储存按键名称
        self.button_names = ["A",       "B",        "X",        "Y",        
                             "LB",      "RB",       "Tab",      "Set",
                             "Xbox",    "LS",       "RS",       "Back",
                             "DPadUp",  "DPadDown", "DPadLeft", "DPadRight"]
        # 储存方向按键状态
        self.dirbutton = (0,0)
        
        # 启动事件监听线程
        self.event_thread = threading.Thread(target=self._event_loop)
        self.event_thread.daemon = True
        self.event_thread.start()
    
    def _event_loop(self):
        """
        事件监听循环
        """
        while self.running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    button_id = event.button
                    self.button_press_times[button_id] = time.time()
                    # 不再立即切换短按标记
                
                elif event.type == pygame.JOYBUTTONUP:
                    button_id = event.button
                    press_duration = time.time() - self.button_press_times[button_id]
                    
                    # 根据按压时间判断是短按还是长按
                    if press_duration < self.long_press_threshold:
                        # 短按：切换短按标记
                        self.button_states[button_id][0] = not self.button_states[button_id][0]
                    else:
                        # 长按：切换长按标记
                        self.button_states[button_id][1] = not self.button_states[button_id][1]
                
                elif event.type == pygame.JOYHATMOTION:
                    # 处理帽子开关（方向键）事件
                    self.dirbutton = event.value
            
            # 定时更新所有状态
            self._update_states()
            
            # 稍微延时以减少CPU占用
            time.sleep(0.01)
    
    def _update_states(self):
        """
        更新所有手柄状态
        """
        # 更新摇杆状态
        self.left_stick = (self.joystick.get_axis(0), -self.joystick.get_axis(1))
        self.right_stick = (self.joystick.get_axis(3), -self.joystick.get_axis(4))
        
        # 更新扳机键状态
        left_trigger = (self.joystick.get_axis(2) + 1) / 2
        right_trigger = (self.joystick.get_axis(5) + 1) / 2
        self.triggers = (left_trigger, right_trigger)
        
        # 更新帽子开关状态
        if self.num_hats > 0:
            self.hat = self.joystick.get_hat(0)
        
        # # 更新所有按键状态
        for i in range(self.num_buttons):
            # self.button_current_states[i] = 
            # self.button_short_flags[i] = 
            # self.button_long_flags[i] = 
            self.namedbutton_states[self.button_names[i]] = [self.joystick.get_button(i),
                                                             self.button_states[i][0],
                                                             self.button_states[i][1]]
        
    
    def reset_button_flags(self, button_id=None):
        """
        重置按键标记位
        
        Args:
            button_id (int, optional): 按键ID，如果为None则重置所有按键
        """
        if button_id is None:
            # 重置所有按键标记
            for button_id in self.button_states:
                self.button_states[button_id] = [False, False]
        else:
            # 重置指定按键标记
            if button_id in self.button_states:
                self.button_states[button_id] = [False, False]
    
    def stop(self):
        """
        停止手柄监听
        """
        self.running = False
        if self.event_thread.is_alive():
            self.event_thread.join(timeout=1)  # 等待线程结束，最多等待1秒
        pygame.quit()

# 使用示例
# 使用示例
if __name__ == "__main__":
    try:
        # 创建手柄控制器实例
        controller = XboxController()
        print("Xbox手柄已连接")
        print("按Ctrl+C退出")
        
        # 循环读取手柄状态（模拟外部程序的主循环）
        for i in range(10000):  # 运行一段时间来演示非阻塞特性
            # 清屏
            print("\033[H\033[J", end="")  # ANSI转义序列清屏
            
            # 打印摇杆状态
            print("=== 摇杆状态 ===")
            print(f"左摇杆:  X={controller.left_stick[0]:+.3f}, Y={controller.left_stick[1]:+.3f}")
            print(f"右摇杆:  X={controller.right_stick[0]:+.3f}, Y={controller.right_stick[1]:+.3f}")
            
            # 打印扳机键状态
            print("\n=== 扳机键状态 ===")
            print(f"左扳机:  {controller.triggers[0]:.3f}")
            print(f"右扳机:  {controller.triggers[1]:.3f}")
            
            # 打印普通按键状态
            print("\n=== 普通按键状态 ===")
            for i, button_name in enumerate(controller.button_names[:controller.num_buttons]):
                if i < len(controller.button_current_states):
                    current_state = controller.namedbutton_states[button_name][0]
                    short_flag = controller.namedbutton_states[button_name][1]
                    long_flag = controller.namedbutton_states[button_name][2]
                    print(f"{button_name:>6}: 当前={int(current_state)}, 短按标记={int(short_flag)}, 长按标记={int(long_flag)}")
            
            # 打印方向键状态
            print("\n=== 方向键状态 ===")
            print(f"帽子开关原始值: {controller.dirbutton}")
            
            # 外部程序可以做其他事情
            time.sleep(0.1)
            
        print("\n外部程序主循环结束，但手柄仍在后台运行")
        
        # 程序可以继续执行其他任务
        time.sleep(2)
        print("程序即将退出")
        controller.stop()
            
    except KeyboardInterrupt:
        print("\n退出程序")
        controller.stop()
    except Exception as e:
        print(f"错误: {e}")