# -*- coding=utf-8 -*-
# !/usr/bin/env python3
import sys
import time
import serial
from cushy_serial import CushySerial
from parameter_interface import *
import math as cm
import struct

class DrEmpower_can(object):
    def __init__(self, com='', uart_baudrate=115200, ser=0):
        self.MainSer = ser
        self.WaitFlag = [0,0,0,0,0,0,0]
        self.MovementDone = 0

    # 模式设置
    def set_mode(self, id_num=0, mode=2):
        """设置一体化关节模式。

        设置一体化关节进入不同的控制模式。

        Args：
            id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID，可以用 0 广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。
              mode：一体化关节模式编号，
                    mode=1：待机模式，一体化关节卸载
                    mode=2：闭环控制模式，运动控制函数必须在闭环控制模式下才能进行控制。（一体化关节上电后的默认模式）

        Returns：
             True：设置成功。
            False：设置失败。

        Raises：
            ---error in set_mode---

        """
        try:
            self.MainSer.send("N{}Q{}\r\n".format(id_num, mode))
        except Exception as e:
            print("---error in set_mode---：", e)
            return False
        return True
    # 重启
    def reboot(self, id_num=0):
        """一体化关节重启函数。

        设置一体化关节软件重启，效果与重新上电类似。

        Args:
            id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID 编号，可以用 0广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。

        Returns:
             True：开始重启。
            False：重启失败。

        Raises:
            “---error in reboot---”

        """
        try:
            self.MainSer.send("N{}O1\r\n".format(id_num))
        except Exception as e:
            print("---error in reboot---：", e)
            return False
        return True

    # 设置当前角度为机械零点，关机重启后依然有效
    def set_zero_position(self, id_num=0):
        """设置一体化关节零点角度函数

        设置当前角度为一体化关节输出轴零点，设置完后当前角度为 0 度，重启后保存。
        注：
            (1) 最好在修改其他参数前使用该函数，因该函数中包含 self.save_config() 函数，会将其他参数一并保存。
            (2) 建议在无负载的情况下执行此命令，否则可能造成关节短暂卸载

        Args：
            id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID，可以用 0 广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。

        Returns：
             True：设置成功。
            False：设置失败。

        Raises：
            “---error in set_zero_position---”

        """
        try:
            self.MainSer.send("N{}P1\r\n".format(id_num))
        except Exception as e:
            print("---error in set_zero_position---：", e)
            return False
        return True
    # 设置当前角度为临时零点，关机重启后失效，注意该函数后不可以使用 self.save_config()，否则会改变原定的零点位置
    def set_zero_position_temp(self, id_num=0):  # 中空关节无此功能
        """设置一体化关节零点角度函数，当次启动有效，重启后失效

        设置当前角度为一体化关节输出轴零点，设置完后当前角度为 0 度，重启后丢失。

        Args：
            id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID，可以用 0 广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。

        Returns：
             True：设置成功。
            False：设置失败。

        Raises：
            “---error in set_zero_position_temp---”

        """
        try:
            self.MainSer.send("N{}P0\r\n".format(id_num))
        except Exception as e:
            print("---error in set_zero_position_temp---：", e)
            return False
        return True

    # 绝对角度控制
    def set_angle(self, id_num=0, angle=0, speed=0, param=0, mode=0):
        """单个一体化关节角度控制函数。

        控制指定 ID 编号的一体化关节按照指定的转速转动到指定的角度（绝对角度，相对于用户设定的零点角度）。

        Args：
            id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID，可以用 0 广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。
             angle：一体化关节绝对角度（°）
             speed：一体化关节转速（r/min），具体含义由 mode 的取值决定。
                    mode=1：目标转速；
                    mode=0/2：前馈转速（r/min）
             param：运动参数，由 mode 取值决定，
                    mode=0：角度输入滤波带宽（<300）；
                    mode=1：启动和停止阶段角加转速（r/min/s）；
                    mode=2：前馈力矩 torque（Nm）。
              mode：角度控制模，一体化关节支持三种角度控制模式，由 mode 取值决定，
                    mode=0：轨迹跟踪模式，适合多个轨迹点输入后进行平滑控制，角度输入滤波带宽参数需设置为指令发送频率的一半；
                    mode=1：梯形轨迹模式，这种模式下可以指定运动过程中的目标转速和启停加转速；
                    mode=2：前馈控制模式，这种模式下的 speed 和 torque 分别为前馈控制量。前馈控制在原有 PID 控制基础上加入转速和力矩前馈，
                              提高系统的响应特性和减少静态误差。

        Note：在 mode=1 梯形轨迹模式中，speed 和 accel 都要大于 0；mode=0 时 speed 不起作用。

        Returns：
             True：运行正常。
            False：出现异常。

        Raises：
            “---error in set_angle---”

        """
        try:
            self.MainSer.send("N{}U{}W{}X{}Y{}\r\n".format(id_num, angle, speed, param, mode))
        except Exception as e:
            print("---error in set_angle---：", e)
            return False
        return True
    # 多个一体化关节绝对角度控制
    def set_angles(self, id_list=[1, 2, 3], angle_list=[0, 0, 0], speed=10, param=10, mode=1):
        """多个一体化关节控制函数。

        控制多个编号的一体化关节按照一定转速转动到指定的角度。

        Args：
               id_list：一体化关节 ID 编号组成的列表。
            angle_list：目标角度（°）组成的列表。
                 speed：指定转速（r/min），由 mode 取值决定：
                        mode=1，多个一体化关节中转速最大者的转速；
                        mode=0/2，或前馈转速。
                 param：运动参数，由 mode 取值决定：
                        mode=0，角度输入滤波带宽（<300）；
                        mode=1，启动和停止阶段加转速（r/min/s）；
                        mode=2，前馈力矩（Nm)。
                  mode：控制模式选择，一体化关节支持三种角度控制模式，由 mode 取值决定：
                        mode=0：多个一体化关节轨迹跟踪模式，适合多个轨迹点输入后进行平滑控制，角度输入带宽参数需设置为指令发送频率的一半；
                        mode=1：多个一体化关节梯形轨迹模式，此时 speed 为这些一体化关节的最快转速（r/min），param为目标加转速（r/min/s）；
                        mode=2：前馈控制模式，这种模式下的 speed 和 torque 分别为前馈控制量。前馈控制在原有PID控制基础上加入转速和力矩前馈，
                                提高系统的响应特性和减少静态误差。

        Returns：
             True：运行正常。
            False：出现异常。

        Raises：
            如果 id_list 和 angle_list 长度不一致时会提示。
            “---error in set_angles---”

        """
        try:
            if len(id_list) == len(angle_list):
                for i in range(len(id_list)):
                    self.MainSer.send("N{}U{}".format(id_list[i], angle_list[i]))
                self.MainSer.send("W{}X{}Y{}T{}\r\n".format(speed, param, mode, len(id_list)))
            else:
                print("id_list 和 angle_list 长度不一致")
        except Exception as e:
            print("---error in set_angles---：", e)
            return False
        return True

    # 相对角度控制
    def step_angle(self, id_num=1, angle=0, speed=0, param=0, mode=0):
        """单个一体化关节相对角度控制函数。

        控制指定编号的一体化关节按照指定的转速相对转动指定的角度（相对角度，相对于发送该指令时的角度）。

        Args：
            id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID，可以用 0 广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。
             angle：一体化关节相对角度（°）
             speed：指定转速（r/min），由 mode 取值决定：
                    mode=1，目标转速；
                    mode=0/2, 前馈转速（r/min）。
             param：运动参数，由 mode 取值决定，
                    mode=0，角度输入滤波带宽（<300）；
                    mode=1，启动和停止阶段加转速（r/min/s）；
                    mode=2，前馈力矩 torque（Nm）。
              mode：角度控制模，一体化关节支持三种角度控制模式，由 mode 取值决定，
                    mode=0：轨迹跟踪模式，适合多个轨迹点输入后进行平滑控制，角度输入滤波带宽参数需设置为指令发送频率的一半；
                    mode=1：梯形轨迹模式，这种模式下可以指定运动过程中的目标转速和启停加转速；
                    mode=2：前馈控制模式，这种模式下的 speed 和 torque 分别为前馈控制量，前馈控制在原有PID控制基础上加入转速和力矩前馈，
                            提高系统的响应特性和减少静态误差。

        Note：在 mode=1 梯形轨迹模式中，speed 和 accel 都要大于 0；mode=0 时 speed 不起作用。

        Returns：
             True：运行正常。
            False：出现异常。

        Raises:
            “---error in step_angle---”

        """
        try:
            pass
        except Exception as e:
            print("---error in step_angle---：", e)
            return False
        return True
    # 多个一体化关节相对角度同步控制
    def step_angles(self, id_list=[1, 2, 3], angle_list=[0, 0, 0], speed=10, param=10, mode=1):
        """多个一体化关节控制函数。

        控制多个一体化关节按照指定的时间先对转动给定角度。

        Args:
               id_list：一体化关节 ID 编号组成的列表。
            angle_list：相对目标角度组成的列表。
                 speed：指定转速（r/min），由 mode 取值决定：
                        mode=1，多个一体化关节中转速最大者的转速；
                        mode=0/2，或前馈转速。
                 param：运动参数，由 mode 取值决定：
                        mode=0，角度输入滤波带宽（<300）；
                        mode=1，启动和停止阶段加转速（r/min/s）；
                        mode=2，前馈力矩（Nm)。
                  mode：控制模式选择，一体化关节支持三种角度控制模式，由 mode 取值决定：
                        mode=0：多个一体化关节轨迹跟踪模式，适合多个轨迹点输入后进行平滑控制，角度输入带宽参数需设置为指令发送频率的一半；
                        mode=1：多个一体化关节梯形轨迹模式，此时 speed 为这些一体化关节的最快转速（r/min），param为目标加转速（r/min/s）；
                        mode=2：前馈控制模式，这种模式下的 speed 和 torque 分别为前馈控制量。前馈控制在原有PID控制基础上加入转速和力矩前馈，
                                提高系统的响应特性和减少静态误差。

        Returns:
             True：运行正常。
            False：出现异常。

        Raises:
            如果 id_list 和 angle_list 长度不一致时会提示。
            ---error in step_angles---

        """
        try:
            if len(id_list) == len(angle_list):
                pass
            else:
                print("id_list 和 angle_list 长度不一致")
        except Exception as e:
            print("---error in step_angles---：", e)
            return False
        return True

    # 单个关节自适应绝对角度控制
    def set_angle_adaptive(self, id_num=0, angle=0, speed=0, torque=0):
        """单个一体化关节角度自适应函数。

        控制指定 ID 编号的一体化关节按照限定的转速和力矩转动到指定的角度（绝对角度，相对于用户设定的零点角度）。
        注：当设置的转速相对于力矩过大，或力矩相对于转速过小，则关节无法在短时间内提供足够的加速度使得转速将为 0，此时若关节未遇阻力会出现在目标角度过冲
        现象，此为物理规律，暂时没有好的解决办法

        Args：
            id_num：一体化关节 ID 编号,如果不知道当前一体化关节 ID，可以用 0 广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。
             angle：一体化关节角度（°）。
             speed：限定转速值（r/min）。
            torque：限定力矩值（Nm)。

        Returns：
             True：运行正常。
            False：出现异常。

        Raises：
            “---error in set_angle_adaptive---”

        """
        factor = 0.01
        try:
            #   所有参数都发
            # self.MainSer.send("N{}d{}b{}c{}\r\n".format(id_num, angle, speed, torque))
            #   只发位置，速度和力矩使用预设
            self.MainSer.send("N{}d{}\r\n".format(id_num, angle))
        except Exception as e:
            print("---error in set_angle_adaptive---：", e)
            return False
        return True
    # 多个关节自适应绝对角度控制
    def set_angles_adaptive(self, id_list=[0, 0, 0], angle_list=[0, 0, 0], speed_list=[0, 0, 0], torque_list=[0, 0, 0]):
        """多个一体化关节角度自适应函数。

        控制多个一体化关节按照限定的转速和力矩转动到指定的角度（绝对角度，相对于用户设定的零点角度）。
        注：当设置的转速相对于力矩过大，或力矩相对于转速过小，则关节无法在短时间内提供足够的加速度使得转速将为 0，此时若关节未遇阻力会出现在目标角度过冲
        现象，此为物理规律，暂时没有好的解决办法

        Args：
                id_list：一体化关节 ID 编号组成的列表。
             angle_list：一体化关节角度（°）组成的列表。
             speed_list：限定转速值（r/min）组成的列表。
            torque_list：限定力矩值（Nm)组成的列表。

        Returns：
             True：运行正常。
            False：出现异常。

        Raises：
            关节数量与参数数量不一致，会提示
            “---error in set_angles_adaptive---”

        """
        try:
            if len(id_list) == len(angle_list):
                for i in range(len(id_list)):
                    #   所有参数都发
                    # self.MainSer.send("N{}d{}b{}c{}".format(id_num, angle, speed, torque))
                    #   只发位置，速度和力矩使用预设
                    self.MainSer.send("N{}d{}".format(id_list[i], angle_list[i]))
                self.MainSer.send("T{}\r\n".format(len(id_list)))
            else:
                print("id_list 和 angle_list 长度不一致")

        except Exception as e:
            print("---error in set_angles_adaptive---：", e)
            return False
        return True

    # 单个关节阻抗控制
    def impedance_control(self, id_num=0, angle=0, speed=0, tff=0, kp=0, kd=0, mode=1):
        """单个一体化关节阻抗控制函数。

        对指定 ID 编号的一体化关节进行阻抗控制。该函数执行结束后关节会停在目标角度 angle，并对外表现出一定柔性。

        Args：
            id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID，可以用 0 广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。
             angle：一体化关节目标角度（°）。
             speed：一体化关节目标转速（r/min）。
               tff：前馈力矩（Nm)。
                kp：角度刚度系数（Nm/°），需大于 0。
                kd：转速阻尼系数（Nm/(r/min)），需大于 0。
              mode: 模式，等于 1 时以角度为控制目标，为 0 时以力为控制目标

        Note：该函数直接控制关节输出力矩，其目标输出力矩计算公式如下：
              torque = kp*( angle – angle_) + tff + kd*(speed – speed_)
              其中 angle_ 和 speed_ 分别为输出轴当前实际角度（°）和当前实际转速（r/min）, kp 和 kd 为刚度系数和阻尼系数。

        Returns：
             True：运行正常
            False：出现异常

        Raises：
            “---error in impedance_control---”

        """
        try:
            pass
        except Exception as e:
            print("---error in impedance_control---：", e)
            return False
        return True
    # 多个关节阻抗控制
    def impedance_control_multi(self, id_list=[1, 2, 3], angle_list=[0, 0, 0], speed_list=[0, 0, 0], tff_list=[0, 0, 0],
                                kp_list=[0, 0, 0], kd_list=[0, 0, 0], mode=1):
        """多个一体化关节阻抗控制函数。

        对多个一体化关节进行阻抗控制。该函数执行结束后被指定的各个关节会停在目标角度 angle，并对外表现出一定柔性。

        Args：
               id_list：一体化关节 ID 编号组成的列表。
            angle_list：一体化关节目标角度组成的列表（°）。
            speed_list：一体化关节目标转速组成的列表（r/min）。
              tff_list：前馈力矩组成的列表（Nm）。
               kp_list：角度刚度系数组成的列表（Nm/°），每个元素均需大于 0。
               kd_list：转速阻尼系数组成的列表（Nm/(r/min)），每个元素均需大于 0。

        Note：该函数直接控制关节输出力矩，其目标输出力矩计算公式如下：
              torque = kp_list[i] * (angle_list[i] – angle_[i]) + tff_list[i] + kd_list[i] * (speed_list[i] – speed_[i])
              其中 angle_[i] 和 speed_[i] 分别为对应关节输出轴当前实际角度（度）和当前实际转速（r/min）, kp_list[i] 和 kd_list[i]
              为刚度系数和阻尼系数。

        Returns：
            True：运行正常。
            False：出现异常。

        Raises：
            “---error in impedance_control_multi---”

        """

        try:
            pass
        except Exception as e:
            print("---error in impedance_control_multi--：", e)
            return False
        return True

    # 单个关节运动助力
    def motion_aid(self, id_num=0, angle=0, speed=0, angle_err=0, speed_err=0, torque=0):
        """单个一体化关节运动跟随与助力函数。

        指定 ID 编号的一体化关节进行运动跟随与助力。
        当关节在停止状态下检测到角度差 angle_err 和转速差 speed_err 时向目标角度 angle 方向提供力矩大小为 torque 的助力，
        并在到达 angle 后停止并保持位置。
        注：
          a、当助力与外部驱动力之和大于阻力，关节会持续转动；
          b、当助力与外部驱动力之和小于阻力，关键开始减速，当转速小于 2 倍转速差值 speed_err 时，关节停止输出助力；
          c、一般情况下，该功能为人进行助力，强烈建议用户将助力力矩设置在人力所能及的范围内，即人力可使关节停止转动；
          d、若必须设置超出人力的力矩，则必须在合理位置设置牢固的机械限位，以避免超出运动范围给人或物体带来损伤。

        Args：
               id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID，可以用 0 广播，此时总线上有多个一体化关节，则多个一体化关节都会执行该操作。
                angle: 目标角度（°），即助力目标角度，该值减去关节当前角度即为助力行程，目标角度范围为 -300~300°。
                speed：限定转速（r/min），即助力的限定转速，防止助力力矩引起的一直加速导致转速过快。
            angle_err：角度差值（°），表示运动跟随与助力的角度灵敏度。
            speed_err：转速差值（r/min），表示运动跟随与助力的转速灵敏度。
               torque：助力力矩（Nm)。

        Returns：
             True：运行正常。
            False：出现异常。

        Raises：
            “---error in motion_follow_aid---”

        """
        try:
            pass
        except Exception as e:
            print("---error in motion_follow_aid---：", e)
            return False
        return True
    # 多个关节运动助力
    def motion_aid_multi(self, id_list=[1, 2, 3], angle_list=[0, 0, 0], speed_list=[0, 0, 0], angle_err_list=[1, 1, 1],
                         speed_err_list=[1, 1, 1], torque_list=[1, 1, 1]):
        """多个个一体化关节运动跟随与助力函数。

        指定多个一体化关节进行运动跟随与助力。
        当关节在停止状态下检测到角度差 angle_er_list[i] 和转速差 speed_err_list[i] 时向目标角度 angle_list[i] 方向提供力矩大小为
        torque_list[i] 的助力，并在到达 angle_list[i] 后停止并保持位置。
        注：
          a、当助力与外部驱动力之和大于阻力，关节会持续转动；
          b、当助力与外部驱动力之和小于阻力，关键开始减速，当转速小于 2 倍转速差值 speed_err_list[i] 时，关节停止输出助力；
          c、一般情况下，该功能为人进行助力，强烈建议用户将助力力矩设置在人力所能及的范围内，即人力可使关节停止转动；
          d、若必须设置超出人力的力矩，则必须在合理位置设置牢固的机械限位，以避免超出运动范围给人或物体带来损伤。

        Args：
                   id_list：一体化关节 ID 编号组成的列表。
                angle_list：目标角度组成的列表（°），即助力目标角度，该值减去关节当前角度即为助力行程。
                speed_list：限定转速组成的列表（r/min），即助力的限定转速，防止助力力矩引起的一直加速导致转速过快。
            angle_err_list：角度差值（°）组成的列表，表示运动跟随与助力的角度灵敏度。
            speed_err_list：转速差值（r/min）组成的列表，表示运动跟随与助力的转速灵敏度。
               torque_list：助力力矩（Nm)组成的列表。

        Returns：
             True：运行正常。
            False：出现异常。

        Raises：
            “---error in motion_follow_aid_multi---”
        """
        try:
            pass
        except Exception as e:
            print("---error in motion_follow_aid_multi---：", e)
            return False
        return True

    # 检查并等待单个关节运动到位
    def position_done(self, id_num=0):
        pass
        """检查并等待单个一体化关节在梯形轨迹模式下是否运动到位函数。

        检查并等待单个一体化关节是否转动到指定角度。
        该函数支持以下三种角度控制模式：
                                 梯形轨迹模式，mode=1
                                 前馈模式，mode=2
                                 自适应模式下，set_angle_adaptive()

        Args:
            id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID，可以用 0 广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。

        Returns:
            True：转动到指定角度。
            None：尚未到达指定角度。

        Raises:
            “---error in position_done---“

        """
        try:
            self.positions_done(id_list=[id_num])
        except Exception as e:
            print("---error in position_done---：", e)
            return False
    # 检查并等待多个关节运动到位
    def positions_done(self, id_list=[1, 2, 3]):
        """检查并等待多个一体化关节在梯形轨迹模式下是否运动到位函数。

        检查并等待多个一体化关节是否全部转动到指定角度。
        该函数支持以下三种角度控制模式：
                                 梯形轨迹模式，mode=1
                                 前馈模式，mode=2
                                 自适应模式下，set_angle_adaptive()

        Args：
            id_num：一体化关节 ID 编号组成的列表。

        Returns：
            True：全部转动到指定角度。
            None：尚未全部到达指定角度。

        Raises：
            ”---error in positions_done---“

        """
        try:
            pass
            # self.MainSer.send("Z1\r\n)
            # for i in range(len(id_list)):
            #     if not self.WaitFlag[i-1]:
            #         self.MainSer.send("N{}Z1\r\n".format(id_list[i]))
            #         self.WaitFlag[i - 1] = 1
        except Exception as e:
            print("---error in positions_done---：", e)
            return False
        return True

    # 转速控制
    def set_speed(self, id_num=0, speed=10, param=0, mode=1):
        """单个一体化关节转速控制函数。

        控制指定 ID 编号的一体化关节按照指定的转速连续整周转动（转动到关节支持的极限角度后自动停止）。

        Args：
            id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID，可以用 0 广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。
             speed：目标转速（r/min）。
             param：运动参数，由 mode 取值决定，
                    mode=0，前馈力矩（Nm)；
                    mode!=0，或目标加转速（r/min/s）。
              mode：控制模式选择，由 mode 取值决定：
                    mode=0，转速直接控制模式，将一体化关节目标转速直接设为 speed；
                    mode!=0，匀加速控制模式，一体化关节将按照目标角加速变化到 speed。

        Note:
            在 mode!=0，即匀加速模式下，如果目标角加速度设置为 0，则一体化关节转速将保持当前值不变。

        Returns：
             True：运行正常。
            False：出现异常。

        Raises:
            "---error in set_speed---"

        """
        try:
            pass
        except Exception as e:
            print("---error in set_speed---：", e)
            return False
        return True
    # 多个一体化关节转速控制
    def set_speeds(self, id_list=[1, 2, 3], speed_list=[10.0, 20.0, 30.0], param=0, mode=1):
        """多个一体化关节转速控制函数。

        控制多个一体化关节按照指定的转速连续整周转动（转动到关节支持的极限角度后自动停止）。

        Args：
               id_list：一体化关节 ID 编号组成的列表。
            speed_list：一体化关节目标转速（r/min）组成的列表。
                 param：运动参数，由 mode 取值决定：
                        mode=0，前馈力矩（Nm)；
                        mode!=0，或目标加转速（r/min/s）。
                  mode：控制模式选择，由 mode 取值决定：
                        mode=0，转速直接控制模式，将一体化关节目标转速直接设为 speed；
                        mode!=0，匀加速控制模式，一体化关节将按照目标角加速变化到 speed。

        Note：
            在 mode!=0，即匀加速模式下，如果目标角加速度设置为 0，则一体化关节转速将保持当前值不变。

        Returns：
             True：运行正常。
            False：出现异常。

        Raises：
            “---error in set_speeds---”

        """
        try:
            pass
        except Exception as e:
            print("---error in set_speeds---：", e)
            return False
        return True

    # 力矩控制
    def set_torque(self, id_num=0, torque=0.1, param=0, mode=1):
        """单个一体化关节力矩（电流）闭环控制函数。

        控制指定 ID 编号的一体化关节输出指定的力矩（Nm），若阻力不足以抵抗该力矩，则关节会持续转动（转动到关节支持的极限角度后自动停止）。

        Args：
            id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID，可以用 0 广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。
            torque：目标力矩（Nm）。
             param：运动参数，由 mode 取值决定：
                    mode=0，参数不起作用；
                    mode!=0，力矩在单位时间内的增量（Nm/s）。
              mode：控制模式选择，由 mode 取值决定：
                    mode=0，力矩直接控制模式，将一体化关节目标转速直接设为 torque；
                    mode!=0，力矩匀速增加模式，一体化关节将按照指定的单位时间内的增量匀速变化到 torque。

        Note：
            在 mode!=0，即力矩匀速增加模式下，如果目标单位时间增量设置为 0，则一体化关节输出力矩将保持当前值不变。

        Returns:
            True：运行正常。
            False：出现异常。

        Raises:
            “---error in set_torque---”

        """
        try:
            self.MainSer.send("N{}V{}X{}Y{}\r\n".format(id_num, torque, param, mode))
        except Exception as e:
            print("---error in set_torque---：", e)
            return False
        return True
    # 多个一体化关节力矩控制
    def set_torques(self, id_list=[1, 2, 3], torque_list=[3.0, 4.0, 5.0], param=0, mode=1):
        """多个一体化关节力矩控制函数。

        控制多个一体化关节输出指定的力矩，若阻力不足以抵抗该力矩，则关节会持续转动（转动到关节支持的极限角度后自动停止）。

        Args：
                id_list：一体化关节 ID 编号组成的列表。
            torque_list：一体化关节目标力矩（Nm）组成的列表。
                  param：运动参数，由 mode 取值决定：
                        mode=0，该参数不起作用；
                        mode!=0，力矩在单位时间内的增量（Nm/s）。
                   mode：控制模式选择，由 mode 取值决定，
                         mode=0，力矩直接控制模式，将一体化关节目标转速直接设为 torque；
                         mode!=0，力矩匀速增加模式，一体化关节将按照指定的单位时间内的增量匀速变化到 torque。

        Note：
            在 mode!=1，即力矩匀速增加模式下，如果目标单位时间增量设置为 0，则一体化关节输出力矩将保持当前值不变。

        Returns：
             True：运行正常。
            False：出现异常。

        Raises:
            "---error in set_torques---"

        """
        try:
            if len(id_list) == len(torque_list):
                for i in range(len(id_list)):
                    self.MainSer.send("N{}V{}".format(id_list[i], torque_list[i]))
                self.MainSer.send("X{}Y{}T{}\r\n".format( param, mode, len(id_list)))
            else:
                print("id_list 和 torque_list 长度不一致")
        except Exception as e:
            print("---error in set_torques---：", e)
            return False
        return True

    # 急停
    def estop(self, id_num=0):
        """急停函数

        控制一体化关节紧急停止。如果想控制多个关节同时急停，则可使 id_num=0。

        Args：
            id_num：一体化关节 ID 编号。如果不知道当前一体化关节 ID，可以用 0 广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。

        Returns:
             True：运行正常。
            False：出现异常。

        Raises：
            “---error in estop---”

        """
        try:
            pass
        except Exception as e:
            print("---error in estop---：", e)
            return False
        return True
    # 读取一体化关节ID号
    def get_id(self):
        """读取一体化关节 ID 编号。

        读取一体化关节 DI 编号。注意使用该函数时总线上只能接 1 个一体化关节。

        Args：
            无。

        Returns：
               id：关节 ID 编号。
            False：出现异常。

        Raises：
            “error in get_id”

        """
        try:
            id = 1
            return id
        except Exception as e:
            print("---error in get_id---：", e)
            return False
    def get_ids(self):
        """读取总线中多个一体化关节的 ID 编号。

            读取总线中多个一体化关节 DI 编号。

            Args：
                无。

            Returns：
                   id：关节 ID 编号组成的列表。
                False：出现异常。

            Raises：
                “error in get_ids”

            """
        try:
            id_list = [1,2,3,4,5,6,7]
            print("总线中存在的不同 ID 号为", id_list)
            return id_list
        except Exception as e:
            print("---error in get_id---：", e)
            return False


    # 设置临时角度限位
    def set_angle_range(self, id_num=0, angle_min=-180, angle_max=180):
        """设置一体化关节运行过程中的极限角度，设置成功后一体化关节的可控制的转动角度将限定在[angle_min, angle_max]范围内。关机重启后失效。

        注：
            a、使用该函数时输出轴角度必须在[angle_min, angle_max]范围内，否则将设置失败；
            b、该功能只在本次开机运行过程中有效，对应地将在关节重启后失效；
            c、关节本身还有一个极限角度属性，默认生效，范围为[-180.5°, 180.5°]，
               该属性不受该函数影响，且每次开机重启均有效，如需重新设置或取消该属性，请使用
               set_angle_range_config() 和 disable_angle_range_config()，详见对应函数说明。

        Args：
               id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID 编号，可以用 0 广播，如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。
            angle_min：最小限位角度（°）。
            angle_max：最大限位角度（°）。

        Returns：
             True：设置成功。
            False：设置失败。

        Raises：
            ---error in set_angle_range---

        """
        try:
            self.MainSer.send("N{}R{}S{}Y0\r\n".format(id_num, angle_min, angle_max))
        except Exception as e:
            print("---error in set_angle_range---：", e)
            return False
        return True
    # 取消临时角度限位
    def disable_angle_range(self, id_num=0):
        """取消本次运行期间一体化关节运行过程中的角度限位。

        Args：
            id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID 编号，可以用 0 广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。

        Returns：
             True：取消成功。
            False：取消失败。

        Raises:
            “---error in disable_angle_range---”

        """
        try:
            self.MainSer.send("N{}R0S0Y0\r\n".format(id_num))
        except Exception as e:
            print("---error in disable_angle_range---：", e)
            return False
        return True
    # 设置角度限位属性
    def set_angle_range_config(self, id_num=0, angle_min=-180, angle_max=180):
        """设置一体化关节极限角度属性，设置成功后一体化关节的可控制的转动角度将限定在[angle_min, angle_max]范围内，每次开机重启均默认有效。

        注：
            a、使用该函数时输出轴角度必须在[angle_min, angle_max]范围内，否则将设置失败；
            b、限位范围设置成功后，再执行 self.save_config() 函数，每次开机重启后均有效；
            c、如需取消该属性影响，请使用 disable_angle_range_config() 函数将该属性关闭，则本次开机该属性不起作用；
               若随后使用 self.save_config() 则该属性将永久失去，如需找回该属性，则再次使用本函数即可。

        Args：
               id_num：一体化关节 ID 编号,如果不知道当前一体化关节 ID 编号，可以用 0 广播，如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。
            angle_min：最小限位角度（°）
            angle_max：最大限位角度（°）

        Returns：
             True：设置成功。
            False：设置失败。

        Raises:
            “---error in set_angle_range_config---”

        """
        try:
            self.MainSer.send("N{}R{}S{}Y1\r\n".format(id_num, angle_min, angle_max))
        except Exception as e:
            print("---error in set_angle_range_config---：", e)
            return False
        return True
    # 取消角度限位属性
    def disable_angle_range_config(self, id_num=0):
        """取消一体化关节角度限位属性。

        Args：
            id_num：一体化关节 ID 编号,如果不知道当前一体化关节 ID 编号，可以用 0广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。

        Returns：
             True：取消成功。
            False：取消失败。

        Raises：
            “---error in disable_angle_range_config---”

        """
        try:
            self.MainSer.send("N{}R0S0Y1\r\n".format(id_num))
        except Exception as e:
            print("---error in disable_angle_range_config---：", e)
            return False
        return True

    # 设置输出轴转速限制
    def set_speed_limit(self, id_num=0, speed_limit=1):
        """设置一体化关节转速限制函数。

            设置一体化关节转速限制 speed_limit （r/min），此后关节转速绝对值不超过 speed_limit。
            注意：
                1. 该函数执行完转速限制在本次开机运行期间有效，关机或重启后将失效。
                2. 若想关机重启前取消该限制，只需设置一个非常大的数值，比如令 speed_limit = 100000。
                3. 如用户决定永久保持转速限制 speed_limit，请紧接着使用 self.save_config 函数（慎用）。
            Args:
                     id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID 编号，可以用 0广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。
                speed_limit：转速限制（r/min）（必须大于 0）。

            Returns:
                 True：设置成功。
                False：设置失败。

            Raises:
                ---error in set_speed_limit---

            """
        try:
            if speed_limit <= 0:
                print("speed_limit 必须大于 0，请重新输入")
                return False
            else:
                self.preset_angle(id_num=id_num, angle=abs(speed_limit), t=0, param=0, mode=1)
                order_num = 0x18
                data = self.format_data([order_num], 'u32', 'encode')  # # 将 order_num 转换成占4字节的数据，
                self.send_command(id_num=id_num, cmd=0x08, data=data, rtr=0)  # 需要用标准帧（数据帧）进行发送，不能用远程帧
        except Exception as e:
            print("---error in set_speed_limit---：", e)
            return False
        return True
    # 设置输出轴力矩限制
    def set_torque_limit(self, id_num=0, torque_limit=1):
        """设置一体化关节力矩限制函数。

            设置一体化关节力矩限制 torque_limit （Nm），此后关节力矩绝对值不超过 torque_limit。
            注意：
                1. 该函数执行完力矩限制在本次开机运行期间有效，关机或重启后将失效。
                2. 若想关机重启前取消该限制，只需设置一个非常大的数值，比如令 torque_limit = 100000。
                3. 如用户决定永久保持力矩限制 torque_limit，请紧接着使用 self.save_config 函数（慎用）。
            Args:
                      id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID 编号，可以用 0广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。
                torque_limit：力矩限制（Nm）（必须大于 0）。

            Returns:
                 True：设置成功。
                False：设置失败。

            Raises:
                ---error in set_toque_limit---

            """
        try:
            if torque_limit <= 0:
                print("torque_limit 必须大于 0，请重新输入")
                return False
            else:
                self.preset_angle(id_num=id_num, angle=abs(torque_limit), t=0, param=0, mode=1)
                order_num = 0x19
                data = self.format_data([order_num], 'u32', 'encode')  # # 将 order_num 转换成占4字节的数据，
                self.send_command(id_num=id_num, cmd=0x08, data=data, rtr=0)  # 需要用标准帧（数据帧）进行发送，不能用远程帧
        except Exception as e:
            print("---error in set_toque_limit---：", e)
            return False
        return True
    # 设置输出轴自适应转速
    def set_speed_adaptive(self, id_num=0, speed_adaptive=1):
        """设置一体化关节转自适应速函数。

            设置一体化关节转速限制 speed_adaptive （r/min），此后关节自适应转速绝对值不超过 speed_adaptive。
            注意：
                1. 该函数执行完转速限制在本次开机运行期间有效，关机或重启后将失效。
                2. 若想关机重启前取消该限制，只需设置一个非常大的数值，比如令 speed_adaptive = 100000。
                3. 本函数需在运动指令之后运行。
            Args:
                        id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID 编号，可以用 0广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。
                speed_adaptive：自适应转速限制（r/min）（必须大于 0）。

            Returns:
                 True：设置成功。
                False：设置失败。

            Raises:
                ---error in set_speed_adaptive---

            """
        try:
            if speed_adaptive <= 0:
                print("speed_adaptive 必须大于 0，请重新输入")
                return False
            else:
                self.preset_angle(id_num=id_num, angle=abs(speed_adaptive), t=0, param=0, mode=1)
                order_num = 0x20
                data = self.format_data([order_num], 'u32', 'encode')  # # 将 order_num 转换成占4字节的数据，
                self.send_command(id_num=id_num, cmd=0x08, data=data, rtr=0)  # 需要用标准帧（数据帧）进行发送，不能用远程帧
        except Exception as e:
            print("---error in set_speed_adaptive---：", e)
            return False
        return True
    # 设置输出轴自适应力矩
    def set_torque_adaptive(self, id_num=0, torque_adaptive=1):
        """设置一体化关节自适应力矩函数。

            设置一体化关节力矩限制 torque_adaptive （Nm），此后关节自适应力矩绝对值不超过 torque_adaptive。
            注意：
                1. 该函数执行完力矩限制在本次开机运行期间有效，关机或重启后将失效。
                2. 若想关机重启前取消该限制，只需设置一个非常大的数值，比如令 torque_adaptive = 100000。
                3. 本函数需在运动指令之后运行。
            Args:
                      id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID 编号，可以用 0广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。
                torque_adaptive：力矩限制（Nm）（必须大于 0）。

            Returns:
                 True：设置成功。
                False：设置失败。

            Raises:
                ---error in set_toque_adaptive---

            """
        try:
            if torque_adaptive <= 0:
                print("torque_adaptive 必须大于 0，请重新输入")
                return False
            else:
                self.preset_angle(id_num=id_num, angle=abs(torque_adaptive), t=0, param=0, mode=1)
                order_num = 0x21
                data = self.format_data([order_num], 'u32', 'encode')  # # 将 order_num 转换成占4字节的数据，
                self.send_command(id_num=id_num, cmd=0x08, data=data, rtr=0)  # 需要用标准帧（数据帧）进行发送，不能用远程帧
        except Exception as e:
            print("---error in set_toque_adaptive---：", e)
            return False
        return True
    # 设置临时 PID
    def set_pid(self, id_num=0, P=20, I=25, D=20):
        """设置一体化关节控制环 PID 函数。

        设置一体化关节控制环的位置增益 P、转速增益 D、积分增益 I，以便实现调整关节控制性能的目的。
        该函数执行完 PID 的值在本次开机运行期间有效，关机或重启后将失效。
        如用户决定永久使用某组 PID 则可在使用该函数设置 PID 后，请紧接着使用 save_config 函数。

        Args:
            id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID 编号，可以用 0广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。
                 P：位置增益（必须大于 0）。
                 I：积分增益（必须大于 0）。
                 D：转速增益（必须大于 0）。

        Returns:
             True：设置成功。
            False：设置失败。

        Raises:
            ---error in set_pid---

        """
        try:
            if P <= 0 or D <= 0 or I <= 0:
                print("请输出大于 0 的 PID 数值")
                return False
            self.write_property(id_num=id_num, property='dr.controller.config.angle_gain', value=P)
            self.write_property(id_num=id_num, property='dr.controller.config.speed_integrator_gain', value=I)
            self.write_property(id_num=id_num, property='dr.controller.config.speed_gain', value=D)
        except Exception as e:
            print("---error in set_pid---：", e)
            return False
        return True


    # 设置CAN波特率
    def set_can_baud_rate(self, id_num=0, baud_rate=500000):
        """设置一体化关节 CAN 波特率。

        设置 CAN 波特率（关机后依然保存）。
        注：建议在无负载的情况下执行此命令，否则可能造成关节短暂卸载

        Args：
               id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID，可以用 0 广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。
            baud_rate：CAN 波特率，支持 125k、250k、500k、1M 中任意一种，修改成功后需手动将上位机 CAN 波特率也修改为相同值。

        Returns：
             True：设置成功。
            False：设置失败。

        Raises：
            ---error in set_can_baud_rate---

        """
        try:
            self.write_property(id_num=id_num, property='dr.can.config.baud_rate', value=baud_rate)
            self.save_config(id_num=id_num)
        except Exception as e:
            print("---error in set_can_baud_rate---：", e)
            return False
        return True
    def write_property(self, id_num=0, property='', value=0):
        """修改一体化关节属性参数。

        修改一体化关节属性参数，这里的属性参数为一体化关节控制参数，存放于 interface_enums.py 文件。

        Args：
              id_num：一体化关节 ID 编号,如果不知道当前一体化关节 ID 编号，可以用 0广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。
            property：需要设置的属性参数名称，例如 "dr.voltage"，"dr.config.can_id"等，
                      具体参数名称见 parameter_interface.py 文件里 property_address 字典里的键值。
               value：对应参数的目标值。

        Returns：
             True：设置成功。
            False：设置失败。

        Raises：
            “---error in self.write_property---”

        """
        try:
            data_types = {'f': 0, 'u16': 1, 's16': 2, 'u32': 3, 's32': 4}  # 字典
            address = property
            if type(property) == str:  # 判断属性名称是否为字符
                address = key_find_value(property)  # 返回属性对应的编码
                if address > 0:
                    pass
                    # print(str(value_find_key(address)) + ' address = ' + str(address))
                else:
                    print('invalid property: ' + str(property))
                    return False
            data_type = property_type.get(value_find_key(address))  # 确定属性的数据类型
            # print(data_type)
            data = self.format_data([address, data_types.get(data_type), value], 'u16 u16 ' + data_type, 'encode')
            self.send_command(id_num=id_num, cmd=0x1F, data=data, rtr=0)  # 需要用标准帧（数据帧）进行发送，不能用远程帧
        except Exception as e:
            print("---error in self.write_property---：", e)
            return False
        return True
    # 保存一体化关节配置
    def save_config(self, id_num=0):
        """保存配置函数。

        保存用户修改的一体化关节属性参数，这里的属性参数为一体化关节控制参数，存放于 interface_enums.py 文件。
        正常情况下，通过 self.write_property 函数修改的属性一体化关节关机或重启之后，会恢复为修改前的值；
        如果想永久保存，则需要用 self.save_config 函数将相关参数保存到 flash 中，关机或重启后不丢失。
        注：建议在无负载的情况下执行此命令，否则可能造成关节短暂卸载

        Args：
            id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID 编号，可以用 0广播，此时如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。

        Returns：
             True：设置成功。
            False：设置失败。

        Raises：
            “---error in self.save_config---”

        """
        try:
            order_num = 0x01
            data = self.format_data([order_num], 'u32', 'encode')
            self.send_command(id_num=id_num, cmd=0x08, data=data, rtr=0)  # 需要用标准帧（数据帧）进行发送，不能用远程帧
        except Exception as e:
            print("---error in self.save_config---：", e)
            return False
        time.sleep(2)
        return True

    """
    其他系统辅助函数，一般情况下无需使用
    """



    # 恢复出厂设置，不改变 ID 号
    def init_config(self, id_num=0):
        """恢复出厂设置。

        恢复出厂时的参数配置，不改变用户设置的 ID 号。
        注：建议在无负载的情况下执行此命令，否则可能造成关节短暂卸载

        Args：
            id_num：一体化关节 ID 编号，如果不知道当前一体化关节 ID，可以用 0 广播，如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。

        Returns:
             True：恢复出厂设置成功。
            False：恢复出厂设置失败。

        Raises:
            "---error in init_config---"

        """
        try:
            data = self.format_data([id_num], 'u32', 'encode')
            self.send_command(id_num=id_num, cmd=0x0E, data=data, rtr=0)  # 需要用标准帧（数据帧）进行发送，不能用远程帧
        except Exception as e:
            print("---error in init_config---：", e)
            return False
        return True
    def clear_uart(self):  # 空串口中残余的数据
        self.uart.flushInput()


    """
    内部辅助函数，用户无需使用
    """

    # 串口发送函数
    def write_data(self, data=[]):
        # global uart
        try:
            result = self.uart.write(data)  # 写数据
            # print("write_data: ", data)
            return result
        except Exception as e:
            print("---error in write_data--：", e)
            print("重启串口")
            self.uart.close()
            self.uart.open()
            result = self.uart.write(data)  # 写数据
            return result

    # 串口接收函数
    def read_data(self, num=16):  # 16个字节
        global READ_FLAG
        READ_FLAG = -1
        byte_list = []
        byte_list_head = 0
        i = 500  # 经过测试，发现正常接收16字节耗时大概为500
        while self.uart.inWaiting() == 0 and i > 0:  # To do:
            i -= 1
            time.sleep(0.001)
            if i == 0:
                print("数据为空，接收数据超时，程序退出")
                sys.exit()
        while byte_list_head != 170:
            byte_list_head = self.uart.read(1)[0]
        byte_list.append(byte_list_head)
        while self.uart.inWaiting() > 0 and len(byte_list) < num:
            byte_list.append(list(self.uart.read(1))[0])
        if len(byte_list) == num:
            READ_FLAG = 1
            return byte_list
        elif len(byte_list) > num:
            byte_list_length = len(byte_list)
            READ_FLAG = 1
            for i in range(byte_list_length - num):
                byte_list.pop(byte_list_length - i - 1)
            return byte_list
        else:
            print("Received data error in read_data(): " + str(byte_list))
            READ_FLAG = -1
            byte_list = []
            return

    def read_data_state(self, n):  # 16*n个字节
        global READ_FLAG
        READ_FLAG = -1
        byte_list = []
        byte_list_head = 0
        i = 500  # 经过测试，发现正常接收16字节耗时大概为500
        while self.uart.inWaiting() == 0 and i > 0:  # To do:
            i -= 1
            time.sleep(0.001)
            if i == 0:
                print("数据为空，接收数据超时，程序退出")
                sys.exit()
        while byte_list_head != 170:
            byte_list_head = self.uart.read(1)[0]
        byte_list.append(byte_list_head)
        while self.uart.inWaiting() > 0 or len(byte_list) < (n * 16):
            byte_list.append(list(self.uart.read(1))[0])
        if len(byte_list) == (n * 16):
            READ_FLAG = 1
            return byte_list
        elif len(byte_list) > (n * 16):
            byte_list_length = len(byte_list)
            READ_FLAG = 1
            for i in range(byte_list_length - (n * 16)):
                byte_list.pop(byte_list_length - i - 1)
            return byte_list
        else:
            print("Received data error in read_data_state(): " + str(byte_list))
            READ_FLAG = -1
            return

    def read_data_state2(self, n):  # 16*n个字节
        global READ_FLAG
        READ_FLAG = -1
        byte_list = []
        byte_list_head = 0
        i = 500
        while self.uart.inWaiting() == 0 and i > 0:
            time.sleep(0.001)
            i -= 1
            if i < 495:
                order_num = 0x24
                data = self.format_data([order_num], 'u32', 'encode')
                self.send_command(id_num=0, cmd=0x08, data=data, rtr=0)
                i = 500
        while byte_list_head != 170:
            byte_list_head = self.uart.read(1)[0]
        byte_list.append(byte_list_head)
        while self.uart.inWaiting() > 0 or len(byte_list) < (n * 16):
            if self.uart.inWaiting() == 0 and len(byte_list) < (n * 16):
                break
            byte_list.append(list(self.uart.read(1))[0])
        if len(byte_list) == (n * 16):
            READ_FLAG = 1
            return byte_list
        elif len(byte_list) > (n * 16):
            byte_list_length = len(byte_list)
            READ_FLAG = 1
            for i in range(byte_list_length - (n * 16)):
                byte_list.pop(byte_list_length - i - 1)
            return byte_list
        else:
            print("Received data error in read_data_state(): " + str(byte_list))
            READ_FLAG = -1
            return None

    def read_data_id(self):  # 16*n个字节
        byte_list = []
        id_list = []
        i = 500  # 经过测试，发现正常接收16字节耗时大概为500
        while self.uart.inWaiting() == 0 and i > 0:  # To do:
            i -= 1
            time.sleep(0.001)
            if i == 0:
                print("数据为空，接收数据超时，程序退出")
                sys.exit()
        while self.uart.inWaiting() > 0:
            byte_list.append(list(self.uart.read(1))[0])
        if len(byte_list) % 16 == 0 and len(byte_list) >= 16:
            for i in range(len(byte_list) // 16):
                jdata = byte_list[i * 16: (i + 1) * 16]
                cdata = self.uart_to_can_ID(data=jdata)
                id_list.append((cdata[1] * 256 + cdata[2] - 1) >> 5)
            return id_list
        else:
            print("Received data error in read_data_id(): " + str(byte_list))
            return

    # USB转CAN模块包模式：CAN报文->串行帧
    def can_to_uart(self, data=[], rtr=0):
        udata = [0xAA, 0, 0, 0x08, 0, 0, 0, 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                 0x00]  # 其中0x08代表CAN读取字头中的DLC（报文长度）
        # udata[1]对应CAN字头中的IDE，udata[2]对应CAN字头中的RTR，udata[3]对应CAN字头中的DLC，udata[4~7]对应id
        # udata 根据USB转CAN模块串口包模式定义
        if len(data) == 11 and data[0] == 0x08:  # 0x08 为预设的一个校验字节
            if rtr == 1:
                udata[2] = 0x01  # rtr 标志位
            for i in range(10):
                udata[6 + i] = data[i + 1]
            return udata
        else:
            return []

    # USB转CAN模块包模式：串行帧->CAN报文
    def uart_to_can(self, data=[]):
        global READ_FLAG
        cdata = [0x08, 0, 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        # if len(data) == 16 and data[3] == 0x08:
        if (len(data) % 16 == 0) and data[3] == 0x08:
            for i in range(10):
                cdata[1 + i] = data[i + 6]
            return cdata
        else:
            READ_FLAG = -1
            return []

    # USB转CAN模块包模式：串行帧->CAN报文，同时计算 CAN 节点 ID
    def uart_to_can_ID(self, data=[]):
        global READ_FLAG
        cdata = [0x08, 0, 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        if len(data) == 16 and data[3] == 0x08:
            for i in range(10):
                cdata[1 + i] = data[i + 6]
            return cdata
        else:
            READ_FLAG = -1
            return [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]

    # CAN发送函数
    def send_command(self, id_num=0, cmd=0x09, data=[], rtr=0):
        cdata = [0x08, 0, 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        id_list = (id_num << 5) + cmd  # id号右移5位 + cmd，此处根据odrive CAN node_id号与cmd_id的角度关系决定
        cdata[1] = id_list >> 8
        cdata[2] = id_list & 0xFF
        for i in range(8):
            cdata[3 + i] = data[i]  # data[]中包含命令的内容，如角度、转速、转矩等
        # print("cdata: ", cdata)
        data = self.can_to_uart(data=cdata, rtr=rtr)
        # print("self.send_command: ", data)
        self.write_data(data)
        self.set_angles_mode_1_flag = 0
        self.estop_flag = 0
        # write_data(data=can_to_uart(data=cdata, rtr=rtr))

    # CAN接收函数
    def receive_data(self):
        udata = self.read_data(16)  # can 报文一共16个字节
        if READ_FLAG == 1:
            cdata = self.uart_to_can(data=udata)
            return cdata[3:]

    def format_data(self, data=[], format="f f", type='decode'):
        # print("self.format_data data:", data)
        # print("self.format_data format:", format)
        format_list = format.split()  # 将数据类型转换成列表
        rdata = []
        if type == 'decode' and len(data) == 8:
            p = 0
            for f in format_list:
                s_f = []
                if f == 'f':
                    s_f = [4, 'f']
                elif f == 'u16':
                    s_f = [2, 'H']
                elif f == 's16':
                    s_f = [2, 'h']
                elif f == 'u32':
                    s_f = [4, 'I']
                elif f == 's32':
                    s_f = [4, 'i']
                ba = bytearray()
                if len(s_f) == 2:
                    for i in range(s_f[0]):
                        ba.append(data[p])
                        p = p + 1
                    rdata.append(struct.unpack(s_f[1], ba)[0])
                else:
                    print('unkown format in self.format_data(): ' + f)
                    return []
            return rdata
        elif type == 'encode' and len(format_list) == len(data):  # 判断数据格式类型数量与数据数量相同
            for i in range(len(format_list)):
                f = format_list[i]
                s_f = []
                if f == 'f':
                    s_f = [4, 'f']  # f 代表float，占4个字节, angle 是浮点数
                elif f == 'u16':
                    s_f = [2, 'H']  # H 代表unsigned short，占2个字节
                elif f == 's16':
                    s_f = [2, 'h']  # h 代表short，占2个字节，speed 和 加转速是短整型
                elif f == 'u32':
                    s_f = [4, 'I']  # I 代表 unsigned int, 占4个字节
                elif f == 's32':
                    s_f = [4, 'i']  # i 代表int，占4个字节
                if len(s_f) == 2:
                    bs = struct.pack(s_f[1], data[i])  # 将数据转换成二进制数
                    for j in range(s_f[0]):
                        rdata.append(bs[j])  # 将数据按字节装进 rdara
                        # print(i, ":", rdata)
                else:
                    print('unkown format in self.format_data(): ' + f)
                    return []
            if len(rdata) < 8:
                for i in range(8 - len(rdata)):
                    rdata.append(0x00)
            return rdata

    # 预设角度
    def preset_angle(self, id_num=1, angle=0, t=0, param=0, mode=0):
        """单个一体化关节角度预设函数。

        预设指定一体化关节编号的一体化关节的目标角度，之后需要用mt或mq指令启动转动。

        Args:
            id_num: 需要设置的一体化关节ID编号,如果不知道当前一体化关节ID，可以用0广播，如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。
            angle: 一体化关节角度（-360~360）*n，支持大角度转动
            t: mode=0,无作用，直接给0即可; mode=1, 运动时间（s）; mode =2, 前馈转速（r/min)
            param: mode=0,角度输入滤波带宽（<300），mode=1,启动和停止阶段加转速（(r/min)/s）; mode =2, 前馈力矩（Nm)
            mode: 角度控制模式选择，一体化关节支持三种角度控制模式，
                  mode = 0: 多个一体化关节轨迹跟踪模式，用于一般绕轴运动，特别适合多个轨迹点输入，角度输入带宽参数需设置为指令发送频率的一半。
                  mode = 1: 多个一体化关节梯形轨迹模式，此时speed用运动时间t（s）表示，param为目标加转速（(r/min)/s）。
                  mode = 2: 前馈控制模式，这种模式下的t为前馈转速，param为前馈力矩。前馈控制在原有PID控制基础上加入转速和力矩前馈，提高系统的响应特性和减少静态误差。

        Note: 在mode=1,梯形轨迹模式中，speed和accel都需要大于0.如果speed=0会导致一体化关节报motor error 并退出闭环控制模式，所以在这种模式下如果speed=0,会被用0.01代替。
              另外如果这种模式下accel=0，一体化关节以最快转速运动到angle,speed参数不再其作用。

        Returns:
            True: 运行正常
            False: 运行过程中出现异常

        Raises:
            "---error in set_angle---"

        """
        factor = 0.01
        if mode == 0:
            f_angle = angle
            s16_time = int(abs(t) / factor)
            if param > 300:
                print("input_filter_width = " + str(param) + ", which is too big and resized to 300")
                param = 300
            s16_width = int(abs(param / factor))
            data = self.format_data([f_angle, s16_time, s16_width], 'f s16 s16', 'encode')
            self.send_command(id_num=id_num, cmd=0x0C, data=data, rtr=0)
        elif mode == 1:
            f_angle = angle
            s16_time = int(abs(t) / factor)  # 运行时间
            s16_accel = int((abs(param)) / factor)
            data = self.format_data([f_angle, s16_time, s16_accel], 'f s16 s16', 'encode')
            self.send_command(id_num=id_num, cmd=0x0C, data=data, rtr=0)  # 将角度、时间、加转速发给关节
        elif mode == 2:
            f_angle = angle
            s16_speed_ff = int((t) / factor)
            s16_torque_ff = int((param) / factor)
            data = self.format_data([f_angle, s16_speed_ff, s16_torque_ff], 'f s16 s16', 'encode')
            self.send_command(id_num=id_num, cmd=0x0C, data=data, rtr=0)

    # 预设转速
    def preset_speed(self, id_num=0, speed=10, param=0, mode=1):
        """单个一体化关节转速预设函数。

        预设指定一体化关节编号的一体化关节的目标转速，之后需要用mv指令启动转动。

        Args:
            id_num: 需要设置的一体化关节ID编号,如果不知道当前一体化关节ID，可以用0广播，如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。
            speed:  目标转速（r/min）
            param:  mode=0, 前馈力矩（Nm); mode!=0,或目标加转速（(r/min)/s）
            mode:   控制模式选择
                    mode=0, 转速前馈控制模式，一体化关节将目标转速直接设为speed
                    mode!=0,转速爬升控制模式，一体化关节将按照目标加转速变化到speed。

        Note:
            在转速爬升模式下，如果目标加转速设置为0，则一体化关节转速将保持当前值不变。

        Returns:
            True: 运行正常
            False: 运行过程中出现异常

        Raises:
            "---error in set_speed---"

        """
        factor = 0.01
        try:
            f_speed = speed
            if mode == 0:
                s16_torque = int((param) / factor)
                if f_speed == 0:
                    s16_torque = 0
                s16_input_mode = int(1 / factor)
                data = self.format_data([f_speed, s16_torque, s16_input_mode], 'f s16 s16', 'encode')
            else:
                s16_ramp_rate = int((param) / factor)
                s16_input_mode = int(2 / factor)
                data = self.format_data([f_speed, s16_ramp_rate, s16_input_mode], 'f s16 s16', 'encode')
            self.send_command(id_num=id_num, cmd=0x0C, data=data, rtr=0)
        except Exception as e:
            print("---error in preset_speed---：", e)
            return False
        return True

    # 预设力矩
    def preset_torque(self, id_num=0, torque=0.1, param=0, mode=1):
        """单个一体化关节力矩预设函数。

        预设指定一体化关节编号的一体化关节目标力矩（Nm）

        Args:
            id_num: 需要设置的一体化关节ID编号,如果不知道当前一体化关节ID，可以用0广播，如果总线上有多个一体化关节，则多个一体化关节都会执行该操作。
            torque: 一体化关节输出（Nm)
            param: mode=0,改参数无意义；mode!=0,力矩上升转速率（Nm/s）
            mode:   控制模式选择
                    mode=0, 力矩直接控制模式，一体化关节将目标力矩直接设为torque
                    mode!=0,力矩爬升控制模式，一体化关节将按照力矩上升速率（Nm/s）变化到torque。

        Note;
            如果一体化关节转速超过您设置的 speed_limit ，一体化关节输出的力矩将会减小。
            可以设置 dr.controller.config.enable_speed_limit = False 来禁止力矩减小。
            另外在力矩爬升控制模式下，如果点击力矩上升速率为0，则点击力矩将在当前值保持不变。

        Returns:
            True: 运行正常
            False: 运行过程中出现异常

        Raises:
            "---error in set_torque---"

        """
        factor = 0.01
        try:
            f_torque = torque
            if mode == 0:
                s16_input_mode = int(1 / factor)
                s16_ramp_rate = 0
            else:
                s16_input_mode = int(6 / factor)
                s16_ramp_rate = int((param) / factor)
            data = self.format_data([f_torque, s16_ramp_rate, s16_input_mode], 'f s16 s16', 'encode')
            self.send_command(id_num=id_num, cmd=0x0C, data=data, rtr=0)
        except Exception as e:
            print("---error in preset_torque---：", e)
            return False
        return True

    def output_cpr(self, id_num=0):

        try:
            order_num = 0x22
            data = self.format_data([order_num], 'u32', 'encode')  # # 将 order_num 转换成占4字节的数据，
            self.send_command(id_num=id_num, cmd=0x08, data=data, rtr=0)  # 需要用标准帧（数据帧）进行发送，不能用远程帧
        except Exception as e:
            print("---error in set_toque_adaptive---：", e)
            return False
        return True

    def clear_uart(self):  # 空串口中残余的数据
        self.uart.flushInput()

    def build_packet(self, data7, speed, para, mode):
        """
        构造完整的数据包。
        参数：
            data7: 7个float列表
            speed: float
            para: float
            mode: int (0~255)
        返回：
            完整的40字节数据包
        """

        packet = bytearray()

        # 第0位：0x92
        packet.append(0x92)

        # 第1~28位：7个 float（4 字节）

        packet += b''.join(struct.pack('<f', f) for f in data7)

        # 第29~32位：speed float
        packet += struct.pack('<f', speed)

        # 第33~36位：para float
        packet += struct.pack('<f', para)

        # 第37位：mode int
        packet.append(mode & 0xFF)

        # 第38位：0x30
        packet.append(0x30)

        # 第39位：前39字节的十六进制和（低 8 位）
        checksum = sum(packet) & 0xFF
        packet.append(checksum)

        return packet
    def build_packet_v2(self, data7, speed, para, mode,
                        WaistF, WaistA, WaistType, ControlMode):
        """
        构造扩展后的49字节数据包。

        参数：
            data7:       list[float] 长度为7
            speed:       float
            para:        float
            mode:        int (0~65535)
            WaistF:      float (将乘1000后转为uint16)
            WaistA:      float (将乘1000后转为uint16)
            WaistType:   int (uint16)
            ControlMode: int (uint16)

        返回：
            bytearray (49字节)
        """
        packet = bytearray()

        # 第0位：包头
        packet.append(0x92)

        # 第1~28位：7个 float（小端）
        packet += b''.join(struct.pack('<f', f) for f in data7)

        # 第29~32位：speed float
        packet += struct.pack('<f', speed)

        # 第33~36位：para float
        packet += struct.pack('<f', para)

        # 第37~38位：mode uint16
        packet += struct.pack('<H', mode & 0xFFFF)

        # 第39~40位：WaistF * 1000 -> uint16
        waist_f_int = int(WaistF * 1000) & 0xFFFF
        packet += struct.pack('<H', waist_f_int)

        # 第41~42位：WaistA * 1000 -> uint16
        waist_a_int = int(WaistA * 1000) & 0xFFFF
        packet += struct.pack('<H', waist_a_int)

        # 第43~44位：WaistType uint16
        packet += struct.pack('<H', WaistType & 0xFFFF)

        # 第45~46位：ControlMode uint16
        packet += struct.pack('<H', ControlMode & 0xFFFF)

        # 第47位：固定包尾
        packet.append(0x30)

        # 第48位：校验和（前48字节之和的低8位）
        checksum = sum(packet) & 0xFF
        packet.append(checksum)

        return packet

    def Protocol_Set_Angles(self, id_list=[1, 2, 3], angle_list=[0, 0, 0], speed=10, param=10, mode=1):
        """多个一体化关节控制函数。

        控制多个编号的一体化关节按照一定转速转动到指定的角度。

        Args：
               id_list：一体化关节 ID 编号组成的列表。
            angle_list：目标角度（°）组成的列表。
                 speed：指定转速（r/min），由 mode 取值决定：
                        mode=1，多个一体化关节中转速最大者的转速；
                        mode=0/2，或前馈转速。
                 param：运动参数，由 mode 取值决定：
                        mode=0，角度输入滤波带宽（<300）；
                        mode=1，启动和停止阶段加转速（r/min/s）；
                        mode=2，前馈力矩（Nm)。
                  mode：控制模式选择，一体化关节支持三种角度控制模式，由 mode 取值决定：
                        mode=0：多个一体化关节轨迹跟踪模式，适合多个轨迹点输入后进行平滑控制，角度输入带宽参数需设置为指令发送频率的一半；
                        mode=1：多个一体化关节梯形轨迹模式，此时 speed 为这些一体化关节的最快转速（r/min），param为目标加转速（r/min/s）；
                        mode=2：前馈控制模式，这种模式下的 speed 和 torque 分别为前馈控制量。前馈控制在原有PID控制基础上加入转速和力矩前馈，
                                提高系统的响应特性和减少静态误差。

        Returns：
             True：运行正常。
            False：出现异常。

        Raises：
            如果 id_list 和 angle_list 长度不一致时会提示。
            “---error in set_angles---”

        """
        try:
            if len(id_list) == 7:
                packet = self.build_packet(angle_list, speed, param, mode)
                self.MainSer.write(packet)
            else:
                print("id_list 和 angle_list 长度不一致")
        except Exception as e:
            print("---error in set_angles---：", e)
            return False
        return True
    def Protocol_Set_Angles_with_Waist(self, id_list=[1, 2, 3], angle_list=[0, 0, 0], speed=10, param=10, mode=1,
                                       WaistF = 1, WaistA = 0, WaistType = 0, ControlMode = 0):
        """多个一体化关节控制函数。

        控制多个编号的一体化关节按照一定转速转动到指定的角度。

        Args：
               id_list：一体化关节 ID 编号组成的列表。
            angle_list：目标角度（°）组成的列表。
                 speed：指定转速（r/min），由 mode 取值决定：
                        mode=1，多个一体化关节中转速最大者的转速；
                        mode=0/2，或前馈转速。
                 param：运动参数，由 mode 取值决定：
                        mode=0，角度输入滤波带宽（<300）；
                        mode=1，启动和停止阶段加转速（r/min/s）；
                        mode=2，前馈力矩（Nm)。
                mode：   控制模式选择，一体化关节支持三种角度控制模式，由 mode 取值决定：
                        mode=0：多个一体化关节轨迹跟踪模式，适合多个轨迹点输入后进行平滑控制，角度输入带宽参数需设置为指令发送频率的一半；
                        mode=1：多个一体化关节梯形轨迹模式，此时 speed 为这些一体化关节的最快转速（r/min），param为目标加转速（r/min/s）；
                        mode=2：前馈控制模式，这种模式下的 speed 和 torque 分别为前馈控制量。前馈控制在原有PID控制基础上加入转速和力矩前馈，
                                提高系统的响应特性和减少静态误差。
                WaistF:         转手绢的频率（一秒转几圈）
                WaistA：        转手绢的幅度幅度（角度）
                WaistType：      转手绢的方式：
                            0.  不转手绢（用来过度控制模式）
                            1：正常旋转
                            2：椭圆甩手卷
                            3. 加速甩手绢
                            4. 周期加减速
                            5. 减小半径，增大速率
                ControlMode：    整体控制模式
                            0：闭环定位模式 闭环控制，不转手腕，手腕将以闭环的形式走到指定基准轴
                            1：闭环动轴模式 闭环控制除了手腕以外的其余关节，手腕闭环到达的位置将作为基准轴（要求先以 mode 0 到达闭环位置，转手腕
                                                                                    （即手腕两个输入的参数不起作用）
                            2：开环动轴模式 闭环控制除了手腕以外的其余关节，手腕的输入参数将直接传入基准轴，转手腕
                                                                                    （即基准轴没有经过闭环校准，是个开环基准）
                            3：纯开环模式 开环控制所有电机，这里输入的是所有电机的绝对位置，下位机不再进行任何计算，不自主转手腕，只做数据转发

        Returns：
             True：运行正常。
            False：出现异常。

        Raises：
            如果 id_list 和 angle_list 长度不一致时会提示。
            “---error in set_angles---”

        """
        try:
            if len(id_list) == 7:
                packet = self.build_packet_v2(angle_list, speed, param, mode, WaistF, WaistA, WaistType, ControlMode)
                self.MainSer.write(packet)
            else:
                print("id_list 和 angle_list 长度不一致")
        except Exception as e:
            print("---error in set_angles---：", e)
            return False
        return True