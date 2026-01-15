"""
CAN 协议解析器和生成器
与 TypeScript canProtocol.ts 实现相匹配，但数据结构对齐 backend/can_protocol.py
"""

import struct
from typing import Dict, Any, List

class MachineState:
    """H2 FCU 完整状态机数据结构 - 与 can_protocol.py 保持一致"""
    
    def __init__(self):
        self.connected = False
        self.last_update = 0
        
        # 系统状态
        self.status = {
            "heartbeat": 0,
            "state": 0,       # 0=关机完成, 1=关机中, 2=运行, 3=急停, 0xF=故障, 0x10=复位, 0x11=启动中
            "faultLevel": 0,  # 0=无, 1=一级, 2=二级, 3=三级
            "dcdcState": 0,   # 0=停止, 1=运行, 2=放电
            "dcdcFaultCode": 0
        }
        
        # 电堆及DCDC电源数据
        self.power = {
            "stackVoltage": 0.0,      # V
            "stackCurrent": 0.0,      # A
            "stackPower": 0.0,        # kW (计算值)
            "dcdcOutVoltage": 0.0,    # V
            "dcdcOutCurrent": 0.0,    # A
            "dcdcInVoltage": 0.0,     # V
            "dcdcInCurrent": 0.0,     # A
            "conductivity": 0.0       # S/m
        }
        
        # 氢气路传感器 (H2)
        self.h2 = {
            "highPressure": 0.0,      # kPa (氢气高压)
            "inletPressure": 0.0,     # kPa (进堆压力)
            "outletPressure": 0.0,    # kPa (出堆压力)
            "inletFlow": 0.0,         # L/min
            "inletTemp": 0.0,         # ℃
            "circulationSpeed": 0,    # rpm (循环泵)
            "separatorPressure": 0.0, # kPa (汽水分离器)
            "concentration": 0.0      # %vol (氢气浓度)
        }
        
        # 空气路传感器 (Air)
        self.air = {
            "inletPressure": 0.0,     # kPa
            "inletTemp": 0.0,         # ℃
            "outletPressure": 0.0,    # kPa
            "outletTemp": 0.0,        # ℃
            "inletFlow": 0.0,         # kg/h
            "humidity": 0.0,          # %
            "compressorSetSpeed": 0,  # rpm
            "compressorRealSpeed": 0  # rpm
        }
        
        # 冷却水路传感器 (Water)
        self.water = {
            "inletPressure": 0.0,     # kPa
            "inletTemp": 0.0,         # ℃
            "outletTemp": 0.0,        # ℃
            "auxOutletTemp": 0.0,     # ℃ (辅助散热出口)
            "auxDcdcTemp": 0.0,       # ℃
            "auxCompTemp": 0.0        # ℃
        }
        
        # 设备温度
        self.temps = {
            "dcdcTemp": 0.0           # ℃
        }
        
        # IO 执行器状态
        self.io = {
            # 开关量 (True/False)
            "h2HighValve": False,     # 氢气高压阀
            "h2HeatValve": False,     # 氢气加热阀
            "h2PurgeValve": False,    # 氢气排氢阀
            "h2Injectors": [False]*4, # 喷射阀 1-4
            "h2CircPump": False,      # 氢气循环泵
            
            "airInletThrottle": False, # 空气进气节气门
            "airOutletThrottle": False,# 空气尾排节气门
            "compressor": False,       # 空压机
            "bypassValve": False,      # 旁通阀
            "mainPump": False,         # 主散热水泵
            "mainFan": False,          # 主散热器(风扇)
            "thermostatState": 0,      # 0=关, 1=小循环, 2=大循环
            
            "waterLevelLow": False,    # 液位低 (原始1为正常，此处转为报警逻辑)
            "auxFan": False,           # 辅助散热器
            "auxPump": False,          # 辅助水泵
            "ptcHeater": False,        # PTC
            
            # 模拟量反馈
            "thermostatPosition": 0,   # %
            "airInletThrottlePos": 0,  # %
            "airOutletThrottlePos": 0, # %
            "h2PurgeCountdown": 0      # s
        }
        
        # 故障码 (8个字节)
        self.faults = {
            "codes": [0] * 8
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization with frontend compatibility"""
        # 计算衍生值
        dcf_power = round(self.power["dcdcOutVoltage"] * self.power["dcdcOutCurrent"], 2)
        dcf_efficiency = 0
        if self.power["stackPower"] > 0:
            dcf_efficiency = round((dcf_power / (self.power["stackPower"] * 1000)) * 100, 1)
        
        return {
            "connected": self.connected,
            "lastUpdate": self.last_update,
            
            # 系统状态 - 直接映射
            "status": {
                "heartbeat": self.status["heartbeat"],
                "state": self.status["state"],
                "faultLevel": self.status["faultLevel"]
            },
            
            # 电源数据 - 映射到前端期望的字段名
            "power": {
                "stackVoltage": self.power["stackVoltage"],
                "stackCurrent": self.power["stackCurrent"],
                "stackPower": self.power["stackPower"],
                "dcfOutVoltage": self.power["dcdcOutVoltage"],    # 前端期望 dcfOutVoltage
                "dcfOutCurrent": self.power["dcdcOutCurrent"],    # 前端期望 dcfOutCurrent
                "dcfPower": dcf_power,
                "dcfEfficiency": dcf_efficiency
            },
            
            # 传感器数据 - 从多个来源聚合
            "sensors": {
                "stackTemp": self.water.get("outletTemp", 25),  # 电堆温度用出水温度
                "ambientTemp": self.air.get("inletTemp", 25),   # 环境温度用进气温度
                "h2CylinderPressure": self.h2.get("highPressure", 0) / 1000.0,  # kPa -> MPa
                "h2InletPressure": self.h2.get("inletPressure", 0) / 1000.0,    # kPa -> MPa
                "h2Concentration": self.h2.get("concentration", 0)  # %vol
            },
            
            # IO状态 - 映射到前端期望的字段
            "io": {
                "h2InletValve": self.io.get("h2HighValve", False),      # 映射: h2HighValve -> h2InletValve
                "h2PurgeValve": self.io.get("h2PurgeValve", False),
                "proportionalValve": False,  # 当前协议未提供
                "heater": self.io.get("ptcHeater", False),              # 映射: ptcHeater -> heater
                "fan1": self.io.get("mainFan", False),                   # 映射: mainFan -> fan1
                "fan2": self.io.get("auxFan", False),                    # 映射: auxFan -> fan2
                "fan1Duty": self.io.get("fan1Duty", 0),                  # 映射: fan1Duty -> fan1Duty (0-100%)
                "dcfMosTemp": self.temps.get("dcdcTemp", 25),           # 映射: dcdcTemp -> dcfMosTemp
                "faultCode": self.faults["codes"][0] if self.faults["codes"] else 0
            }
        }


def parse_msg1_status(data: bytes, state: MachineState) -> None:
    """
    解析 CAN ID 0x18FF01F0 - 系统状态
    Byte 0: 心跳计数器
    Byte 1: 状态[1:0] | 故障等级[3:2]
    """
    if len(data) < 2:
        return
    
    state.status["heartbeat"] = data[0]
    state.status["state"] = data[1] & 0x03
    state.status["faultLevel"] = (data[1] >> 2) & 0x03


def parse_msg2_power(data: bytes, state: MachineState) -> None:
    """
    解析 CAN ID 0x18FF02F0 - 电源数据
    Bytes 0-1: 电堆电压 (uint16, 因子 0.01)
    Bytes 2-3: 电堆电流 (uint16, 因子 0.1)
    Bytes 4-5: DCF 输出电压 (uint16, 因子 0.01)
    Bytes 6-7: DCF 输出电流 (uint16, 因子 0.1)
    """
    if len(data) < 8:
        return
    
    stack_v = struct.unpack('<H', data[0:2])[0] * 0.01
    stack_i = struct.unpack('<H', data[2:4])[0] * 0.1
    dcf_v = struct.unpack('<H', data[4:6])[0] * 0.01
    dcf_i = struct.unpack('<H', data[6:8])[0] * 0.1
    
    stack_p = stack_v * stack_i / 1000.0 # kW
    
    state.power["stackVoltage"] = round(stack_v, 2)
    state.power["stackCurrent"] = round(stack_i, 1)
    state.power["stackPower"] = round(stack_p, 1)
    
    # 映射到 MachineState 的字段
    state.power["dcdcOutVoltage"] = round(dcf_v, 2)
    state.power["dcdcOutCurrent"] = round(dcf_i, 1)


def parse_msg3_sensors(data: bytes, state: MachineState) -> None:
    """
    解析 CAN ID 0x18FF03F0 - 传感器数据
    Bytes 0-1: 电堆温度 (uint16, 因子 0.1, 偏移 -40°C)
    Bytes 2-3: 氢脉压力/氢气瓶压力 (uint16, 因子 0.01 MPa)
    Bytes 4-5: 连氢压力/氢气进气压力 (uint16, 因子 0.01 MPa)
    Byte 6: 氢气浓度 (uint8, 因子 0.5 %vol)
    Byte 7: 预留
    """
    if len(data) < 8:
        return
    
    stack_temp = struct.unpack('>H', data[0:2])[0] * 0.1 - 40
    h2_cyl_press_mpa = struct.unpack('>H', data[2:4])[0] * 0.01
    h2_inlet_press_mpa = struct.unpack('>H', data[4:6])[0] * 0.01
    h2_conc = data[6] * 0.5  # 氢气浓度
    
    # 映射到最接近的字段
    state.water["outletTemp"] = round(stack_temp, 1)   # 电堆温度 -> 出水温度
    state.h2["highPressure"] = round(h2_cyl_press_mpa * 1000, 2)   # MPa -> kPa
    state.h2["inletPressure"] = round(h2_inlet_press_mpa * 1000, 2)# MPa -> kPa
    state.h2["concentration"] = round(h2_conc, 1)  # 氢气浓度 %vol


def parse_msg4_io(data: bytes, state: MachineState) -> None:
    """
    解析 CAN ID 0x18FF04F0 - IO 状态
    Byte 0: IO 标志位 (bit0=进气, bit1=排气, bit2=比例, bit3=加热, bit4=风扇1, bit5=风扇2)
    Byte 1: 风扇1 占空比 (%)
    Bytes 2-3: DCF MOS 温度 (int16, 因子 0.1, 偏移 -40°C)
    Bytes 4-5: 故障码 (uint16)
    """
    if len(data) < 6:
        return
    
    flags = data[0]
    fan1_duty = data[1]  # 风扇1占空比 (0-100%)
    dcf_mos_temp = struct.unpack('>H', data[2:4])[0] * 0.1 - 40
    fault_code = struct.unpack('>H', data[4:6])[0]
    
    # IO Flags Mapping
    state.io["h2HighValve"] = bool(flags & 0x01)   # 进气阀 -> 氢气高压阀 (近似)
    state.io["h2PurgeValve"] = bool(flags & 0x02)
    # 比例阀 -> 无直接映射
    state.io["ptcHeater"] = bool(flags & 0x04)     # 加热器 -> PTC 加热器
    state.io["mainFan"] = bool(flags & 0x08)       # 风扇1 -> 主风扇
    state.io["auxFan"] = bool(flags & 0x10)        # 风扇2 -> 辅助风扇
    state.io["fan1Duty"] = fan1_duty               # 风扇1占空比
    
    state.temps["dcdcTemp"] = round(dcf_mos_temp, 1)
    
    # 故障码 - 更新故障数组的第一个字节
    # MachineState 期望 8 字节数组，我们将 uint16 放入前 2 个字节
    state.faults["codes"][0] = fault_code & 0xFF
    state.faults["codes"][1] = (fault_code >> 8) & 0xFF


def generate_control_packets(control: Dict[str, Any]) -> List[tuple[int, bytes]]:
    """
    生成 ID 0x18FF10A0 的 CAN 控制报文
    返回 (arb_id, data_bytes) 元组的列表
    
    Byte 0: 模式[1:0] | 指令[4:2]
    Byte 1: 手动控制标志位
    Byte 2: 风扇1 目标转速 (%)
    Bytes 3-4: DCF 目标电压 (uint16, 因子 0.1)
    Bytes 5-6: DCF 目标电流 (uint16, 因子 0.1)
    Byte 7: 保留
    """
    data = bytearray(8)
    
    # Byte 0: 模式与指令
    # MachineState 使用 "AUTO" 字符串或 int? 
    # 原始 generate_control_packet 使用 1 代表自动。
    # 我们应该支持 server.py 发送的内容。
    # 通常 server 发送的是前端传来的数据。
    
    mode_val = control.get("mode") # 前端可能发送 "AUTO" 或 1
    if mode_val == "AUTO" or mode_val == 1:
        mode = 1
    else:
        mode = 0
        
    cmd_val = control.get("command", 0)
    # 如果指令是像 "START" 这样的字符串，如果需要则映射为 int，
    # 但根据之前的代码，看起来它是特定于该协议的。
    # 让我们假设输入匹配此协议的预期或简单的 int 映射
    # 此特定协议在先前版本中似乎期望指令为简单整数
    if isinstance(cmd_val, str):
         # 如果此协议期望简单整数，则将字符串映射为简单整数
         if cmd_val == "START": command = 1
         elif cmd_val == "STOP": command = 0 
         else: command = 0
    else:
        command = int(cmd_val) & 0x07

    data[0] = (mode & 0x03) | ((command & 0x07) << 2)
    
    # Byte 1: 手动控制标志位 (仅在手动模式下)
    if mode == 0:  # Manual Mode
        flags = 0
        if control.get("forceInletValve", False):
            flags |= 0x01
        if control.get("forcePurgeValve", False):
            flags |= 0x02
        if control.get("forceHeater", False):
            flags |= 0x04
        if control.get("forceFan1", False):
            flags |= 0x08
        if control.get("forceFan2", False):
            flags |= 0x10
        data[1] = flags
    
    # Byte 2: 风扇1 目标转速
    fan1_speed = max(0, min(100, int(control.get("fan1TargetSpeed", 50))))
    data[2] = fan1_speed
    
    # Bytes 3-4: DCF 目标电压 (因子 0.1)
    target_v = int(control.get("dcfTargetVoltage", 24.0) * 10)
    data[3] = target_v & 0xFF
    data[4] = (target_v >> 8) & 0xFF
    
    # Bytes 5-6: DCF 目标电流 (因子 0.1)
    target_i = int(control.get("dcfTargetCurrent", 5.0) * 10)
    data[5] = target_i & 0xFF
    data[6] = (target_i >> 8) & 0xFF
    
    # Byte 7: 保留
    data[7] = 0
    
    return [(0x18FF10A0, bytes(data))]


# Message parsers mapping
MESSAGE_PARSERS = {
    0x18FF01F0: parse_msg1_status,
    0x18FF02F0: parse_msg2_power,
    0x18FF03F0: parse_msg3_sensors,
    0x18FF04F0: parse_msg4_io,
}
