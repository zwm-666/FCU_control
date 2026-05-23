"""
CAN 协议解析器和生成器
基于: 80kW燃料电池测试台架手动&自动测试CAN通信协议 (2023.2)
"""

import struct
from typing import Dict, Any, List

class MachineState:
    """H2 FCU 完整状态机数据结构"""
    
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
            "separatorPressure": 0.0  # kPa (汽水分离器)
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
                "dcfVoltage": self.power["dcdcOutVoltage"],    # 映射: dcdcOutVoltage -> dcfVoltage
                "dcfCurrent": self.power["dcdcOutCurrent"],    # 映射: dcdcOutCurrent -> dcfCurrent
                "dcfPower": dcf_power,
                "dcfEfficiency": dcf_efficiency
            },
            
            # 传感器数据 - 从多个来源聚合
            "sensors": {
                "stackTemp": self.water.get("outletTemp", 25),  # 电堆温度用出水温度
                "ambientTemp": self.air.get("inletTemp", 25),   # 环境温度用进气温度
                "h2CylinderPressure": self.h2.get("highPressure", 0) / 1000.0,  # kPa -> MPa
                "h2InletPressure": self.h2.get("inletPressure", 0) / 1000.0,    # kPa -> MPa
                "h2Concentration": 0  # 当前协议未提供氢气浓度传感器
            },
            
            # IO状态 - 映射到前端期望的字段
            "io": {
                "h2InletValve": self.io.get("h2HighValve", False),      # 映射: h2HighValve -> h2InletValve
                "h2PurgeValve": self.io.get("h2PurgeValve", False),
                "proportionalValve": False,  # 当前协议未提供
                "heater": self.io.get("ptcHeater", False),              # 映射: ptcHeater -> heater
                "fan1": self.io.get("mainFan", False),                   # 映射: mainFan -> fan1
                "fan2": self.io.get("auxFan", False),                    # 映射: auxFan -> fan2
                "fan1Duty": 0,  # 当前协议未提供风扇占空比反馈
                "dcfMosTemp": self.temps.get("dcdcTemp", 25),           # 映射: dcdcTemp -> dcfMosTemp
                "faultCode": self.faults["codes"][0] if self.faults["codes"] else 0
            }
        }

# ==============================================================================
# 解析函数定义
# ==============================================================================

def parse_msg_1_main(data: bytes, state: MachineState) -> None:
    """
    ID: 0x1824A7A4 (周期 100ms) - FCU 主控 1
    Byte 1: 心跳
    Byte 2: 故障等级(bit0-1), 状态(bit2-7) (文档: 10-9bit, 16-11bit)
    Byte 3-4: 电堆电压 (LSB)
    Byte 5-6: 电堆电流 (LSB)
    Byte 7-8: 氢气循环泵转速 (LSB)
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"🔧 parse_msg_1_main被调用! 数据长度={len(data)}")
    
    if len(data) < 8:
        logger.warning(f"⚠️ 数据长度不足: {len(data)} < 8")
        return

    # Byte 1: Heartbeat
    state.status["heartbeat"] = data[0]
    
    # Byte 2: Status & Fault Level
    # 假设 bit 0-1 是故障等级, bit 2-7 是状态
    b2 = data[1]
    state.status["faultLevel"] = b2 & 0x03
    state.status["state"] = (b2 >> 2) & 0x3F
    
    # Byte 3-4: Stack Voltage (1V/bit)
    stack_v = struct.unpack('<H', data[2:4])[0] * 1.0
    
    # Byte 5-6: Stack Current (1A/bit)
    stack_i = struct.unpack('<H', data[4:6])[0] * 1.0
    
    # Byte 7-8: H2 Pump Speed (1rpm/bit)
    pump_speed = struct.unpack('<H', data[6:8])[0] * 1.0

    state.power["stackVoltage"] = stack_v
    state.power["stackCurrent"] = stack_i
    state.power["stackPower"] = round(stack_v * stack_i / 1000.0, 2) # kW
    state.h2["circulationSpeed"] = int(pump_speed)
    
    # 无条件打印解析结果
    logger.info(f"✅ 解析完成: 心跳={state.status['heartbeat']}, 状态={state.status['state']}, "
               f"电压={stack_v}V, 电流={stack_i}A, connected={state.connected}")


def parse_msg_2_h2(data: bytes, state: MachineState) -> None:
    """
    ID: 0x1825A7A4 (周期 200ms) - FCU 主控 2 (氢气路)
    Byte 1-2: 氢气高压 (LSB, 1kPa)
    Byte 3: 进堆氢气压力 (1kPa)
    Byte 4-5: 进堆氢气流量 (LSB, L/min)
    Byte 6: 出堆氢气压力 (1kPa)
    Byte 7: 进堆氢气温度 (1C, off -40)
    """
    if len(data) < 7: return
    
    state.h2["highPressure"] = struct.unpack('<H', data[0:2])[0]
    state.h2["inletPressure"] = data[2]
    state.h2["inletFlow"] = struct.unpack('<H', data[3:5])[0]
    state.h2["outletPressure"] = data[5]
    state.h2["inletTemp"] = data[6] - 40


def parse_msg_3_air(data: bytes, state: MachineState) -> None:
    """
    ID: 0x1826A7A4 (周期 200ms) - FCU 主控 3 (空气路)
    Byte 1: 进堆空气压力 (1kPa)
    Byte 2: 进堆空气温度 (1C, off -40)
    Byte 3: 出堆空气压力 (1kPa)
    Byte 4: 出堆空气温度 (1C, off -40)
    Byte 5-6: 进堆空气流量 (LSB, kg/h)
    Byte 7: 出堆空气相对湿度 (1%)
    Byte 8: 节温器阀芯位置反馈 (1%)
    """
    if len(data) < 8: return

    state.air["inletPressure"] = data[0]
    state.air["inletTemp"] = data[1] - 40
    state.air["outletPressure"] = data[2]
    state.air["outletTemp"] = data[3] - 40
    state.air["inletFlow"] = struct.unpack('<H', data[4:6])[0]
    state.air["humidity"] = data[6]
    state.io["thermostatPosition"] = data[7]


def parse_msg_4_water(data: bytes, state: MachineState) -> None:
    """
    ID: 0x1827A7A4 (周期 200ms) - FCU 主控 4 (水路 & 空压机)
    Byte 1: 进堆循环水压力 (1kPa)
    Byte 2: 进堆循环水温度 (1C, off -40)
    Byte 3: 出堆循环水温度 (1C, off -40)
    Byte 4: 电导率 (0.1 S/m)
    Byte 5-6: 空压机给定转速 (LSB, 1rpm)
    Byte 7-8: 空压机实际转速 (LSB, 1rpm)
    """
    if len(data) < 8: return

    state.water["inletPressure"] = data[0]
    state.water["inletTemp"] = data[1] - 40
    state.water["outletTemp"] = data[2] - 40
    state.power["conductivity"] = round(data[3] * 0.1, 2)
    state.air["compressorSetSpeed"] = struct.unpack('<H', data[4:6])[0]
    state.air["compressorRealSpeed"] = struct.unpack('<H', data[6:8])[0]


def parse_msg_5_io(data: bytes, state: MachineState) -> None:
    """
    ID: 0x1828A7A4 (周期 200ms) - FCU 主控 5 (IO状态)
    Byte 1: H2阀状态 (bit0-7)
    Byte 2: 空气阀/泵状态 (bit0-7)
    Byte 3: 辅助散热状态 (bit0-3...)
    Byte 4: 排氢倒计时
    Byte 5: 进气节气门反馈
    Byte 6: 尾排节气门反馈
    Byte 7-8: 给定节气门开度 (此处不解析给定值，仅关注反馈)
    """
    if len(data) < 6: return

    # Byte 1
    b1 = data[0]
    state.io["h2HighValve"] = bool(b1 & 0x01)
    state.io["h2HeatValve"] = bool(b1 & 0x02)
    state.io["h2PurgeValve"] = bool(b1 & 0x04)
    state.io["h2Injectors"][0] = bool(b1 & 0x08)
    state.io["h2Injectors"][1] = bool(b1 & 0x10)
    state.io["h2Injectors"][2] = bool(b1 & 0x20)
    state.io["h2Injectors"][3] = bool(b1 & 0x40)
    state.io["h2CircPump"] = bool(b1 & 0x80)

    # Byte 2
    b2 = data[1]
    state.io["airInletThrottle"] = bool(b2 & 0x01)
    state.io["airOutletThrottle"] = bool(b2 & 0x02)
    state.io["compressor"] = bool(b2 & 0x04)
    state.io["bypassValve"] = bool(b2 & 0x08)
    state.io["mainPump"] = bool(b2 & 0x10)
    state.io["mainFan"] = bool(b2 & 0x20)
    # 节温器状态 (Bits 6-7): 00关闭, 01小, 10大
    therm_bits = (b2 >> 6) & 0x03
    state.io["thermostatState"] = therm_bits

    # Byte 3
    b3 = data[2]
    # Bit 0: 液位 (0低-红, 1正常-绿) -> 转换为报警逻辑 True=Warning
    state.io["waterLevelLow"] = not bool(b3 & 0x01)
    state.io["auxFan"] = bool(b3 & 0x02)
    state.io["auxPump"] = bool(b3 & 0x04)
    state.io["ptcHeater"] = bool(b3 & 0x08)

    # Byte 4-6
    state.io["h2PurgeCountdown"] = data[3]
    state.io["airInletThrottlePos"] = data[4]
    state.io["airOutletThrottlePos"] = data[5]


def parse_msg_6_aux(data: bytes, state: MachineState) -> None:
    """
    ID: 0x1829A7A4 (周期 200ms) - FCU 主控 6 (辅助温度 & DCDC状态)
    Byte 1: 辅助散热出口温度
    Byte 2: 辅助 DCDC 温度
    Byte 3: 辅助空压机温度
    Byte 4: 汽水分离器压力 (1kPa)
    Byte 5: DCDC 运行温度
    Byte 6: DCDC 状态
    Byte 7: DCDC 故障码
    """
    if len(data) < 7: return

    state.water["auxOutletTemp"] = data[0] - 40
    state.water["auxDcdcTemp"] = data[1] - 40
    state.water["auxCompTemp"] = data[2] - 40
    state.h2["separatorPressure"] = data[3]
    state.temps["dcdcTemp"] = data[4] - 40
    state.status["dcdcState"] = data[5]
    state.status["dcdcFaultCode"] = data[6]


def parse_msg_7_dcdc_power(data: bytes, state: MachineState) -> None:
    """
    ID: 0x1831A7A4 (周期 400ms) - FCU 主控 7 (DCDC 电力)
    注意：文档说明 "Byte 1 高8字节", "Byte 2 低8字节"，暗示大端模式 (Big-Endian)
    Byte 1-2: 输出电压 (0.1V)
    Byte 3-4: 输出电流 (0.1A)
    Byte 5-6: 输入电压 (0.1V)
    Byte 7-8: 输入电流 (0.1A)
    """
    if len(data) < 8: return

    # 使用 Big-Endian (>) 解析
    state.power["dcdcOutVoltage"] = struct.unpack('>H', data[0:2])[0] * 0.1
    state.power["dcdcOutCurrent"] = struct.unpack('>H', data[2:4])[0] * 0.1
    state.power["dcdcInVoltage"] = struct.unpack('>H', data[4:6])[0] * 0.1
    state.power["dcdcInCurrent"] = struct.unpack('>H', data[6:8])[0] * 0.1


def parse_msg_8_faults(data: bytes, state: MachineState) -> None:
    """
    ID: 0x1830A7A4 (周期 200ms) - FCU 主控 8 (故障码)
    Byte 1-8: Fault Code 1 - 8
    """
    if len(data) < 8: return
    
    # 直接存储原始字节，由前端或上层逻辑解析具体含义
    for i in range(8):
        state.faults["codes"][i] = data[i]


# 消息解析器映射 (ID -> Function)
MESSAGE_PARSERS = {
    0x1824A7A4: parse_msg_1_main,
    0x1825A7A4: parse_msg_2_h2,
    0x1826A7A4: parse_msg_3_air,
    0x1827A7A4: parse_msg_4_water,
    0x1828A7A4: parse_msg_5_io,
    0x1829A7A4: parse_msg_6_aux,
    0x1831A7A4: parse_msg_7_dcdc_power,
    0x1830A7A4: parse_msg_8_faults
}

def generate_control_packets(control: Dict[str, Any]) -> List[tuple[int, bytes]]:
    """
    生成所有上位机控制报文 (基于 2023.01.10 协议)
    返回格式: [(ID, bytes), (ID, bytes), ...]
    """
    packets = []

    # ==========================================================================
    # 1. 报文 1: 系统控制 & 开关量 (ID: 0x18FF0B27)
    # ==========================================================================
    data_27 = bytearray(8)
    
    # --- Byte 1-3: 手动开关量 (Bit flags) ---
    # 仅在手动模式下有效，这里根据 control['io'] 状态填充
    # Byte 1
    if control.get("h2HighValve", False): data_27[0] |= 0x01
    if control.get("h2PurgeValve", False): data_27[0] |= 0x02
    if control.get("h2HeatValve", False): data_27[0] |= 0x04
    # ... (喷射阀 1-4 略，可按需补充 0x08, 0x10, 0x20, 0x40)
    if control.get("airInletThrottle", False): data_27[0] |= 0x80 # Bit 8
    
    # Byte 2
    if control.get("airOutletThrottle", False): data_27[1] |= 0x01
    if control.get("bypassValve", False): data_27[1] |= 0x02
    if control.get("auxFan", False): data_27[1] |= 0x04
    if control.get("auxPump", False): data_27[1] |= 0x08
    if control.get("ptcHeater", False): data_27[1] |= 0x10
    if control.get("h2CircPump", False): data_27[1] |= 0x20
    if control.get("compressor", False): data_27[1] |= 0x40
    if control.get("mainPump", False): data_27[1] |= 0x80

    # Byte 3
    # ... (主散热风扇 1-6, 节温器等标志位)
    if control.get("dcdcPrecharge", False): data_27[2] |= 0x80 # Bit 24

    # --- Byte 5-6: 目标电流 (Little-Endian) ---
    # 0.1A/bit
    target_current = int(control.get("stackTargetCurrent", 0) * 10)
    struct.pack_into('<H', data_27, 4, target_current)

    # --- Byte 7: 模式与指令 (核心控制) ---
    # Bit 56-55: 工作模式 (00=手动, 11=自动)
    mode_bits = 0x03 if control.get("mode") == "AUTO" else 0x00
    
    # Bit 54-53: 状态指令 (00=关机, 11=启动, 01=复位, 10=急停)
    cmd_str = control.get("command", "NONE")
    cmd_bits = 0x00 # 默认为关机/无操作
    if cmd_str == "START":
        cmd_bits = 0x03 # Binary 11
    elif cmd_str == "RESET":
        cmd_bits = 0x01 # Binary 01
    elif cmd_str == "EMERGENCY_STOP":
        cmd_bits = 0x02 # Binary 10
    elif cmd_str == "STOP":
        cmd_bits = 0x00 # Binary 00
    
    # Bit 50-49: DCDC控制 (0=停止, 1=启动, 2=放电)
    dcdc_bits = control.get("dcdcCommand", 0) & 0x03

    # 组合 Byte 7
    # 模式在最高2位 (bit 7-6 of the byte, corresponding to 56-55 in protocol)
    # 注意：协议说是 56-55 bit，相对于 Byte 7 来说是 bit 7-6 (从0开始数)
    byte7_val = (mode_bits << 6) | (cmd_bits << 4) | (dcdc_bits << 0) # DCDC位置需确认，协议说是50-49，即Byte7的bit 1-2
    # 修正位移：
    # Byte7: [7:6]=Mode, [5:4]=Cmd, [3:2]=Reserved, [1:0]=DCDC
    byte7_val = (mode_bits << 6) | (cmd_bits << 4) | (dcdc_bits << 1) # 假设DCDC是bit 1-2
    data_27[6] = byte7_val

    packets.append((0x18FF0B27, bytes(data_27)))

    # ==========================================================================
    # 2. 报文 2: 执行器设定值 (ID: 0x18FF0B28)
    # ==========================================================================
    data_28 = bytearray(8)
    
    # Byte 1: 进气节气门 (0-100%)
    data_28[0] = int(control.get("airInletThrottlePos", 0))
    # Byte 2: 尾排节气门
    data_28[1] = int(control.get("airOutletThrottlePos", 0))
    
    # Byte 3-4: 空压机转速 (Little-Endian)
    comp_speed = int(control.get("compressorTargetSpeed", 0))
    struct.pack_into('<H', data_28, 2, comp_speed)
    
    # Byte 5: 水泵转速 (0-100%)
    data_28[4] = int(control.get("mainPumpSpeed", 0))
    # Byte 6: 节温器 (7-92%)
    data_28[5] = int(control.get("thermostatPos", 0))
    
    # Byte 7-8: 氢循泵转速 (Little-Endian)
    h2_pump_speed = int(control.get("h2PumpTargetSpeed", 0))
    struct.pack_into('<H', data_28, 6, h2_pump_speed)

    packets.append((0x18FF0B28, bytes(data_28)))

    # ==========================================================================
    # 3. 报文 6: DCDC参数 (ID: 0x18FF0B32)
    # ==========================================================================
    data_32 = bytearray(8)
    
    # Byte 4-5: 输出电压 (Big-Endian per doc!)
    dcdc_volt = int(control.get("dcdcTargetVoltage", 0) * 10)
    struct.pack_into('>H', data_32, 3, dcdc_volt) # pack into index 3,4
    
    # Byte 6-7: 输入限流 (Big-Endian per doc!)
    dcdc_limit = int(control.get("dcdcInputLimit", 0) * 10)
    struct.pack_into('>H', data_32, 5, dcdc_limit) # pack into index 5,6

    packets.append((0x18FF0B32, bytes(data_32)))

    return packets