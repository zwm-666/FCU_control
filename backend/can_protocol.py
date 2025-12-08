"""
CAN 协议解析器和生成器
与 TypeScript canProtocol.ts 实现相匹配
"""

import struct
from typing import Dict, Any, List


class MachineState:
    """Complete state of the H2 FCU machine"""
    
    def __init__(self):
        self.connected = False
        self.last_update = 0
        self.status = {
            "heartbeat": 0,
            "state": 0,  # 0=OFF, 1=START, 2=RUN, 3=FAULT
            "faultLevel": 0  # 0=NORMAL, 1=WARNING, 2=SEVERE, 3=EMERGENCY
        }
        self.power = {
            "stackVoltage": 0.0,
            "stackCurrent": 0.0,
            "stackPower": 0.0,
            "dcfOutVoltage": 0.0,
            "dcfOutCurrent": 0.0,
            "dcfPower": 0.0,
            "dcfEfficiency": 0.0
        }
        self.sensors = {
            "stackTemp": 0.0,
            "ambientTemp": 0.0,
            "h2CylinderPressure": 0.0,
            "h2InletPressure": 0.0,
            "h2Concentration": 0.0
        }
        self.io = {
            "h2InletValve": False,
            "h2PurgeValve": False,
            "proportionalValve": False,
            "heater": False,
            "fan1": False,
            "fan2": False,
            "fan1Duty": 0,
            "dcfMosTemp": 0.0,
            "faultCode": 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "connected": self.connected,
            "lastUpdate": self.last_update,
            "status": self.status,
            "power": self.power,
            "sensors": self.sensors,
            "io": self.io
        }


def parse_msg1_status(data: bytes, state: MachineState) -> None:
    """
    Parse CAN ID 0x18FF01F0 - System Status
    Byte 0: Heartbeat counter
    Byte 1: State[1:0] | FaultLevel[3:2]
    """
    if len(data) < 2:
        return
    
    state.status["heartbeat"] = data[0]
    state.status["state"] = data[1] & 0x03
    state.status["faultLevel"] = (data[1] >> 2) & 0x03


def parse_msg2_power(data: bytes, state: MachineState) -> None:
    """
    Parse CAN ID 0x18FF02F0 - Power Data
    Bytes 0-1: Stack Voltage (uint16, factor 0.01)
    Bytes 2-3: Stack Current (uint16, factor 0.1)
    Bytes 4-5: DCF Output Voltage (uint16, factor 0.01)
    Bytes 6-7: DCF Output Current (uint16, factor 0.1)
    """
    if len(data) < 8:
        return
    
    stack_v = struct.unpack('<H', data[0:2])[0] * 0.01
    stack_i = struct.unpack('<H', data[2:4])[0] * 0.1
    dcf_v = struct.unpack('<H', data[4:6])[0] * 0.01
    dcf_i = struct.unpack('<H', data[6:8])[0] * 0.1
    
    stack_p = stack_v * stack_i
    dcf_p = dcf_v * dcf_i
    
    # Calculate efficiency
    efficiency = (dcf_p / stack_p * 100) if stack_p > 0 else 0
    if efficiency > 100:
        efficiency = 99.9
    
    state.power["stackVoltage"] = round(stack_v, 2)
    state.power["stackCurrent"] = round(stack_i, 1)
    state.power["stackPower"] = round(stack_p, 1)
    state.power["dcfOutVoltage"] = round(dcf_v, 2)
    state.power["dcfOutCurrent"] = round(dcf_i, 1)
    state.power["dcfPower"] = round(dcf_p, 1)
    state.power["dcfEfficiency"] = round(efficiency, 1)


def parse_msg3_sensors(data: bytes, state: MachineState) -> None:
    """
    Parse CAN ID 0x18FF03F0 - Sensor Data
    Bytes 0-1: Stack Temperature (int16, factor 0.1, offset -40°C)
    Bytes 2-3: Ambient Temperature (int16, factor 0.1, offset -40°C)
    Bytes 4-5: H2 Cylinder Pressure (uint16, factor 0.01 MPa)
    Byte 6: H2 Inlet Pressure (uint8, factor 0.01 MPa)
    Byte 7: H2 Concentration (uint8, factor 0.5 %vol)
    """
    if len(data) < 8:
        return
    
    stack_temp = struct.unpack('<h', data[0:2])[0] * 0.1 - 40
    ambient_temp = struct.unpack('<h', data[2:4])[0] * 0.1 - 40
    h2_cyl_press = struct.unpack('<H', data[4:6])[0] * 0.01
    h2_inlet_press = data[6] * 0.01
    h2_conc = data[7] * 0.5
    
    state.sensors["stackTemp"] = round(stack_temp, 1)
    state.sensors["ambientTemp"] = round(ambient_temp, 1)
    state.sensors["h2CylinderPressure"] = round(h2_cyl_press, 2)
    state.sensors["h2InletPressure"] = round(h2_inlet_press, 2)
    state.sensors["h2Concentration"] = round(h2_conc, 1)


def parse_msg4_io(data: bytes, state: MachineState) -> None:
    """
    Parse CAN ID 0x18FF04F0 - IO Status
    Byte 0: IO Flags (bit0=inlet, bit1=purge, bit2=prop, bit3=heater, bit4=fan1, bit5=fan2)
    Byte 1: Fan1 Duty Cycle (%)
    Bytes 2-3: DCF MOS Temperature (int16, factor 0.1, offset -40°C)
    Bytes 4-5: Fault Code (uint16)
    """
    if len(data) < 6:
        return
    
    flags = data[0]
    fan1_duty = data[1]
    dcf_mos_temp = struct.unpack('<h', data[2:4])[0] * 0.1 - 40
    fault_code = struct.unpack('<H', data[4:6])[0]
    
    state.io["h2InletValve"] = bool(flags & 0x01)
    state.io["h2PurgeValve"] = bool(flags & 0x02)
    state.io["proportionalValve"] = bool(flags & 0x04)
    state.io["heater"] = bool(flags & 0x08)
    state.io["fan1"] = bool(flags & 0x10)
    state.io["fan2"] = bool(flags & 0x20)
    state.io["fan1Duty"] = fan1_duty
    state.io["dcfMosTemp"] = round(dcf_mos_temp, 1)
    state.io["faultCode"] = fault_code


def generate_control_packet(control: Dict[str, Any]) -> bytes:
    """
    Generate CAN control packet for ID 0x18FF10A0
    
    Byte 0: Mode[1:0] | Command[4:2]
    Byte 1: Manual control flags
    Byte 2: Fan1 Target Speed (%)
    Bytes 3-4: DCF Target Voltage (uint16, factor 0.1)
    Bytes 5-6: DCF Target Current (uint16, factor 0.1)
    Byte 7: Reserved
    """
    data = bytearray(8)
    
    # Byte 0: Mode and Command
    mode = 1 if control.get("mode") == 1 else 0  # 0=MANUAL, 1=AUTO
    command = control.get("command", 0) & 0x07
    data[0] = mode | (command << 2)
    
    # Byte 1: Manual control flags (only in manual mode)
    if mode == 0:  # MANUAL
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
    
    # Byte 2: Fan1 Target Speed
    fan1_speed = max(0, min(100, int(control.get("fan1TargetSpeed", 50))))
    data[2] = fan1_speed
    
    # Bytes 3-4: DCF Target Voltage (factor 0.1)
    target_v = int(control.get("dcfTargetVoltage", 24.0) * 10)
    data[3] = target_v & 0xFF
    data[4] = (target_v >> 8) & 0xFF
    
    # Bytes 5-6: DCF Target Current (factor 0.1)
    target_i = int(control.get("dcfTargetCurrent", 5.0) * 10)
    data[5] = target_i & 0xFF
    data[6] = (target_i >> 8) & 0xFF
    
    # Byte 7: Reserved
    data[7] = 0
    
    return bytes(data)


# Message parsers mapping
MESSAGE_PARSERS = {
    0x18FF01F0: parse_msg1_status,
    0x18FF02F0: parse_msg2_power,
    0x18FF03F0: parse_msg3_sensors,
    0x18FF04F0: parse_msg4_io,
}
