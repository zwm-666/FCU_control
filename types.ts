
// Enums based on Protocol Definition

export enum SystemState {
  OFF = 0,
  START = 1,
  RUN = 2,
  FAULT = 3,
}

export enum FaultLevel {
  NORMAL = 0,
  WARNING = 1,
  SEVERE = 2,
  EMERGENCY = 3,
}

export enum WorkMode {
  MANUAL = 0,
  AUTO = 1,
}

export enum ControlCommand {
  NONE = 0,
  START = 1,
  RESET = 2,
  EMERGENCY_STOP = 3,
  SHUTDOWN = 4,
}

export interface SystemStatus {
  heartbeat: number;
  state: SystemState;
  faultLevel: FaultLevel;
}

export interface PowerData {
  stackVoltage: number; // V
  stackCurrent: number; // A
  stackPower: number;   // W (Derived)
  dcfOutVoltage: number; // V
  dcfOutCurrent: number; // A
  dcfPower: number;      // W (Derived)
  dcfEfficiency: number; // % (Derived)
}

export interface SensorData {
  stackTemp: number; // C
  ambientTemp: number; // C
  h2CylinderPressure: number; // MPa
  h2InletPressure: number; // MPa
  h2Concentration: number; // %vol
}

export interface IOStatus {
  h2InletValve: boolean;
  h2PurgeValve: boolean;
  proportionalValve: boolean;
  heater: boolean;
  fan1: boolean;
  fan2: boolean;
  fan1Duty: number; // %
  dcfMosTemp: number; // C
  faultCode: number;
}

// Complete State of the Machine
export interface MachineState {
  connected: boolean;
  lastUpdate: number;
  status: SystemStatus;
  power: PowerData;
  sensors: SensorData;
  io: IOStatus;
}

// Control State (Tx)
export interface ControlState {
  mode: WorkMode;
  command: ControlCommand;
  // Manual Overrides
  forceInletValve: boolean;
  forcePurgeValve: boolean;
  forceHeater: boolean;
  forceFan1: boolean;
  forceFan2: boolean;
  // Setpoints
  fan1TargetSpeed: number; // %
  dcfTargetVoltage: number; // V
  dcfTargetCurrent: number; // A
}

export interface ConnectionConfig {
  interfaceType: string;
  channel: string;
  bitrate: string;
}

export const FAULT_CODES: Record<number, string> = {
  0x00: "系统正常",
  0x01: "DCF 输入欠压 (严重)",
  0x02: "DCF 输入过压 (严重)",
  0x03: "DCF 输入过流 (紧急)",
  0x04: "DCF 输出欠压 (警告)",
  0x05: "DCF 输出过压 (严重)",
  0x06: "DCF 输出过流 (警告)",
  0x07: "DCF 内部过热 (严重)",
  0x10: "电堆过热 (严重)",
  0x11: "氢气泄漏 / 浓度过高 (紧急)",
  0x12: "氢气入口压力异常 (严重)",
  0x13: "CAN 通讯超时 (警告)",
  0x20: "风扇 1 故障 (警告)",
  0x21: "风扇 2 故障 (警告)",
};

export const INITIAL_MACHINE_STATE: MachineState = {
  connected: false,
  lastUpdate: 0,
  status: { heartbeat: 0, state: SystemState.OFF, faultLevel: FaultLevel.NORMAL },
  power: { stackVoltage: 0, stackCurrent: 0, stackPower: 0, dcfOutVoltage: 0, dcfOutCurrent: 0, dcfPower: 0, dcfEfficiency: 0 },
  sensors: { stackTemp: 25, ambientTemp: 25, h2CylinderPressure: 0, h2InletPressure: 0, h2Concentration: 0 },
  io: { h2InletValve: false, h2PurgeValve: false, proportionalValve: false, heater: false, fan1: false, fan2: false, fan1Duty: 0, dcfMosTemp: 25, faultCode: 0 }
};

export const INITIAL_CONTROL_STATE: ControlState = {
  mode: WorkMode.MANUAL,
  command: ControlCommand.NONE,
  forceInletValve: false,
  forcePurgeValve: false,
  forceHeater: false,
  forceFan1: false,
  forceFan2: false,
  fan1TargetSpeed: 50,
  dcfTargetVoltage: 24.0,
  dcfTargetCurrent: 5.0,
};
