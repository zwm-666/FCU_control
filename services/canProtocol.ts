import { MachineState, ControlState, SystemState, FaultLevel, WorkMode, INITIAL_MACHINE_STATE } from '../types';

// Helper to read Little Endian values
const getUint16 = (data: Uint8Array, offset: number) => data[offset] + (data[offset + 1] << 8);
const getInt16 = (data: Uint8Array, offset: number) => {
  const val = getUint16(data, offset);
  return val > 32767 ? val - 65536 : val;
};

// --- RX PARSERS ---

export const parseMsg1_Status = (data: Uint8Array, current: MachineState): MachineState => {
  const heartbeat = data[0];
  const state = data[1] & 0x03;
  const faultLevel = (data[1] >> 2) & 0x03;

  return {
    ...current,
    status: { heartbeat, state, faultLevel },
    lastUpdate: Date.now(),
  };
};

export const parseMsg2_Power = (data: Uint8Array, current: MachineState): MachineState => {
  const stackV = getUint16(data, 0) * 0.01;
  const stackI = getUint16(data, 2) * 0.1;
  const dcfV = getUint16(data, 4) * 0.01;
  const dcfI = getUint16(data, 6) * 0.1;

  const stackP = stackV * stackI;
  const dcfP = dcfV * dcfI;
  // Calculate efficiency, cap at 100% just in case of measurement noise in real sensors, though simulation is controlled
  let eff = stackP > 0 ? (dcfP / stackP) * 100 : 0;
  if (eff > 100) eff = 99.9; 

  return {
    ...current,
    power: {
      stackVoltage: parseFloat(stackV.toFixed(2)),
      stackCurrent: parseFloat(stackI.toFixed(1)),
      stackPower: parseFloat(stackP.toFixed(1)),
      dcfOutVoltage: parseFloat(dcfV.toFixed(2)),
      dcfOutCurrent: parseFloat(dcfI.toFixed(1)),
      dcfPower: parseFloat(dcfP.toFixed(1)),
      dcfEfficiency: parseFloat(eff.toFixed(1)),
    }
  };
};

export const parseMsg3_Sensors = (data: Uint8Array, current: MachineState): MachineState => {
  const stackT = (getInt16(data, 0) * 0.1) - 40;
  const ambT = (getInt16(data, 2) * 0.1) - 40;
  const h2CylP = getUint16(data, 4) * 0.01;
  const h2InP = data[6] * 0.01;
  const h2Conc = data[7] * 0.5;

  return {
    ...current,
    sensors: {
      stackTemp: parseFloat(stackT.toFixed(1)),
      ambientTemp: parseFloat(ambT.toFixed(1)),
      h2CylinderPressure: parseFloat(h2CylP.toFixed(2)),
      h2InletPressure: parseFloat(h2InP.toFixed(2)),
      h2Concentration: parseFloat(h2Conc.toFixed(1)),
    }
  };
};

export const parseMsg4_IO = (data: Uint8Array, current: MachineState): MachineState => {
  const flags = data[0];
  const fan1Duty = data[1];
  const dcfMosTemp = (getInt16(data, 2) * 0.1) - 40;
  const faultCode = getUint16(data, 4);

  return {
    ...current,
    io: {
      h2InletValve: !!(flags & 0x01),
      h2PurgeValve: !!(flags & 0x02),
      proportionalValve: !!(flags & 0x04),
      heater: !!(flags & 0x08),
      fan1: !!(flags & 0x10),
      fan2: !!(flags & 0x20),
      fan1Duty,
      dcfMosTemp: parseFloat(dcfMosTemp.toFixed(1)),
      faultCode,
    }
  };
};

// --- TX GENERATOR ---

export const generateControlPacket = (control: ControlState): { id: number, data: number[] } => {
  const data = new Array(8).fill(0);

  // Byte 0: Mode and Command
  let byte0 = 0;
  byte0 |= (control.mode === WorkMode.AUTO ? 1 : 0) & 0x03;
  byte0 |= (control.command & 0x07) << 2;
  data[0] = byte0;

  // Byte 1: Forced Control (Manual Mode Only)
  if (control.mode === WorkMode.MANUAL) {
    let byte1 = 0;
    if (control.forceInletValve) byte1 |= 0x01;
    if (control.forcePurgeValve) byte1 |= 0x02;
    if (control.forceHeater) byte1 |= 0x04; // Heater bit 2 in byte 1 based on manual mapping
    if (control.forceFan1) byte1 |= 0x08; // Fan 1 bit 3
    if (control.forceFan2) byte1 |= 0x10; // Fan 2 bit 4
    // Note: Proportional valve missing from explicit byte 1 mapping in prompt, assuming managed internally or by automation for now.
    data[1] = byte1;
  }

  // Byte 2: Fan 1 Speed
  data[2] = Math.min(100, Math.max(0, Math.round(control.fan1TargetSpeed)));

  // Byte 3-4: DCF Target Voltage (uint16, 0.1 factor)
  const targetV_scaled = Math.round(control.dcfTargetVoltage * 10);
  data[3] = targetV_scaled & 0xFF;
  data[4] = (targetV_scaled >> 8) & 0xFF;

  // Byte 5-6: DCF Target Current (uint16, 0.1 factor)
  const targetI_scaled = Math.round(control.dcfTargetCurrent * 10);
  data[5] = targetI_scaled & 0xFF;
  data[6] = (targetI_scaled >> 8) & 0xFF;

  // Byte 7: Reserved

  return { id: 0x18FF10A0, data };
};

// --- SIMULATION HELPERS ---

export const simulateIncomingData = (prev: MachineState): MachineState => {
  const isRunning = prev.status.state === SystemState.RUN;
  
  // Create copies of byte arrays to simulate raw CAN
  const msg1 = new Uint8Array(8);
  const msg2 = new Uint8Array(8);
  const msg3 = new Uint8Array(8);
  const msg4 = new Uint8Array(8);

  // MSG 1
  msg1[0] = (prev.status.heartbeat + 1) % 256;
  msg1[1] = prev.status.state | (prev.status.faultLevel << 2);

  // MSG 2 (Power) - Add some noise/sine wave
  const time = Date.now() / 1000;
  const baseV = isRunning ? 45 + Math.sin(time) * 2 : 0;
  const baseI = isRunning ? 20 + Math.cos(time * 0.5) * 5 : 0;
  
  // Convert to uint16 based on factors
  const sv = Math.max(0, baseV * 100); // 0.01 factor
  const si = Math.max(0, baseI * 10);  // 0.1 factor
  
  msg2[0] = sv & 0xFF; msg2[1] = sv >> 8;
  msg2[2] = si & 0xFF; msg2[3] = si >> 8;
  
  // DCDC Simulation
  // We want Efficiency to be between 90% and 99%
  // Stack Power = baseV * baseI
  // DCF Power = Stack Power * efficiency
  // DCF Voltage = 24V (Fixed target usually)
  // DCF Current = DCF Power / DCF Voltage
  
  const targetEfficiency = 0.92 + (Math.sin(time) * 0.03); // Oscillates between ~0.89 and 0.95
  const realStackPower = baseV * baseI;
  const dcfPower = realStackPower * targetEfficiency;
  
  const dcfVoltageReal = isRunning ? 24.0 + (Math.random() * 0.1) : 0;
  const dcfCurrentReal = isRunning && dcfVoltageReal > 0 ? dcfPower / dcfVoltageReal : 0;

  const dcfV_int = Math.round(dcfVoltageReal * 100); // Factor 0.01
  const dcfI_int = Math.round(dcfCurrentReal * 10);  // Factor 0.1

  msg2[4] = dcfV_int & 0xFF; msg2[5] = dcfV_int >> 8;
  msg2[6] = dcfI_int & 0xFF; msg2[7] = dcfI_int >> 8;

  // MSG 3 (Sensors)
  const tempS = (60 + Math.sin(time/5) * 5 + 40) * 10; // ~60C
  const tempA = (25 + 40) * 10;
  const pressC = 1200 + Math.sin(time/10) * 10; // 12MPa
  const pressI = 50; // 0.5 MPa
  
  msg3[0] = tempS & 0xFF; msg3[1] = tempS >> 8;
  msg3[2] = tempA & 0xFF; msg3[3] = tempA >> 8;
  msg3[4] = pressC & 0xFF; msg3[5] = pressC >> 8;
  msg3[6] = pressI;
  msg3[7] = 0; // H2 Conc

  // MSG 4 (IO)
  let ioFlags = 0;
  if (isRunning) {
    ioFlags |= 0x01; // Inlet Open
    ioFlags |= 0x10; // Fan 1 Run
    ioFlags |= 0x20; // Fan 2 Run
  }
  msg4[0] = ioFlags;
  msg4[1] = isRunning ? 65 : 0; // 65% PWM
  const mosT = (45 + 40) * 10;
  msg4[2] = mosT & 0xFF; msg4[3] = mosT >> 8;
  
  // Process all
  let next = parseMsg1_Status(msg1, prev);
  next = parseMsg2_Power(msg2, next);
  next = parseMsg3_Sensors(msg3, next);
  next = parseMsg4_IO(msg4, next);

  return next;
}