import { MachineState, ControlState, WorkMode } from "../types";

// Helper to read Little Endian values
const getUint16 = (data: Uint8Array, offset: number) =>
  data[offset] + (data[offset + 1] << 8);

const getInt16 = (data: Uint8Array, offset: number) => {
  const val = getUint16(data, offset);
  return val > 32767 ? val - 65536 : val;
};

// --- RX PARSERS ---

export const parseMsg1_Status = (
  data: Uint8Array,
  current: MachineState,
): MachineState => {
  const heartbeat = data[0];
  const state = data[1] & 0x03;
  const faultLevel = (data[1] >> 2) & 0x03;

  return {
    ...current,
    status: { heartbeat, state, faultLevel },
    lastUpdate: Date.now(),
  };
};

export const parseMsg2_Power = (
  data: Uint8Array,
  current: MachineState,
): MachineState => {
  const stackV = getUint16(data, 0) * 0.01;
  const stackI = getUint16(data, 2) * 0.1;
  const dcfV = getUint16(data, 4) * 0.01;
  const dcfI = getUint16(data, 6) * 0.1;

  const stackP = stackV * stackI;
  const dcfP = dcfV * dcfI;
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
    },
  };
};

export const parseMsg3_Sensors = (
  data: Uint8Array,
  current: MachineState,
): MachineState => {
  const stackT = getInt16(data, 0) * 0.1 - 40;
  const ambT = getInt16(data, 2) * 0.1 - 40;
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
    },
  };
};

export const parseMsg4_IO = (
  data: Uint8Array,
  current: MachineState,
): MachineState => {
  const flags = data[0];
  const fan1Duty = data[1];
  const dcfMosTemp = getInt16(data, 2) * 0.1 - 40;
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
    },
  };
};

// --- TX GENERATOR ---

export const generateControlPacket = (
  control: ControlState,
): { id: number; data: number[] } => {
  const data = new Array(8).fill(0);

  // Byte 0: Mode[1:0] | Command[4:2]
  let byte0 = 0;
  byte0 |= (control.mode === WorkMode.AUTO ? 1 : 0) & 0x03;
  byte0 |= (control.command & 0x07) << 2;
  data[0] = byte0;

  // Byte 1: Forced control flags (manual mode only)
  if (control.mode === WorkMode.MANUAL) {
    let byte1 = 0;
    if (control.forceInletValve) byte1 |= 0x01;
    if (control.forcePurgeValve) byte1 |= 0x02;
    if (control.forceHeater) byte1 |= 0x04;
    if (control.forceFan1) byte1 |= 0x08;
    if (control.forceFan2) byte1 |= 0x10;
    data[1] = byte1;
  }

  // Byte 2: Fan 1 target speed (0-100%)
  data[2] = Math.min(100, Math.max(0, Math.round(control.fan1TargetSpeed)));

  // Bytes 3-4: DCF target voltage (uint16 LE, factor 0.1)
  const targetV = Math.round(control.dcfTargetVoltage * 10);
  data[3] = targetV & 0xff;
  data[4] = (targetV >> 8) & 0xff;

  // Bytes 5-6: DCF target current (uint16 LE, factor 0.1)
  const targetI = Math.round(control.dcfTargetCurrent * 10);
  data[5] = targetI & 0xff;
  data[6] = (targetI >> 8) & 0xff;

  // Byte 7: Reserved
  data[7] = 0;

  return { id: 0x18ff10a0, data };
};
