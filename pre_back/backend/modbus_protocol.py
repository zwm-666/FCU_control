"""
Modbus RTU helpers for the H2 FCU backend.

The backend acts as a Modbus master. The STM32 FCU acts as slave 1 by default.
Register values carry the same byte payloads that the previous CAN protocol used,
so the existing dashboard parser can continue to consume 8-byte payload blocks.
"""

from dataclasses import dataclass
from typing import Dict, List


READ_HOLDING_REGISTERS = 0x03
WRITE_MULTIPLE_REGISTERS = 0x10

CONTROL_REGISTER = 0x0100
REGISTERS_PER_PAYLOAD = 4
MODBUS_PARITY_ALIASES = {
    "N": "N",
    "NONE": "N",
    "NO": "N",
    "E": "E",
    "EVEN": "E",
    "O": "O",
    "ODD": "O",
    "M": "M",
    "MARK": "M",
    "S": "S",
    "SPACE": "S",
}


@dataclass(frozen=True)
class ModbusStatusBlock:
    message_type: int
    can_id: int
    start_register: int
    quantity: int = REGISTERS_PER_PAYLOAD


STATUS_BLOCKS: List[ModbusStatusBlock] = [
    ModbusStatusBlock(0x01, 0x18FF01F0, 0x0000),
    ModbusStatusBlock(0x02, 0x18FF02F0, 0x0010),
    ModbusStatusBlock(0x03, 0x18FF03F0, 0x0020),
    ModbusStatusBlock(0x04, 0x18FF04F0, 0x0030),
]

CAN_ID_TO_STATUS_BLOCK: Dict[int, ModbusStatusBlock] = {
    block.can_id: block for block in STATUS_BLOCKS
}


def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def append_crc(frame_without_crc: bytes) -> bytes:
    crc = crc16_modbus(frame_without_crc)
    return frame_without_crc + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def normalize_modbus_parity(parity: str) -> str:
    normalized = MODBUS_PARITY_ALIASES.get(str(parity).strip().upper())
    if normalized is None:
        raise ValueError("parity must be one of: N, E, O, M, S")
    return normalized


def validate_crc(frame: bytes) -> None:
    if len(frame) < 4:
        raise ValueError("modbus frame too short")
    expected = crc16_modbus(frame[:-2])
    received = frame[-2] | (frame[-1] << 8)
    if received != expected:
        raise ValueError("bad modbus crc")


def build_read_holding_registers_request(slave_id: int, start_register: int, quantity: int) -> bytes:
    if not 1 <= slave_id <= 247:
        raise ValueError("slave_id must be 1..247")
    if not 1 <= quantity <= 125:
        raise ValueError("quantity must be 1..125")

    frame = bytes(
        [
            slave_id,
            READ_HOLDING_REGISTERS,
            (start_register >> 8) & 0xFF,
            start_register & 0xFF,
            (quantity >> 8) & 0xFF,
            quantity & 0xFF,
        ]
    )
    return append_crc(frame)


def decode_read_holding_registers_response(response: bytes, slave_id: int, expected_quantity: int) -> bytes:
    validate_crc(response)
    expected_len = 3 + expected_quantity * 2 + 2
    if len(response) != expected_len:
        raise ValueError("unexpected read response length")
    if response[0] != slave_id:
        raise ValueError("unexpected slave id")
    if response[1] != READ_HOLDING_REGISTERS:
        raise ValueError("unexpected function code")
    if response[2] != expected_quantity * 2:
        raise ValueError("unexpected byte count")
    return bytes(response[3:-2])


def build_write_multiple_registers_request(slave_id: int, start_register: int, payload: bytes) -> bytes:
    if len(payload) == 0 or len(payload) % 2 != 0:
        raise ValueError("payload must contain whole 16-bit registers")

    quantity = len(payload) // 2
    if quantity > 123:
        raise ValueError("too many registers for one Modbus write")

    frame = bytes(
        [
            slave_id,
            WRITE_MULTIPLE_REGISTERS,
            (start_register >> 8) & 0xFF,
            start_register & 0xFF,
            (quantity >> 8) & 0xFF,
            quantity & 0xFF,
            len(payload),
        ]
    ) + bytes(payload)
    return append_crc(frame)


def decode_write_multiple_registers_response(
    response: bytes,
    slave_id: int,
    expected_start_register: int,
    expected_quantity: int,
) -> int:
    validate_crc(response)
    if len(response) != 8:
        raise ValueError("unexpected write response length")
    if response[0] != slave_id:
        raise ValueError("unexpected slave id")
    if response[1] != WRITE_MULTIPLE_REGISTERS:
        raise ValueError("unexpected function code")

    start_register = (response[2] << 8) | response[3]
    quantity = (response[4] << 8) | response[5]
    if start_register != expected_start_register or quantity != expected_quantity:
        raise ValueError("write response does not match request")
    return quantity
