"""
RS485 frame protocol for the H2 FCU backend.

The payload bytes intentionally stay compatible with the existing CAN parser:
each RS485 message type carries the same 8-byte payload that the CAN ID used.
"""

from dataclasses import dataclass
from typing import Dict, List


FRAME_HEADER = b"\xAA\x55"
MAX_PAYLOAD_LEN = 255
SERIAL_CAN_PAYLOAD_LEN = 8
SERIAL_CAN_FRAME_LEN = 16

RS485_TYPE_TO_CAN_ID: Dict[int, int] = {
    0x01: 0x18FF01F0,  # 系统状态
    0x02: 0x18FF02F0,  # 电源数据
    0x03: 0x18FF03F0,  # 传感器数据
    0x04: 0x18FF04F0,  # IO 状态
    0x10: 0x18FF10A0,  # 控制命令
}

CAN_ID_TO_RS485_TYPE: Dict[int, int] = {
    can_id: message_type for message_type, can_id in RS485_TYPE_TO_CAN_ID.items()
}


@dataclass(frozen=True)
class RS485Frame:
    message_type: int
    payload: bytes


def checksum8(data: bytes) -> int:
    """Return the low 8 bits of the byte sum used by the MCU serial frame."""
    return sum(data) & 0xFF


def crc16_modbus(data: bytes) -> int:
    """Return CRC-16/MODBUS over data."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def build_rs485_frame(message_type: int, payload: bytes) -> bytes:
    """Build AA 55 CAN_ID_LE LEN PAYLOAD SUM for the current MCU firmware."""
    if not 0 <= message_type <= 0xFF:
        raise ValueError("message_type must fit in one byte")
    if len(payload) != SERIAL_CAN_PAYLOAD_LEN:
        raise ValueError("payload must be exactly 8 bytes for RS485 CAN-compatible frame")

    can_id = RS485_TYPE_TO_CAN_ID.get(message_type)
    if can_id is None:
        raise ValueError("unknown RS485 message type")

    body = can_id.to_bytes(4, "little") + bytes([len(payload)]) + bytes(payload)
    return FRAME_HEADER + body + bytes([checksum8(body)])


def build_legacy_rs485_frame(message_type: int, payload: bytes) -> bytes:
    """Build legacy AA 55 TYPE LEN PAYLOAD CRC_LO CRC_HI."""
    if not 0 <= message_type <= 0xFF:
        raise ValueError("message_type must fit in one byte")
    if len(payload) > MAX_PAYLOAD_LEN:
        raise ValueError("payload too long for RS485 frame")
    body = bytes([message_type, len(payload)]) + bytes(payload)
    crc = crc16_modbus(body)
    return FRAME_HEADER + body + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def decode_legacy_rs485_frame(frame: bytes) -> RS485Frame:
    """Decode and validate one complete legacy TYPE/LEN/CRC frame."""
    if len(frame) < 6:
        raise ValueError("frame too short")
    if frame[:2] != FRAME_HEADER:
        raise ValueError("bad frame header")

    payload_len = frame[3]
    expected_len = 2 + 1 + 1 + payload_len + 2
    if len(frame) != expected_len:
        raise ValueError("frame length mismatch")

    body = frame[2:-2]
    received_crc = frame[-2] | (frame[-1] << 8)
    expected_crc = crc16_modbus(body)
    if received_crc != expected_crc:
        raise ValueError("bad frame crc")

    return RS485Frame(message_type=frame[2], payload=bytes(frame[4:-2]))


def decode_serial_can_frame(frame: bytes) -> RS485Frame:
    """Decode and validate the current MCU 16-byte CAN-ID serial frame."""
    if len(frame) != SERIAL_CAN_FRAME_LEN:
        raise ValueError("frame length mismatch")
    if frame[:2] != FRAME_HEADER:
        raise ValueError("bad frame header")
    if frame[6] != SERIAL_CAN_PAYLOAD_LEN:
        raise ValueError("unexpected payload length")

    expected_checksum = checksum8(frame[2:-1])
    if frame[-1] != expected_checksum:
        raise ValueError("bad frame checksum")

    can_id = int.from_bytes(frame[2:6], "little")
    message_type = CAN_ID_TO_RS485_TYPE.get(can_id)
    if message_type is None:
        raise ValueError("unknown CAN ID")

    return RS485Frame(message_type=message_type, payload=bytes(frame[7:15]))


def decode_rs485_frame(frame: bytes) -> RS485Frame:
    """Decode either the current MCU frame or the legacy TYPE/LEN/CRC frame."""
    if len(frame) == SERIAL_CAN_FRAME_LEN:
        try:
            return decode_serial_can_frame(frame)
        except ValueError:
            pass
    return decode_legacy_rs485_frame(frame)


class RS485FrameParser:
    """Incremental parser for noisy or fragmented serial streams."""

    def __init__(self):
        self._buffer = bytearray()

    def _looks_like_serial_can_frame(self) -> bool:
        if len(self._buffer) < SERIAL_CAN_FRAME_LEN:
            return False
        if self._buffer[:2] != FRAME_HEADER:
            return False
        can_id = int.from_bytes(self._buffer[2:6], "little")
        return self._buffer[6] == SERIAL_CAN_PAYLOAD_LEN and can_id in CAN_ID_TO_RS485_TYPE

    def feed(self, data: bytes) -> List[Dict]:
        self._buffer.extend(data)
        messages: List[Dict] = []

        while True:
            header_index = self._buffer.find(FRAME_HEADER)
            if header_index < 0:
                self._buffer[:] = self._buffer[-1:] if self._buffer.endswith(b"\xAA") else b""
                break

            if header_index > 0:
                del self._buffer[:header_index]

            if len(self._buffer) < 6:
                break

            if self._looks_like_serial_can_frame():
                raw_frame = bytes(self._buffer[:SERIAL_CAN_FRAME_LEN])
                del self._buffer[:SERIAL_CAN_FRAME_LEN]

                try:
                    decoded = decode_serial_can_frame(raw_frame)
                except ValueError:
                    continue

                can_id = RS485_TYPE_TO_CAN_ID[decoded.message_type]
                messages.append(
                    {
                        "arbitration_id": can_id,
                        "data": decoded.payload,
                        "is_extended_id": True,
                        "timestamp": 0,
                        "rs485_type": decoded.message_type,
                    }
                )
                continue

            payload_len = self._buffer[3]
            frame_len = 2 + 1 + 1 + payload_len + 2
            if len(self._buffer) < frame_len:
                break

            raw_frame = bytes(self._buffer[:frame_len])
            del self._buffer[:frame_len]

            try:
                decoded = decode_rs485_frame(raw_frame)
            except ValueError:
                continue

            can_id = RS485_TYPE_TO_CAN_ID.get(decoded.message_type)
            if can_id is None:
                continue

            messages.append(
                {
                    "arbitration_id": can_id,
                    "data": decoded.payload,
                    "is_extended_id": True,
                    "timestamp": 0,
                    "rs485_type": decoded.message_type,
                }
            )

        return messages
