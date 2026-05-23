import unittest

from modbus_protocol import (
    CONTROL_REGISTER,
    STATUS_BLOCKS,
    build_read_holding_registers_request,
    build_write_multiple_registers_request,
    decode_read_holding_registers_response,
    decode_write_multiple_registers_response,
    normalize_modbus_parity,
)


class ModbusProtocolTest(unittest.TestCase):
    def test_builds_standard_read_holding_registers_request(self):
        request = build_read_holding_registers_request(slave_id=1, start_register=0x0000, quantity=4)

        self.assertEqual(request, bytes.fromhex("01 03 00 00 00 04 44 09"))

    def test_decodes_status_block_response_to_original_payload_bytes(self):
        payload = bytes([0x12, 0x02, 0, 0, 0, 0, 0, 0])
        response = bytes.fromhex("01 03 08 12 02 00 00 00 00 00 00 36 C2")

        decoded = decode_read_holding_registers_response(response, slave_id=1, expected_quantity=4)

        self.assertEqual(decoded, payload)

    def test_builds_and_decodes_control_write_multiple_registers(self):
        payload = bytes([0x04, 0x19, 0x50, 0xE0, 0x01, 0x64, 0x00, 0x00])

        request = build_write_multiple_registers_request(
            slave_id=1,
            start_register=CONTROL_REGISTER,
            payload=payload,
        )
        response = request[:6] + bytes.fromhex("C0 36")

        self.assertEqual(request[:7], bytes.fromhex("01 10 01 00 00 04 08"))
        self.assertEqual(decode_write_multiple_registers_response(response, 1, CONTROL_REGISTER, 4), 4)

    def test_register_map_keeps_existing_four_status_payloads(self):
        self.assertEqual([block.message_type for block in STATUS_BLOCKS], [0x01, 0x02, 0x03, 0x04])
        self.assertEqual([block.quantity for block in STATUS_BLOCKS], [4, 4, 4, 4])

    def test_normalizes_modbus_serial_parity_names(self):
        self.assertEqual(normalize_modbus_parity("none"), "N")
        self.assertEqual(normalize_modbus_parity("even"), "E")
        self.assertEqual(normalize_modbus_parity("odd"), "O")
        self.assertEqual(normalize_modbus_parity("mark"), "M")
        self.assertEqual(normalize_modbus_parity("space"), "S")

    def test_rejects_invalid_modbus_serial_parity(self):
        with self.assertRaisesRegex(ValueError, "parity"):
            normalize_modbus_parity("invalid")


if __name__ == "__main__":
    unittest.main()
