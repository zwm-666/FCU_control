import builtins
import unittest
from unittest.mock import patch

import server


class FakeSerial:
    def __init__(self, responses, stale=b""):
        self.responses = list(responses)
        self.written = []
        self.is_open = True
        self.stale = bytes(stale)

    @property
    def in_waiting(self):
        return len(self.stale)

    def reset_input_buffer(self):
        self.stale = b""
        return None

    def write(self, data):
        self.written.append(bytes(data))

    def flush(self):
        return None

    def read(self, _size):
        if self.stale:
            data = self.stale
            self.stale = b""
            return data
        if self.responses:
            return self.responses.pop(0)
        return b""

    def close(self):
        self.is_open = False


class RS485DriverTest(unittest.TestCase):
    def test_open_reports_how_to_install_pyserial_and_switch_to_virtual_mode(self):
        driver = server.RS485Driver("COM3", 115200)
        original_import = builtins.__import__

        def import_without_pyserial(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "serial":
                raise ImportError("No module named 'serial'")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=import_without_pyserial):
            with self.assertLogs(server.logger, level="ERROR") as log_output:
                opened = driver.open()

        self.assertFalse(opened)
        combined_logs = "\n".join(log_output.output)
        self.assertIn("pyserial is not installed", combined_logs)
        self.assertIn("pip install -r backend/requirements.txt", combined_logs)
        self.assertIn('CAN_INTERFACE_TYPE = "virtual"', combined_logs)

    def test_receive_decodes_rs485_frame(self):
        driver = server.RS485Driver("COM11", 115200)
        driver.serial = FakeSerial([bytes.fromhex("AA 55 F0 01 FF 18 08 12 02 00 00 00 00 00 00 24")])
        driver.running = True

        with self.assertLogs(server.logger, level="INFO") as log_output:
            messages = driver.receive()

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["arbitration_id"], 0x18FF01F0)
        self.assertEqual(messages[0]["data"], bytes.fromhex("12 02 00 00 00 00 00 00"))
        combined_logs = "\n".join(log_output.output)
        self.assertIn("RS485 RX", combined_logs)
        self.assertIn("AA 55 F0 01 FF 18 08 12 02 00 00 00 00 00 00 24", combined_logs)

    def test_receive_logs_invalid_rs485_frame(self):
        driver = server.RS485Driver("COM11", 115200)
        driver.serial = FakeSerial([bytes.fromhex("AA 55 01 08 12 02 00 00 00 00 00 00 E5 18")])
        driver.running = True

        with self.assertLogs(server.logger, level="WARNING") as log_output:
            messages = driver.receive()

        self.assertEqual(messages, [])
        combined_logs = "\n".join(log_output.output)
        self.assertIn("Discarding invalid RS485 bytes", combined_logs)

    def test_send_encodes_rs485_control_frame(self):
        driver = server.RS485Driver("COM11", 115200)
        driver.serial = FakeSerial([])
        driver.running = True
        payload = bytes.fromhex("04 19 50 E0 01 64 00 00")

        with self.assertLogs(server.logger, level="INFO") as log_output:
            sent = driver.send(server.CAN_TX_ID, payload)

        self.assertTrue(sent)
        self.assertEqual(driver.serial.written, [bytes.fromhex("AA 55 A0 10 FF 18 08 04 19 50 E0 01 64 00 00 81")])
        combined_logs = "\n".join(log_output.output)
        self.assertIn("RS485 TX", combined_logs)
        self.assertIn("AA 55 A0 10 FF 18 08 04 19 50 E0 01 64 00 00 81", combined_logs)

    def test_receive_decodes_rs485_frame_with_noise_prefix(self):
        driver = server.RS485Driver("COM11", 115200)
        driver.serial = FakeSerial([bytes.fromhex("99 88 AA 55 F0 01 FF 18 08 12 02 00 00 00 00 00 00 24")])
        driver.running = True

        with self.assertLogs(server.logger, level="INFO") as log_output:
            messages = driver.receive()

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["arbitration_id"], 0x18FF01F0)

    def test_receive_still_decodes_legacy_rs485_frame(self):
        driver = server.RS485Driver("COM11", 115200)
        driver.serial = FakeSerial([bytes.fromhex("AA 55 01 08 12 02 00 00 00 00 00 00 E5 17")])
        driver.running = True

        messages = driver.receive()

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["arbitration_id"], 0x18FF01F0)
        self.assertEqual(messages[0]["data"], bytes.fromhex("12 02 00 00 00 00 00 00"))


class CANWebSocketServerDiagnosisTest(unittest.TestCase):
    def test_server_initializes_diagnosis_with_model_prediction_disabled(self):
        with patch.object(server, "DIAGNOSIS_AVAILABLE", True):
            with patch.object(server, "OnlineDiagnosis") as mock_diagnosis:
                server.CANWebSocketServer()

        _, kwargs = mock_diagnosis.call_args
        self.assertFalse(kwargs["enable_model_prediction"])


if __name__ == "__main__":
    unittest.main()
