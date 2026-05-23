from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FirmwareSourceTest(unittest.TestCase):
    def test_rs485_uses_board_default_baudrate(self):
        rs485_header = (ROOT / "Hardware" / "485.h").read_text(encoding="utf-8")

        self.assertIn("#define RS485_BAUDRATE                 9600U", rs485_header)

    def test_firmware_sends_boot_probe_before_other_hardware_init(self):
        main_source = (ROOT / "User" / "main.c").read_text(encoding="utf-8")

        self.assertIn("static void TransmitSerialBootProbe(void);", main_source)
        self.assertIn(
            "RS485_Init();\n"
            "    TransmitSerialBootProbe();\n"
            "\n"
            "    ADC1_Mode_Config();",
            main_source,
        )

    def test_voltage_channels_are_not_zero_offset_calibrated(self):
        adc_source = (ROOT / "Hardware" / "bsp_adc.c").read_text(encoding="utf-8")

        self.assertIn(
            "float Get_CalibratedInputVoltage(void)\n"
            "{\n"
            "    return Get_ADC4_InputVoltage();\n"
            "}",
            adc_source,
        )
        self.assertIn(
            "float Get_CalibratedOutputVoltage(void)\n"
            "{\n"
            "    return Get_ADC3_OutputVoltage();\n"
            "}",
            adc_source,
        )
        self.assertNotIn("g_vi_calib.input_voltage_offset =", adc_source)
        self.assertNotIn("g_vi_calib.output_voltage_offset =", adc_source)


if __name__ == "__main__":
    unittest.main()
