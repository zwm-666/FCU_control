"""
Configuration for H2 FCU backend transport.
"""

# Interface Selection
# Options: "rs485" (custom serial frame over USB-RS485), "zlg" (USB-CAN hardware), "virtual" (simulation)
CAN_INTERFACE_TYPE = "rs485"

# RS485 Configuration
# Windows examples: "COM3", "COM5"
# Linux examples: "/dev/ttyUSB0", "/dev/ttyCH341USB0"
RS485_PORT = "COM12"
RS485_BAUDRATE = 9600
RS485_TIMEOUT = 0.01

# ZLG USB-CAN Device Configuration
# Device Type (for USBCAN devices):
#   USBCAN1 = 3
#   USBCAN2 = 4
CANALYST_DEVICE_TYPE = 4  # USBCAN2

# Device Index: 0 for first device, 1 for second, etc.
CANALYST_DEVICE_INDEX = 0

# CAN Channel: 0 or 1
CANALYST_CHANNEL = 0

# CAN Bitrate (must match your FCU configuration)
CAN_BITRATE = 250000

# WebSocket Server Configuration
WEBSOCKET_HOST = "0.0.0.0"  # Listen on all interfaces
WEBSOCKET_PORT = 8765

# Diagnosis Configuration
# False: 仅使用规则诊断，不加载/不调用深度学习模型
# True: 启用模型预测，模型不可用时仍会自动回退到规则诊断
DIAGNOSIS_ENABLE_MODEL_PREDICTION = False

# CAN Message IDs (matching frontend protocol)
CAN_RX_IDS = {
    0x18FF01F0: "status",      # 系统状态
    0x18FF02F0: "power",       # 功率数据
    0x18FF03F0: "sensors",     # 传感器数据
    0x18FF04F0: "io",          # IO状态
}

CAN_TX_ID = 0x18FF10A0  # 控制命令

# Update rate (Hz) for broadcasting machine state
BROADCAST_RATE = 10  # 10 Hz = 100ms interval
