"""
Configuration for CAN Backend Server
"""

# CAN Interface Selection
# Options: "zlg" (Hardware), "virtual" (Simulation)
CAN_INTERFACE_TYPE = "virtual"

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
