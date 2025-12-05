"""
Configuration for CAN Backend Server
"""

# CAN Interface Configuration
# For ZLG-compatible USB-CAN adapter (using ControlCAN.dll)
CAN_INTERFACE = "zlgcan"

# ZLG Device Configuration
# Device Type: 
#   - USBCAN1 = 3
#   - USBCAN2 = 4
#   - USBCANFD = 41 (for CAN-FD devices)
ZLG_DEVICE_TYPE = 4  # USBCAN2 (most common)

# Device Index: 0 for first device, 1 for second, etc.
ZLG_DEVICE_INDEX = 0

# CAN Channel: 0 or 1 (most ZLG devices have 2 channels)
ZLG_CHANNEL = 0

# CAN Bitrate (must match your FCU configuration)
# Common values: 10000, 20000, 50000, 100000, 125000, 250000, 500000, 800000, 1000000
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
