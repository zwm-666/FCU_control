# H2 FCU CAN Backend 使用说明

## 概述

这是一个 Python 后端服务，用于 H2 FCU 控制台与 CAN 总线之间的通信。
它使用 `ctypes` 直接调用 ZLG `ControlCAN.dll` 驱动（CANalyst-II 接口），并通过 WebSocket 实时转发数据给前端。

**版本特点**: 
- 不依赖特定的 CAN 库 (如 python-can, zlgcan)
- 直接通过 DLL 交互，更稳定
- 专为 64位 Python 环境优化

---

## 依赖要求

1. **Python 环境**: 
   - 推荐使用 **Anaconda Python** (64位)
   - 仅需一个依赖包: `websockets`

2. **硬件/驱动**:
   - CANalyst-II (ZLG USBCAN-2 兼容设备)
   - 必须安装设备驱动
   - `ControlCAN.dll` (64位) 必须存在于 `kerneldlls/` 或当前目录

---

## 快速安装与启动

### 1. 安装依赖

```powershell
# 如果使用 Anaconda (推荐)
conda activate base
conda install websockets

# 或者使用 pip
pip install websockets
```

### 2. 启动服务器

> ⚠️ **重要**: 启动前请务必关闭其他占用 CAN 设备的程序（如 USB-CAN Tool）！

```powershell
# 在 backend 目录下运行
conda run -n base python server.py
# 或者直接 python server.py (取决于你的环境)
```

**启动成功的标志**:
- 控制台显示 "✓ CAN Bus initialized (ZLG Driver)"
- 控制台显示 "✓ WebSocket server started"

### 3. 连接前端

启动前端项目 (`npm run dev`)，点击右上角 "连接" 按钮。默认连接 `ws://localhost:8765`。

---

## 文件结构

- `server.py`: **核心服务**。包含 CAN 驱动封装 (ZLGCanDriver) 和 WebSocket 服务器。
- `config.py`: **配置文件**。设置 CAN 波特率、通道、设备类型等。
- `can_protocol.py`: **协议定义**。负责 CAN 报文的解析和生成。
- `requirements.txt`: Python 依赖列表。
- `kerneldlls/`: 存放 64位 DLL 文件 (`ControlCAN.dll`, `usbcan.dll`)。

---

## 配置说明 (config.py)

```python
# 设备类型 (USBCAN2 = 4)
CANALYST_DEVICE_TYPE = 4

# 设备索引和通道
CANALYST_DEVICE_INDEX = 0
CANALYST_CHANNEL = 0

# CAN 波特率 (支持 10K - 1M)
CAN_BITRATE = 250000  # 默认 250K

# WebSocket 端口
WEBSOCKET_PORT = 8765
```

---

## 协议定义

| CAN ID | 方向 | 描述 | 内容 |
|--------|------|------|------|
| 0x18FF01F0 | RX | 系统状态 | 心跳, 运行状态, 故障等级 |
| 0x18FF02F0 | RX | 功率数据 | 堆电压/电流, DCF 输出电压/电流 |
| 0x18FF03F0 | RX | 传感器 | 温度, 压力, 氢气浓度 |
| 0x18FF04F0 | RX | IO状态 | 阀门, 风扇, 故障码 |
| 0x18FF10A0 | TX | 控制命令 | 模式, 启停, 目标值 |

---

## 常见问题排查

### 1. `ModuleNotFoundError: No module named 'websockets'`
**原因**: Python 环境不匹配。系统可能有多个 Python，而 `websockets` 安装在另一个环境中。
**解决**: 使用 Anaconda 环境运行：`conda run -n base python server.py`

### 2. `Device open failed` 或 `CAN init failed`
**原因**: 
- USB-CAN 设备未连接
- **USB-CAN Tool 软件正在运行**（设备被独占）
- 驱动未正确安装
**解决**: 关闭所有其他 CAN 软件，重新插拔 USB，重试。

### 3. `OSError: [WinError 126] The specified module could not be found`
**原因**: DLL 加载失败。通常是因为 Python 是 64位，但 DLL 是 32位。
**解决**: 确保 `kerneldlls/` 文件夹下是 64位版本的 `ControlCAN.dll` 和 `usbcan.dll`。

### 4. 接收不到数据
**原因**: 波特率不匹配或终端电阻未连接。
**解决**: 检查 `config.py` 中的 `CAN_BITRATE` 是否与设备一致 (默认 250K)。检查 CAN 总线 H/L 接线。
