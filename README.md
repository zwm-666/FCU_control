# FCU_control
# H2 FCU CAN Backend 使用说明

## 概述

这是一个 Python 后端服务，用于桥接 CAN 总线和前端 Dashboard。它使用 **ZLG 兼容的 USB-CAN 适配器**（ControlCAN.dll 接口）读取 CAN 数据，并通过 WebSocket 实时转发给前端。

## 系统要求

- Python 3.8+
- ZLG 兼容的 USB-CAN 适配器（支持 ControlCAN.dll）
- ControlCAN.dll 驱动文件

## 安装步骤

### 1. 安装 ZLG 驱动和 DLL

确保您有以下文件：
- `ControlCAN.dll` - ZLG CAN 设备驱动库
- 通常随 USB-CAN 工具软件一起提供

**放置 DLL 文件：**
- 将 `ControlCAN.dll` 复制到以下任一位置：
  - `backend/` 目录（与 server.py 同级）
  - `C:\Windows\System32\`
  - Python 安装目录的 `Scripts\` 文件夹

### 2. 安装 Python 依赖

```bash
cd backend
pip install -r requirements.txt
```

## 配置

编辑 `backend/config.py` 文件：

```python
# CAN 接口配置
CAN_INTERFACE = "zlgcan"         # ZLG 接口

# ZLG 设备类型
# USBCAN1 = 3, USBCAN2 = 4, USBCANFD = 41
ZLG_DEVICE_TYPE = 4              # 大多数设备使用 USBCAN2

# 设备索引（0 = 第一个设备）
ZLG_DEVICE_INDEX = 0


# CAN 通道（0 或 1，大多数 ZLG 设备有 2 个通道）
ZLG_CHANNEL = 0

# CAN 波特率（必须与 FCU 一致）
CAN_BITRATE = 250000
```

### 设备类型参考：
- **USBCAN1** (`3`) - 早期单通道设备
- **USBCAN2** (`4`) - 双通道设备（最常见）
- **USBCANFD** (`41`) - 支持 CAN-FD 的设备

## 运行

### 启动后端服务器

```bash
cd backend
python server.py
```

成功启动后，您会看到：

```
============================================================
H2 FCU CAN to WebSocket Bridge Server
============================================================
INFO - Initializing CAN interface: zlgcan
INFO - ZLG Device Type: 4, Index: 0, Channel: 0
INFO - Bitrate: 250000
INFO - ✓ CAN bus initialized successfully
INFO - Starting WebSocket server on 0.0.0.0:8765
INFO - ✓ WebSocket server started
INFO - Frontend should connect to: ws://localhost:8765
INFO - CAN receive loop started
```

### 启动前端

在另一个终端窗口：

```bash
npm run dev
```

然后在浏览器中访问 `http://localhost:3000`，点击右上角的"连接"按钮。

## CAN 协议

### 接收 (RX) 消息

| CAN ID       | 描述           | 数据内容 |
|--------------|----------------|----------|
| 0x18FF01F0   | 系统状态       | Heartbeat, State, FaultLevel |
| 0x18FF02F0   | 功率数据       | Stack Voltage/Current, DCF Voltage/Current |
| 0x18FF03F0   | 传感器数据     | Temperatures, Pressures, H2 Concentration |
| 0x18FF04F0   | IO 状态        | Valves, Fans, Fault Code |

### 发送 (TX) 消息

| CAN ID       | 描述           | 数据内容 |
|--------------|----------------|----------|
| 0x18FF10A0   | 控制命令       | Mode, Command, Setpoints |

## 故障排查

### 问题：CAN bus 初始化失败

**可能原因：**
1. ControlCAN.dll 未找到
2. ZLG USB-CAN 设备未连接
3. 设备类型或通道配置错误
4. 设备已被其他程序占用（如 USB-CAN Tool）

**解决方法：**
1. 确认 ControlCAN.dll 在正确位置
2. 检查 USB 连接
3. 使用 USB-CAN Tool 测试硬件是否正常
4. 关闭其他使用该设备的程序
5. 尝试不同的设备类型值（3, 4, 或 41）

### 问题：找不到 ControlCAN.dll

**错误信息：**
```
OSError: [WinError 126] 找不到指定的模块
```

**解决方法：**
1. 从 USB-CAN Tool 安装目录复制 `ControlCAN.dll`
2. 将 DLL 文件放到 `backend/` 目录
3. 或者将 DLL 复制到 `C:\Windows\System32\`
4. 确保 DLL 版本与您的设备兼容（32位 vs 64位）

### 问题：前端无法连接

**可能原因：**
1. 后端服务器未启动
2. WebSocket 端口被占用
3. 防火墙阻止

**解决方法：**
1. 确认后端服务器正在运行
2. 检查端口 8765 是否被占用
3. 添加防火墙例外

### 问题：无 CAN 数据

**可能原因：**
1. FCU 未上电
2. CAN 总线未连接
3. 波特率不匹配
4. 选择了错误的 CAN 通道


**解决方法：**
1. 检查 FCU 电源
2. 检查 CAN H/L 线路
3. 确认 config.py 中的波特率与 FCU 一致
4. 在 config.py 中尝试另一个通道（0 或 1）

## 调试

### 查看详细日志

修改 `server.py` 中的日志级别：

```python
logging.basicConfig(
    level=logging.DEBUG,  # 改为 DEBUG
    ...
)
```

### 使用 USB-CAN Tool 测试

在运行 Python 后端之前，可以使用 USB-CAN Tool 验证：
1. CAN 硬件是否正常
2. FCU 是否在发送数据
3. 波特率是否正确
4. 选择正确的通道

**注意：** 运行 Python 后端时，必须关闭 USB-CAN Tool，因为设备只能被一个程序打开。

## 架构

```
CAN Bus <---> ZLG USB-CAN <---> Python Backend <---> WebSocket <---> React Frontend
              (ControlCAN.dll)   (server.py)                         (App.tsx)
```

## 文件说明

- `server.py` - WebSocket 服务器主程序
- `can_protocol.py` - CAN 消息解析和生成
- `config.py` - 配置文件（ZLG 设备参数）
- `requirements.txt` - Python 依赖
- `ControlCAN.dll` - ZLG CAN 设备驱动库（需自行获取）

## 重要提示

> ⚠️ **ControlCAN.dll 位数匹配**
> 
> 确保 ControlCAN.dll 的位数（32位/64位）与您的 Python 版本匹配：
> - 64位 Python → 使用 64位 ControlCAN.dll
> - 32位 Python → 使用 32位 ControlCAN.dll
> 
> 检查 Python 位数：`python -c "import struct; print(struct.calcsize('P') * 8)"`

> 💡 **设备独占访问**
> 
> ZLG USB-CAN 设备一次只能被一个程序打开。运行 Python 后端时，必须关闭 USB-CAN Tool 或其他使用该设备的程序。

