# H2 FCU Modern Dashboard (氢燃料电池控制系统监控平台)

这是一个现代化的氢燃料电池控制单元 (FCU) 监控与调试平台。项目采用前后端分离架构，前端使用 React 实现响应式数据可视化与控制，后端使用 Python 直接调用 ZLG CAN 驱动 (ControlCAN.dll) 实现高性能的 CAN 总线通讯。

---

## 1. 技术栈 (Tech Stack)

### 前端 (Frontend)
- **核心框架**: React 19, TypeScript, Vite 6
- **UI 组件**: TailwindCSS (样式), Lucide React (图标)
- **图表**: Recharts (实时数据波形)
- **通讯**: WebSocket (原生 API)

### 后端 (Backend)
- **语言**: Python 3.9+ (推荐使用 Anaconda)
- **CAN 驱动**: `ctypes` 直接调用 `ControlCAN.dll` (ZLG 官方驱动库)
- **通讯**: `websockets`, `asyncio`
- **特点**: 无需 `python-can` 或 `zlgcan` 库，直接操作 DLL，延迟更低，兼容性更好。

---

## 2. 环境要求 (Prerequisites)

- **操作系统**: Windows 10/11 (因为涉及到 ControlCAN.dll 调用，必须在 Windows 下运行)
- **Node.js**: v16+ (推荐 v18 或 v20)
- **Python**: 3.8+ (强烈推荐使用 Anaconda 环境管理)
- **硬件**: ZLG USBCAN-2 或兼容的 USB-CAN 适配器 (如周立功、创芯科技等)

---

## 3. 硬件依赖与接线 (Hardware & Wiring)

### 驱动文件
后端依赖厂商提供的 DLL 文件，已放置在 `backend/kerneldlls/` 目录下：
- `ControlCAN.dll` (核心驱动)
- `usbcan.dll` (依赖库)

### 硬件接线
1. **CAN_H** 接 FCU 的 CAN_H
2. **CAN_L** 接 FCU 的 CAN_L
3. **终端电阻**: 务必在总线两端各并联一个 **120Ω 终端电阻**，否则通讯极不稳定。
4. **共地**: 建议将 USB-CAN 的 GND 与 FCU 的 GND 连接，防止共模干扰。

---

## 4. 安装与启动 (Installation & Startup)

### 4.1 后端服务 (Backend)

后端负责与 CAN 硬件通讯，并通过 WebSocket 将数据转发给前端。

**步骤 1: 准备环境**
```powershell
cd backend
# 如果使用 Anaconda (推荐)
conda activate base
# 安装依赖 (仅需 websockets)
pip install websockets
```

**步骤 2: 配置 (可选)**
编辑 `backend/config.py` 文件：
- `CAN_INTERFACE_TYPE`: 设置为 `"zlg"` (真实硬件) 或 `"virtual"` (模拟测试)
- `CAN_BITRATE`: 设置波特率 (默认 250000)
- `CANALYST_CHANNEL`: 设置 CAN 通道 (0 或 1)

**步骤 3: 启动服务器**
```powershell
# 方法 A:直接使用 python (推荐，无缓冲延迟)
python -u server.py

# 方法 B: 使用 Anaconda python 全路径 (解决找不到模块问题)
D:\anaconda\python.exe -u server.py
```
> **注意**: 启动成功后，您应该能看到 `H2 FCU CAN to WebSocket Bridge Server` 的欢迎信息。

### 4.2 前端界面 (Frontend)

**步骤 1: 安装依赖**
```powershell
# 回到项目根目录
cd .. 
npm install
```

**步骤 2: 启动开发服务器**
```powershell
npm run dev
```

**步骤 3: 访问**
打开浏览器访问终端显示的地址 (通常是 `http://localhost:5173` 或 `http://localhost:3000`)。

---

## 5. 常见故障排查 (Troubleshooting)

### Q1: 后端终端没有任何输出 (No Output)
**原因**: Python 的输出缓冲导致。
**解决**: 启动时加上 `-u` 参数，例如 `python -u server.py`。我们在代码中也加入了 `flush=True` 来强制刷新。另外尽量避免使用 `conda run` 命令直接运行，因为它会吞掉部分输出。

### Q2: 提示 "VCI_OpenDevice failed"
**原因**:
1. USB-CAN 设备未插好。
2. 驱动未正确安装 (请在设备管理器确认是否有 "ZLG USBCAN" 设备)。
3. 设备正在被其他软件 (如 CANTest) 占用。CAN 设备是独占的，**必须关闭其他 CAN 软件**。
4. `config.py` 中的设备类型 (`CANALYST_DEVICE_TYPE`) 配置错误。

### Q3: 前端显示 "在线" 但数据不跳动
**原因**:
1. 后端是 `zlg` 模式，但没有 CAN 报文进来。检查接线、终端电阻、波特率。
2. 如果你在没有硬件的情况下测试，请将 `backend/config.py` 中的 `CAN_INTERFACE_TYPE` 改为 `"virtual"`，这样后端会生成模拟数据。

### Q4: 报 "ModuleNotFoundError: No module named 'websockets'"
**原因**: 当前 Python 环境未安装依赖。
**解决**: 运行 `pip install websockets`。如果已安装但仍报错，请检查你是运行在哪个 Python 环境 (`where python`)。

---

## 6. 其它功能

- **双击图表**: 重置缩放。
- **手动模式**: 只有在前端点击“手动”按钮切换模式后，右侧的调试参数开关才可操作。
- **日志**: 后端会在终端打印所有发送 (TX) 的控制指令和接收 (RX) 的心跳包。