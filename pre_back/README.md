# H2 FCU Modern Dashboard (氢燃料电池控制系统监控平台)

这是一个面向氢燃料电池控制单元（FCU）的现代化监控与调试平台。前端负责实时数据显示与控制交互，后端通过 Python 直接调用 `ControlCAN.dll` 与 CAN 硬件通信，再通过 WebSocket 将数据推送给前端。

---

## 1. 技术栈 (Tech Stack)

### 前端 (Frontend)
- **核心框架**: React 19、TypeScript、Vite 6
- **UI 组件**: TailwindCSS、Lucide React
- **图表**: Recharts
- **通讯方式**: 原生 WebSocket

### 后端 (Backend)
- **语言**: Python 3.9+
- **CAN 通讯**: `ctypes` + `ControlCAN.dll`
- **网络服务**: `websockets`、`asyncio`
- **运行模式**: `zlg`（真实硬件） / `virtual`（模拟数据）

---

## 2. 环境要求 (Prerequisites)

- **操作系统**: Windows 10/11
- **Node.js**: 16+，推荐 18 或 20
- **Python**: 3.9+
- **硬件**: ZLG USBCAN-2 或兼容 USB-CAN 适配器（真实设备模式下需要）

### 推荐启动方式
- **只有界面演示 / 无硬件**: 使用 `virtual` 模式，最省事。
- **联调真实 FCU**: 使用 `zlg` 模式，并确认驱动、接线、终端电阻都正确。

---

## 3. 硬件依赖与接线 (Hardware & Wiring)

### 驱动文件
后端依赖的 DLL 已放在 `backend/kerneldlls/`：
- `ControlCAN.dll`
- `usbcan.dll`

### 接线注意事项
1. **CAN_H** 连接 FCU 的 CAN_H
2. **CAN_L** 连接 FCU 的 CAN_L
3. **终端电阻**: 总线两端各接一个 **120Ω** 终端电阻
4. **共地**: 建议 USB-CAN 与 FCU 共地，减少共模干扰

---

## 4. 安装与启动 (Installation & Startup)

### 4.1 快速启动（推荐首次使用）

如果你只是先确认系统能跑起来：

1. 将 `backend/config.py` 中 `CAN_INTERFACE_TYPE` 设为 `"virtual"`
2. 启动后端：`python -u server.py`
3. 回到根目录启动前端：`npm install && npm run dev`
4. 浏览器打开 `http://localhost:3000`

这样即使没有 CAN 硬件，也可以看到界面和模拟数据流。

### 4.2 后端服务 (Backend)

后端负责与 CAN 硬件通讯，并将数据通过 WebSocket 推送给前端，默认地址为 `ws://localhost:8765`。

**步骤 1：安装依赖**
```powershell
cd backend
pip install -r requirements.txt
```

**步骤 2：检查配置**
编辑 `backend/config.py`：

| 配置项 | 说明 | 常见值 |
| :--- | :--- | :--- |
| `CAN_INTERFACE_TYPE` | CAN 接口模式 | `"virtual"` / `"zlg"` |
| `CAN_BITRATE` | CAN 波特率 | `250000` |
| `CANALYST_CHANNEL` | CAN 通道号 | `0` 或 `1` |
| `WEBSOCKET_PORT` | WebSocket 端口 | `8765` |

**步骤 3：启动后端**
```powershell
# 推荐：直接用当前环境的 Python 启动
python -u server.py

# 如果需要显式指定解释器，也可以使用完整路径
D:\anaconda\python.exe -u server.py
D:\python3.9\python.exe -u server.py
```

**启动成功后你应看到**：
- WebSocket 服务启动日志
- `Frontend should connect to: ws://localhost:8765`
- `virtual` 模式下会持续产生模拟数据

### 4.3 前端界面 (Frontend)

**步骤 1：安装依赖**
```powershell
cd ..
npm install
```

**步骤 2：启动开发服务器**
```powershell
npm run dev
```

**步骤 3：打开页面**
浏览器访问 `http://localhost:3000`。

**前端连接正常的表现**：
- 页面能正常打开
- WebSocket 状态显示为在线
- 图表和关键指标持续刷新

---

## 5. 常见故障排查 (Troubleshooting)

### Q1: 后端终端没有任何输出
**原因**: Python 输出缓冲或启动方式不合适。  
**解决**: 使用 `python -u server.py`，尽量不要用 `conda run` 启动后端。

### Q2: 提示 `VCI_OpenDevice failed`
**可能原因**:
1. USB-CAN 设备未正确连接
2. 设备驱动未正确安装
3. 设备被其他软件占用（如 CANTest）
4. `backend/config.py` 中设备参数不匹配

**建议处理顺序**: 先关掉其他 CAN 软件，再检查设备管理器、接线和配置。

### Q3: 前端显示在线，但数据不跳动
**可能原因**:
1. 后端在 `zlg` 模式，但总线上没有有效报文
2. 波特率与 FCU 不一致
3. 终端电阻或接线不正确
4. 没有硬件，却仍在使用 `zlg` 模式

**解决**: 无硬件时先切到 `virtual` 模式，确认前后端链路正常后再切回真实设备。

### Q4: 报 `ModuleNotFoundError: No module named 'websockets'`
**原因**: 当前 Python 环境缺少依赖。  
**解决**:
```powershell
cd backend
pip install -r requirements.txt
```
如果仍报错，请用 `where python` 确认当前使用的是哪个 Python。

### Q5: 浏览器打不开页面或端口不对
**原因**: 前端开发服务器默认端口为 `3000`。  
**解决**: 以终端输出为准，优先访问 `http://localhost:3000`。

### Q6: 页面能打开，但一直显示离线
**原因**: 前端能启动，但后端 WebSocket 未运行或端口不一致。  
**解决**:
1. 确认后端已启动
2. 确认日志中显示 `ws://localhost:8765`
3. 检查本机防火墙是否拦截本地端口

---

## 6. 其它功能

- **双击图表**: 重置缩放
- **手动模式**: 切换到手动后，右侧调试参数开关才可操作
- **日志输出**: 后端终端会打印发送（TX）和接收（RX）日志，便于联调
