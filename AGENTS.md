# AGENTS.md - Antigravity 代理开发指南

此文件为在 **H2 FCU Modern Dashboard** 仓库中运行的 AI 代理提供关键信息。

## 1. 项目概览
氢燃料电池单元 (FCU) 的实时监控与控制仪表盘。
- **前端**: React 19, TypeScript, Vite 6, TailwindCSS 4, Recharts。
- **后端**: Python 3.9+, WebSocket 服务器, ZLG CAN 驱动 (通过 `ctypes` 调用 `ControlCAN.dll`)。

## 2. 核心语言规则 (CRITICAL)
- **对话语言**: 以后所有与用户的对话必须使用 **中文**。
- **文档语言**: 所有生成的 Markdown (.md) 文件必须使用 **中文**。
- **代码例外**: 代码实现（变量名、注释、逻辑）保持使用 **英文**。

## 3. 关键命令

### 前端 (根目录)
| 操作 | 命令 |
| :--- | :--- |
| 安装依赖 | `npm install` |
| 启动开发服务器 | `npm run dev` |
| 构建项目 | `npm run build` |
| 预览构建 | `npm run preview` |

### 后端 (backend/ 目录)
| 操作 | 命令 |
| :--- | :--- |
| 安装依赖 | `pip install websockets` |
| 启动服务器 | `python -u server.py` |
| 模拟模式 | 在 `backend/config.py` 中设置 `CAN_INTERFACE_TYPE = "virtual"` |

## 4. 代码风格指南

### 4.1 TypeScript / React
- **缩进**: 4 个空格。
- **分号**: 强制使用。
- **导入**: 
  - 使用相对路径。
  - 允许使用 `@/` 别名指向根目录。
  - 排序: React 相关 -> 外部库 -> 本地类型/服务 -> 组件 -> 样式。
- **命名规范**:
  - 组件: `PascalCase` (如 `MetricCard.tsx`)。
  - 变量/函数: `camelCase` (如 `isSystemRunning`)。
  - 常量: `UPPER_SNAKE_CASE` (如 `HISTORY_LENGTH`)。
  - 接口/枚举: `PascalCase`。
- **类型安全**:
  - 严禁使用 `any`，优先使用显式类型。
  - 系统状态和故障等级使用枚举 (见 `types.ts`)。
- **组件**:
  - 使用 Hooks 的函数组件。
  - 图标统一使用 `lucide-react`。

### 4.2 Python (后端)
- **缩进**: 4 个空格。
- **命名规范**:
  - 类: `PascalCase`。
  - 函数/变量: `snake_case`。
- **类型提示**: 必须包含函数签名类型提示。

## 5. 状态管理模式
- **局部状态**: UI 切换使用 `useState`。
- **全局状态**: 由 `App.tsx` 管理并通过 Props 下发。
- **WebSocket 同步**: `MachineState` 接收后端推送，`ControlState` 向后端发送。

## 6. CAN 协议实现
- **协议定义**: 同时存在于 `services/canProtocol.ts` (前端) 和 `backend/can_protocol1.py` (后端)。
- **字节序**: 小端序 (Little-endian)。
- **核心 ID**: `0x18FF01F0` (心跳), `0x18FF02F0` (功率), `0x18FF03F0` (传感器), `0x18FF10A0` (控制)。

## 7. 代理须知 (Agent Tips)
- **性能**: 前端接收频率为 10Hz，避免在渲染主循环中进行高能耗计算。
- **安全**: 系统涉及硬件控制，所有“写入”操作（阀门、加热器）必须包含安全检查或用户确认。
- **环境**: 后端仅支持 Windows (需加载 `.dll`)。
