# Repository Guidelines

## Project Structure & Module Organization
- 前端入口在 `index.tsx`，主界面编排集中于 `App.tsx`，共享类型放在 `types.ts`。
- 可复用界面组件位于 `components/`，通信与协议逻辑位于 `services/`，例如 `services/websocketService.ts` 与 `services/canProtocol.ts`。
- 后端位于 `backend/`：`server.py` 提供 WebSocket 桥接，`config.py` 管理运行模式，`can_protocol*.py` 维护 CAN 协议，`kerneldlls/` 存放 Windows 驱动 DLL。
- 构建辅助脚本在 `scripts/`。当前没有独立 `tests/` 目录，已有后端校验脚本为 `backend/test_model_accuracy.py`。

## Build, Test, and Development Commands
- `npm install`：安装前端依赖。
- `npm run dev`：启动 Vite 开发服务器，默认监听 `http://localhost:3000`。
- `npm run build`：执行生产构建；提交前优先运行它验证 TypeScript 与打包配置。
- `npm run preview`：本地预览构建结果。
- `npm run test:tailwind`：检查 Tailwind 构建链路是否正常。
- `cd backend && pip install -r requirements.txt && python -u server.py`：启动后端；无硬件时将 `backend/config.py` 中 `CAN_INTERFACE_TYPE` 设为 `"virtual"`。

## Coding Style & Naming Conventions
- TypeScript / React 使用 4 空格缩进、强制分号、严格类型检查；避免 `any`。
- 组件文件使用 `PascalCase`，变量与函数使用 `camelCase`，常量使用 `UPPER_SNAKE_CASE`。
- Python 使用 4 空格缩进；类名 `PascalCase`，函数与变量 `snake_case`，函数签名保留类型提示。
- 导入顺序保持为 React → 第三方库 → 本地模块；优先相对路径，也可使用 `@/` 指向仓库根目录。

## Testing Guidelines
- 当前前端没有统一测试框架；UI 变更至少运行 `npm run build` 和 `npm run test:tailwind`。
- 后端测试或验证脚本沿用 `test_*.py` 命名，并尽量与相关模块放在同一目录。
- 新增测试时优先覆盖 CAN 协议解析、WebSocket 数据流和诊断逻辑，避免只做静态快照式断言。

## Commit & Pull Request Guidelines
- 现有提交同时存在中文和英文主题，如 `Add README to repo root`、`页面修改`；请保持祈使语气、主题简短、范围明确，必要时使用 `feat:`、`fix:`、`docs:`。
- Pull Request 应包含：变更摘要、影响范围（前端/后端/协议）、关联问题，以及 UI 截图或关键终端日志。
- 若修改 `backend/config.py`、CAN ID、控制确认流程或硬件依赖，请在描述中单独标注风险与验证方式。

## Security & Configuration Tips
- 后端依赖 `ControlCAN.dll`，仅支持 Windows；不要提交本机专用路径、凭据或未说明的数据文件。
- 所有控制写入都视为安全敏感操作；涉及阀门、加热器或模式切换时，界面与后端都应保留确认或保护逻辑。
- 避免提交生成物，例如 `backend/__pycache__/`；若必须更新脚本输出，请在 PR 中说明来源与用途。
