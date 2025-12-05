# 如何运行后端

由于您使用 Anaconda 环境，请使用以下命令启动后端服务器：

```bash
cd backend
D:\anaconda\python.exe server.py
```

或者，将 Anaconda Python 添加到 PATH 环境变量后直接使用 `python server.py`。

## 运行前准备

1. 将 `ControlCAN.dll` 放到 `backend/` 目录
2. 连接 ZLG USB-CAN 设备
3. 关闭 USB-CAN Tool（设备独占访问）
