# H2 FCU 后端 RS485 协议说明

本文档用于把单片机端原 CAN 通信改为 RS485 串口通信。当前后端匹配主控固件中的普通 UART 二进制帧：保留原 CAN 扩展帧 ID 和 8 字节 payload，RS485 只负责增加帧头、长度和 1 字节累加校验。

## 1. 串口参数

默认配置在 `config.py`：

| 参数 | 默认值 |
|---|---:|
| 串口号 | `COM3` |
| 波特率 | `9600` |
| 数据位 | 8 |
| 校验 | None |
| 停止位 | 1 |
| 超时 | 10 ms |

后端没有 RS485 方向控制脚，USB-RS485 模块通常自动收发切换。若单片机使用半双工芯片，如 MAX485，需要在发送前拉高 DE/RE，发送完成后切回接收。

## 2. 帧格式

固定 16 字节：

```text
AA 55 ID0 ID1 ID2 ID3 LEN DATA0 DATA1 DATA2 DATA3 DATA4 DATA5 DATA6 DATA7 SUM
```

| 字段 | 长度 | 说明 |
|---|---:|---|
| Header | 2 | 固定 `AA 55` |
| ID | 4 | 原 CAN 扩展帧 ID，小端序 |
| Len | 1 | Payload 字节数，当前固定为 `08` |
| Payload | 8 | 与原 CAN 报文 8 字节数据完全一致 |
| SUM | 1 | 从 `ID0` 到 `DATA7` 所有字节累加，取低 8 位 |

后端仍保留旧 `AA 55 TYPE LEN PAYLOAD CRC16` 帧的接收兼容，但发送给 MCU 的控制帧使用当前 16 字节 CAN-ID 格式。

## 3. 帧 ID 映射

| CAN ID | 方向 | 含义 |
|---:|---|---|
| `0x18FF01F0` | MCU -> 后端 | 系统状态 |
| `0x18FF02F0` | MCU -> 后端 | 电源数据 |
| `0x18FF03F0` | MCU -> 后端 | 传感器数据 |
| `0x18FF04F0` | MCU -> 后端 | IO 状态 |
| `0x18FF10A0` | 后端 -> MCU | 控制命令 |

## 4. MCU 发送帧 Payload

### ID `0x18FF01F0` 系统状态

| Byte | 说明 |
|---:|---|
| 0 | 心跳计数器，0-255 循环 |
| 1 | bit0-1：系统状态；bit2-3：故障等级 |
| 2-7 | 预留，填 0 |

系统状态建议：`0=关机完成`，`1=关机中`，`2=运行`，`3=急停/故障停机`。

### ID `0x18FF02F0` 电源数据

| Byte | 类型 | 说明 |
|---:|---|---|
| 0-1 | uint16 little-endian | 电堆电压，单位 0.01 V |
| 2-3 | uint16 little-endian | 电堆电流，单位 0.1 A |
| 4-5 | uint16 little-endian | DCF 输出电压，单位 0.01 V |
| 6-7 | uint16 little-endian | DCF 输出电流，单位 0.1 A |

### ID `0x18FF03F0` 传感器数据

| Byte | 类型 | 说明 |
|---:|---|---|
| 0-1 | uint16 big-endian | 电堆温度，真实值 = raw * 0.1 - 40，单位 C |
| 2-3 | uint16 big-endian | 氢瓶压力，单位 0.01 MPa |
| 4-5 | uint16 big-endian | 进氢压力，单位 0.01 MPa |
| 6 | uint8 | 氢气浓度，单位 0.5 %vol |
| 7 | uint8 | 预留，填 0 |

注意：当前后端解析器这里沿用了原项目实现，`0x03` 使用 big-endian；`0x02` 使用 little-endian。单片机端需按表格分别处理。

### ID `0x18FF04F0` IO 状态

| Byte | 类型 | 说明 |
|---:|---|---|
| 0 | bit flags | bit0=进氢阀，bit1=排氢阀，bit3=加热器，bit4=风扇1，bit5=风扇2 |
| 1 | uint8 | 风扇1 占空比，0-100 |
| 2-3 | uint16 big-endian | DCF MOS 温度，真实值 = raw * 0.1 - 40，单位 C |
| 4-5 | uint16 big-endian | 故障码 |
| 6-7 | uint8 | 预留，填 0 |

## 5. MCU 接收控制帧 Payload

后端发送 CAN ID `0x18FF10A0`。

| Byte | 说明 |
|---:|---|
| 0 | bit0-1：模式，`0=手动`，`1=自动`；bit2-4：命令，`0=STOP`，`1=START` |
| 1 | 手动控制标志：bit0=进氢阀，bit1=排氢阀，bit2=加热器，bit3=风扇1，bit4=风扇2 |
| 2 | 风扇1 目标占空比，0-100 |
| 3-4 | DCF 目标电压，uint16 little-endian，单位 0.1 V |
| 5-6 | DCF 目标电流，uint16 little-endian，单位 0.1 A |
| 7 | 预留 |

## 6. 示例帧

系统状态示例：心跳 `0x12`，运行状态 `0x02`。

```text
AA 55 F0 01 FF 18 08 12 02 00 00 00 00 00 00 24
```

控制命令示例：自动 START，手动标志 `0x19`，风扇 80%，目标电压 48.0 V，目标电流 10.0 A。

```text
AA 55 A0 10 FF 18 08 04 19 50 E0 01 64 00 00 81
```

## 7. 单片机代码改造建议

1. 保留现有 CAN payload 组包函数，把 `CAN_Send_ExtMsg_ID(id, data, 8)` 替换为 `RS485_Send_Frame(id, data)`。
2. 串口接收中断只负责把字节放入环形缓冲区，主循环里解析完整帧并校验 `SUM`。
3. 收到 ID `0x18FF10A0` 且校验正确后，再按原控制 CAN payload 的逻辑更新运行模式、启动停止、阀、风扇和 DCF 目标值。
5. 若使用 MAX485：发送帧前 `DE=1`，等待 USART TC 发送完成后 `DE=0`。

## 8. 后端配置

后端默认改为：

```python
CAN_INTERFACE_TYPE = "rs485"
RS485_PORT = "COM3"
RS485_BAUDRATE = 9600
```

把 `COM3` 改成 Windows 设备管理器里 USB-RS485 的实际串口号即可。
