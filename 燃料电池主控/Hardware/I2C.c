#include "I2C.h"
#include "delay.h" // 依赖延时函数，在 EEPROM 写入后等待内部写周期完成

/*
 * 模块名称：I2C
 * 模块功能：完成 I2C1 外设初始化以及 24C02 EEPROM 的基础读写操作。
 * 说明：
 * 1. SCL 使用 PB6，SDA 使用 PB7；
 * 2. 当前采用标准模式 100kHz，优先保证稳定性；
 * 3. 缓冲区读写基于单字节循环实现，逻辑简单但效率较低。
 */

/**
  * @brief  I2C1 初始化
  * @note   SCL=PB6，SDA=PB7
  * @retval 无
  */
void I2C_EE_Init(void)
{
    GPIO_InitTypeDef  GPIO_InitStructure;
    I2C_InitTypeDef   I2C_InitStructure;

    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);     // 使能 GPIOB 时钟
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_I2C1, ENABLE);      // 使能 I2C1 外设时钟

    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_6 | GPIO_Pin_7;    // 选择 PB6、PB7 作为 I2C 引脚
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;         // 配置 GPIO 输出速度
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_OD;           // I2C 总线需使用复用开漏输出
    GPIO_Init(GPIOB, &GPIO_InitStructure);                    // 初始化 SCL 与 SDA 引脚

    I2C_InitStructure.I2C_Mode = I2C_Mode_I2C;                // 配置为标准 I2C 模式
    I2C_InitStructure.I2C_DutyCycle = I2C_DutyCycle_2;        // 标准占空比配置
    I2C_InitStructure.I2C_OwnAddress1 = 0x00;                 // 主机模式下本机地址无实际意义
    I2C_InitStructure.I2C_Ack = I2C_Ack_Enable;               // 默认使能应答
    I2C_InitStructure.I2C_AcknowledgedAddress = I2C_AcknowledgedAddress_7bit; // 使用 7 位地址模式
    I2C_InitStructure.I2C_ClockSpeed = 100000;                // 设置总线速率为 100kHz
    I2C_Init(I2C1, &I2C_InitStructure);                       // 写入 I2C1 配置
    I2C_Cmd(I2C1, ENABLE);                                    // 使能 I2C1 外设

    I2C_AcknowledgeConfig(I2C1, ENABLE);                      // 额外确保 I2C 应答功能已打开
}

/**
  * @brief  向 EEPROM 写入 1 个字节
  * @param  WriteAddr 写入地址（0~255）
  * @param  Data      要写入的数据
  * @retval 无
  */
void I2C_EE_WriteByte(u8 WriteAddr, u8 Data)
{
    I2C_GenerateSTART(I2C1, ENABLE);                                          // 发送起始信号，开始一次 I2C 传输
    while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT));              // 等待主机模式选择完成

    I2C_Send7bitAddress(I2C1, EEPROM_ADDR, I2C_Direction_Transmitter);        // 发送 EEPROM 地址并指定写方向
    while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED)); // 等待从机应答并进入发送模式

    I2C_SendData(I2C1, WriteAddr);                                             // 发送 EEPROM 内部存储地址
    while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));          // 等待地址字节发送完成

    I2C_SendData(I2C1, Data);                                                  // 发送要写入的数据字节
    while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));          // 等待数据发送完成

    I2C_GenerateSTOP(I2C1, ENABLE);                                            // 发送停止信号，结束本次写操作

    Delay_ms(10);                                                              // 等待 EEPROM 完成内部写周期，避免下一次访问无应答
}

/**
  * @brief  从 EEPROM 读取 1 个字节
  * @param  ReadAddr 要读取的地址
  * @retval 读取到的数据
  */
u8 I2C_EE_ReadByte(u8 ReadAddr)
{
    u8 temp = 0;

    while (I2C_GetFlagStatus(I2C1, I2C_FLAG_BUSY));                            // 等待 I2C 总线空闲，防止与上一次传输冲突

    I2C_GenerateSTART(I2C1, ENABLE);                                           // 发送第一次起始信号
    while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT));               // 等待进入主机模式

    I2C_Send7bitAddress(I2C1, EEPROM_ADDR, I2C_Direction_Transmitter);         // 先以写方向发送器件地址，用于指定内部读地址
    while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED)); // 等待进入发送模式

    I2C_SendData(I2C1, ReadAddr);                                              // 写入要读取的 EEPROM 内部地址
    while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));          // 等待地址发送完成

    I2C_GenerateSTART(I2C1, ENABLE);                                           // 发送重复起始信号，切换为读操作
    while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT));               // 等待再次进入主机模式

    I2C_Send7bitAddress(I2C1, EEPROM_ADDR, I2C_Direction_Receiver);            // 以读方向发送器件地址
    while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_RECEIVER_MODE_SELECTED));    // 等待进入接收模式

    I2C_AcknowledgeConfig(I2C1, DISABLE);                                      // 只读取 1 字节，因此提前关闭应答
    I2C_GenerateSTOP(I2C1, ENABLE);                                            // 在接收该字节后发送停止信号

    while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_RECEIVED));             // 等待接收数据完成
    temp = I2C_ReceiveData(I2C1);                                              // 读取接收到的数据字节

    I2C_AcknowledgeConfig(I2C1, ENABLE);                                       // 恢复应答使能，便于后续继续通信

    return temp;                                                               // 返回读取结果
}

/**
  * @brief  连续写入缓冲区数据
  * @note   当前实现未做分页写优化，而是逐字节循环写入
  * @param  pBuffer        数据缓冲区指针
  * @param  WriteAddr      起始写地址
  * @param  NumByteToWrite 要写入的字节数
  * @retval 无
  */
void I2C_EE_WriteBuffer(u8* pBuffer, u8 WriteAddr, u16 NumByteToWrite)
{
    u16 i;
    for (i = 0; i < NumByteToWrite; i++)                       // 逐字节写入，逻辑直观但效率不高
    {
        I2C_EE_WriteByte(WriteAddr + i, pBuffer[i]);           // 当前地址加偏移写入 1 字节数据
    }
}

/**
  * @brief  连续读取缓冲区数据
  * @param  pBuffer       数据缓冲区指针
  * @param  ReadAddr      起始读地址
  * @param  NumByteToRead 要读取的字节数
  * @retval 无
  */
void I2C_EE_ReadBuffer(u8* pBuffer, u8 ReadAddr, u16 NumByteToRead)
{
    u16 i;
    for (i = 0; i < NumByteToRead; i++)                        // 逐字节顺序读取，简化读流程
    {
        pBuffer[i] = I2C_EE_ReadByte(ReadAddr + i);            // 从 EEPROM 连续地址读取数据到缓冲区
    }
}

/**
  * @brief  检查 EEPROM 是否存在
  * @retval 0 表示正常，1 表示异常
  */
u8 I2C_EE_Check(void)
{
    u8 temp;

    temp = I2C_EE_ReadByte(255);                               // 先读取末地址当前内容，用于后续校验
    if (temp == 0x55) return 0;                                // 若本来就是 0x55，则说明读写通路正常

    I2C_EE_WriteByte(255, 0x55);                               // 若不是 0x55，则尝试写入测试值 0x55
    temp = I2C_EE_ReadByte(255);                               // 再次读取该地址进行校验
    if (temp == 0x55) return 0;                                // 读回正确则说明 EEPROM 正常

    return 1;                                                  // 否则认为 EEPROM 检测失败
}
