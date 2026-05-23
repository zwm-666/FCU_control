#include "can.h"

/*
 * 模块名称：CAN
 * 模块功能：完成 CAN1 的硬件初始化、标准帧/扩展帧发送和接收中断缓存。
 * 说明：
 * 1. 当前波特率配置为 250kbps；
 * 2. 过滤器配置为全接收，便于上层自行筛选报文；
 * 3. 接收中断将最新一帧写入全局变量，供上层读取。
 */

/* CAN 接收相关全局变量 */
volatile uint8_t CAN_RX_BUF[CAN_RX_LEN]; // 接收数据缓冲区
volatile uint8_t CAN_RX_FLAG = 0;        // 接收完成标志
volatile uint16_t CAN_RX_ID = 0;         // 最近一次标准帧 ID
volatile uint32_t CAN_RX_EXT_ID = 0;     // 最近一次扩展帧 ID
volatile uint8_t CAN_RX_IDE = 0;         // 帧类型标志，0=标准帧，1=扩展帧

/**
  * @brief  CAN 硬件初始化
  * @note   波特率：250kbps
  * @note   硬件：STM32F103C8T6，PA11(RX)，PA12(TX)
  * @retval 无
  */
void CAN_Hardware_Init(void)
{
    GPIO_InitTypeDef        GPIO_InitStructure;
    CAN_InitTypeDef         CAN_InitStructure;
    CAN_FilterInitTypeDef   CAN_FilterInitStructure;
    NVIC_InitTypeDef        NVIC_InitStructure;

    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);      // 使能 GPIOA 时钟
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_CAN1, ENABLE);       // 使能 CAN1 外设时钟

    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_11;                 // PA11 用作 CAN_RX
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU;              // 输入上拉模式
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;          // 配置 GPIO 速度
    GPIO_Init(GPIOA, &GPIO_InitStructure);                     // 初始化 RX 引脚

    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_12;                 // PA12 用作 CAN_TX
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;            // 复用推挽输出
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;          // 配置 GPIO 速度
    GPIO_Init(GPIOA, &GPIO_InitStructure);                     // 初始化 TX 引脚

    CAN_DeInit(CAN1);                                          // 先复位 CAN1，清除旧配置
    CAN_StructInit(&CAN_InitStructure);                        // 用默认值初始化结构体

    CAN_InitStructure.CAN_TTCM = DISABLE;                      // 关闭时间触发通信模式
    CAN_InitStructure.CAN_ABOM = DISABLE;                      // 关闭自动离线管理
    CAN_InitStructure.CAN_AWUM = DISABLE;                      // 关闭自动唤醒功能
    CAN_InitStructure.CAN_NART = DISABLE;                      // 允许自动重发
    CAN_InitStructure.CAN_RFLM = DISABLE;                      // FIFO 满时允许新报文覆盖旧报文
    CAN_InitStructure.CAN_TXFP = DISABLE;                      // 发送优先级按标识符排序
    CAN_InitStructure.CAN_Mode = CAN_Mode_Normal;              // 设置为正常工作模式

    /*
     * 波特率计算：
     * APB1 = 36MHz
     * Baud = 36M / (Prescaler * (1 + BS1 + BS2))
     * 当前配置：Prescaler=9，BS1=12，BS2=3，SJW=1
     * 结果：36M / (9 * 16) = 250kHz
     */
    CAN_InitStructure.CAN_SJW = CAN_SJW_1tq;                  // 同步跳转宽度 1 个时间片
    CAN_InitStructure.CAN_BS1 = CAN_BS1_12tq;                 // 位段 1 为 12 个时间片
    CAN_InitStructure.CAN_BS2 = CAN_BS2_3tq;                  // 位段 2 为 3 个时间片
    CAN_InitStructure.CAN_Prescaler = 9;                      // 预分频系数
    CAN_Init(CAN1, &CAN_InitStructure);                       // 写入 CAN 初始化参数

    CAN_FilterInitStructure.CAN_FilterNumber = 0;             // 使用过滤器 0
    CAN_FilterInitStructure.CAN_FilterMode = CAN_FilterMode_IdMask;   // 标识符掩码模式
    CAN_FilterInitStructure.CAN_FilterScale = CAN_FilterScale_32bit;  // 32 位过滤器
    CAN_FilterInitStructure.CAN_FilterIdHigh = 0x0000;        // 过滤器 ID 高位清零
    CAN_FilterInitStructure.CAN_FilterIdLow = 0x0000;         // 过滤器 ID 低位清零
    CAN_FilterInitStructure.CAN_FilterMaskIdHigh = 0x0000;    // 掩码清零，表示不过滤任何 ID
    CAN_FilterInitStructure.CAN_FilterMaskIdLow = 0x0000;     // 掩码清零，表示全部接收
    CAN_FilterInitStructure.CAN_FilterFIFOAssignment = CAN_Filter_FIFO0; // 接收数据放入 FIFO0
    CAN_FilterInitStructure.CAN_FilterActivation = ENABLE;    // 使能过滤器
    CAN_FilterInit(&CAN_FilterInitStructure);                 // 应用过滤器配置

    CAN_ITConfig(CAN1, CAN_IT_FMP0, ENABLE);                  // 使能 FIFO0 报文挂起中断

    NVIC_InitStructure.NVIC_IRQChannel = USB_LP_CAN1_RX0_IRQn; // CAN1 RX0 与 USB_LP 共用中断向量
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 1;  // 设置抢占优先级
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;         // 设置响应优先级
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;            // 使能中断通道
    NVIC_Init(&NVIC_InitStructure);                            // 写入 NVIC 配置
}

/**
  * @brief  发送一个默认标准帧
  * @param  msg 数据指针
  * @param  len 数据长度，最大 8 字节
  * @retval 0 表示成功，1 表示失败
  */
uint8_t CAN_Send_Msg(uint8_t* msg, uint8_t len)
{
    return CAN_Send_Msg_ID(0x12, msg, len);                   // 默认使用标准帧 ID 0x12 发送
}

/**
  * @brief  按指定标准帧 ID 发送数据
  * @param  std_id 标准帧 ID
  * @param  msg    数据指针
  * @param  len    数据长度，最大 8 字节
  * @retval 0 表示成功，1 表示失败
  */
uint8_t CAN_Send_Msg_ID(uint16_t std_id, uint8_t* msg, uint8_t len)
{
    uint8_t mbox;
    uint16_t i = 0;
    uint8_t tx_status;
    CanTxMsg TxMessage;

    if (len > 8) {
        len = 8;                                              // CAN 单帧最多只允许 8 字节数据
    }

    TxMessage.StdId = std_id;                                 // 设置标准帧 ID
    TxMessage.ExtId = 0x12;                                   // 扩展 ID 字段在标准帧模式下不生效
    TxMessage.IDE = CAN_Id_Standard;                          // 指定发送标准帧
    TxMessage.RTR = CAN_RTR_Data;                             // 指定为数据帧
    TxMessage.DLC = len;                                      // 设置数据长度码

    for (i = 0; i < len; i++)
        TxMessage.Data[i] = msg[i];                           // 拷贝待发送数据到发送报文结构体

    mbox = CAN_Transmit(CAN1, &TxMessage);                    // 请求发送，并返回使用的邮箱号
    if (mbox == CAN_NO_MB) {
        return 1;                                             // 没有空闲邮箱时直接返回失败
    }

    tx_status = CAN_TransmitStatus(CAN1, mbox);               // 读取当前邮箱发送状态
    while ((tx_status == CAN_TxStatus_Pending) && (i < 0xFFFF))
    {
        i++;                                                  // 通过计数器实现简单超时等待
        tx_status = CAN_TransmitStatus(CAN1, mbox);           // 轮询发送状态直到完成或超时
    }

    if ((i >= 0xFFFF) || (tx_status != CAN_TxStatus_Ok)) {
        return 1;                                             // 超时或发送状态异常则返回失败
    }
    return 0;                                                 // 发送成功
}

/**
  * @brief  按指定扩展帧 ID 发送数据
  * @param  ext_id 扩展帧 ID
  * @param  msg    数据指针
  * @param  len    数据长度，最大 8 字节
  * @retval 0 表示成功，1 表示失败
  */
uint8_t CAN_Send_ExtMsg_ID(uint32_t ext_id, uint8_t* msg, uint8_t len)
{
    uint8_t mbox;
    uint16_t i = 0;
    uint8_t tx_status;
    CanTxMsg TxMessage;

    if (len > 8) {
        len = 8;                                              // 限制单帧数据长度不超过 8 字节
    }

    TxMessage.StdId = 0;                                      // 扩展帧发送时标准 ID 字段不使用
    TxMessage.ExtId = ext_id & 0x1FFFFFFF;                    // 仅保留 29 位有效扩展 ID
    TxMessage.IDE = CAN_Id_Extended;                          // 指定为扩展帧格式
    TxMessage.RTR = CAN_RTR_Data;                             // 指定为数据帧
    TxMessage.DLC = len;                                      // 设置数据长度

    for (i = 0; i < len; i++) {
        TxMessage.Data[i] = msg[i];                           // 将待发送数据写入报文结构体
    }

    mbox = CAN_Transmit(CAN1, &TxMessage);                    // 请求发送扩展帧
    if (mbox == CAN_NO_MB) {
        return 1;                                             // 无可用邮箱则发送失败
    }

    tx_status = CAN_TransmitStatus(CAN1, mbox);               // 获取发送状态
    while ((tx_status == CAN_TxStatus_Pending) && (i < 0xFFFF)) {
        i++;                                                  // 简单超时计数
        tx_status = CAN_TransmitStatus(CAN1, mbox);           // 持续轮询直到发送完成或超时
    }

    if ((i >= 0xFFFF) || (tx_status != CAN_TxStatus_Ok)) {
        return 1;                                             // 超时或发送失败
    }
    return 0;                                                 // 发送成功
}

/**
  * @brief  USB_LP_CAN1_RX0 中断服务函数
  * @note   STM32F103C8T6 中 USB 低优先级与 CAN1 RX0 共用该中断入口
  * @retval 无
  */
void USB_LP_CAN1_RX0_IRQHandler(void)
{
    CanRxMsg RxMessage;

    if (CAN_GetITStatus(CAN1, CAN_IT_FMP0) != RESET)          // 判断 FIFO0 是否有挂起接收中断
    {
        CAN_Receive(CAN1, CAN_FIFO0, &RxMessage);             // 从 FIFO0 取出一帧接收到的 CAN 报文

        uint8_t len = RxMessage.DLC;                          // 读取该帧的数据长度
        CAN_RX_ID = RxMessage.StdId;                          // 保存标准帧 ID
        CAN_RX_EXT_ID = RxMessage.ExtId;                      // 保存扩展帧 ID
        CAN_RX_IDE = (RxMessage.IDE == CAN_Id_Extended) ? 1U : 0U; // 保存帧格式类型
        if (len > 8) len = 8;                                 // 再次防护，避免长度异常越界

        for (uint8_t i = 0; i < len; i++)
        {
            CAN_RX_BUF[i] = RxMessage.Data[i];                // 将收到的数据逐字节保存到全局缓冲区
        }

        CAN_RX_FLAG = 1;                                      // 置位接收标志，通知上层有新数据

        CAN_ClearITPendingBit(CAN1, CAN_IT_FMP0);             // 清除 FIFO0 接收中断挂起标志
    }
}
