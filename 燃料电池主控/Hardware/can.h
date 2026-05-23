#ifndef __CAN_H
#define __CAN_H

#include "stm32f10x.h"
#include <stdio.h>
#include <stdint.h>

/*
 * 模块名称：CAN
 * 模块功能：提供 CAN1 的基础初始化、收发接口以及接收中断缓存。
 * 说明：
 * 1. 当前实现支持标准帧和扩展帧发送；
 * 2. 接收数据进入全局缓冲区，由上层按标识符进一步解析；
 * 3. 本模块默认使用 FIFO0 接收中断。
 */

/* CAN 接收缓冲区长度定义 */
#define CAN_RX_LEN  8   // 单帧最大数据长度为 8 字节

/*
 * 全局接收状态变量：
 * 1. CAN_RX_BUF 保存最近一次接收到的数据；
 * 2. CAN_RX_FLAG 表示是否收到新帧；
 * 3. CAN_RX_ID / CAN_RX_EXT_ID / CAN_RX_IDE 用于区分标准帧或扩展帧标识符。
 */
extern volatile uint8_t CAN_RX_BUF[CAN_RX_LEN]; // 最近一次接收到的 CAN 数据
extern volatile uint8_t CAN_RX_FLAG;            // 接收完成标志，1 表示收到新数据
extern volatile uint16_t CAN_RX_ID;             // 最近一次接收到的标准帧 ID
extern volatile uint32_t CAN_RX_EXT_ID;         // 最近一次接收到的扩展帧 ID
extern volatile uint8_t CAN_RX_IDE;             // 帧格式标志：0=标准帧，1=扩展帧

/**
  * @brief  CAN 硬件初始化
  * @retval 无
  */
void CAN_Hardware_Init(void);

/**
  * @brief  发送默认标准帧
  * @param  msg 数据缓冲区指针
  * @param  len 数据长度，最大 8 字节
  * @retval 0 表示成功，1 表示失败
  */
uint8_t CAN_Send_Msg(uint8_t* msg, uint8_t len);

/**
  * @brief  按指定标准帧 ID 发送数据
  * @param  std_id 标准帧 ID
  * @param  msg    数据缓冲区指针
  * @param  len    数据长度，最大 8 字节
  * @retval 0 表示成功，1 表示失败
  */
uint8_t CAN_Send_Msg_ID(uint16_t std_id, uint8_t* msg, uint8_t len);

/**
  * @brief  按指定扩展帧 ID 发送数据
  * @param  ext_id 扩展帧 ID
  * @param  msg    数据缓冲区指针
  * @param  len    数据长度，最大 8 字节
  * @retval 0 表示成功，1 表示失败
  */
uint8_t CAN_Send_ExtMsg_ID(uint32_t ext_id, uint8_t* msg, uint8_t len);

/**
  * @brief  轮询方式接收数据（当前主要使用中断接收）
  * @param  buf 数据缓存指针
  * @retval 接收到的数据长度或状态值，具体依赖实现
  */
uint8_t CAN_Receive_Msg(uint8_t *buf);

#endif
