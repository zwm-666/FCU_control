#ifndef __I2C_H
#define __I2C_H

#include "stm32f10x.h"

/*
 * 模块名称：I2C
 * 模块功能：提供基于 I2C1 的 EEPROM（24C02）读写接口。
 * 说明：
 * 1. 当前默认器件地址配置为 A0=A1=A2=0；
 * 2. 提供单字节读写、缓冲区读写和器件检测函数；
 * 3. 上层可使用本模块存取运行参数或掉电保存数据。
 */

/* EEPROM 硬件参数定义 */
#define EEPROM_ADDR         0xA0    // 24C02 器件地址（A0=A1=A2=0）
#define EEPROM_PAGE_SIZE    8       // 24C02 每页大小为 8 字节

/**
  * @brief  I2C1 初始化
  * @retval 无
  */
void I2C_EE_Init(void);

/**
  * @brief  向指定地址写入 1 个字节
  * @param  WriteAddr EEPROM 内部地址
  * @param  Data      要写入的数据
  * @retval 无
  */
void I2C_EE_WriteByte(u8 WriteAddr, u8 Data);

/**
  * @brief  从指定地址读取 1 个字节
  * @param  ReadAddr EEPROM 内部地址
  * @retval 读取到的数据
  */
u8 I2C_EE_ReadByte(u8 ReadAddr);

/**
  * @brief  连续写入多个字节
  * @param  pBuffer        数据缓冲区指针
  * @param  WriteAddr      起始写地址
  * @param  NumByteToWrite 要写入的字节数
  * @retval 无
  */
void I2C_EE_WriteBuffer(u8* pBuffer, u8 WriteAddr, u16 NumByteToWrite);

/**
  * @brief  连续读取多个字节
  * @param  pBuffer       数据缓冲区指针
  * @param  ReadAddr      起始读地址
  * @param  NumByteToRead 要读取的字节数
  * @retval 无
  */
void I2C_EE_ReadBuffer(u8* pBuffer, u8 ReadAddr, u16 NumByteToRead);

/**
  * @brief  检查 EEPROM 是否正常响应
  * @retval 0 表示正常，1 表示异常
  */
u8 I2C_EE_Check(void);

#endif
