#ifndef __BSP_VALVE_H
#define __BSP_VALVE_H

#include "stm32f10x.h"

/*
 * 模块名称：bsp_valve
 * 模块功能：提供燃料电池系统电磁阀/执行输出的底层控制接口。
 * 硬件映射：
 * 1. MOUT1 -> PA0，对应进气阀；
 * 2. MOUT2 -> PA1，对应排气阀；
 * 3. MOUT3 -> PB8，对应第三路高边输出（当前用于风扇电源控制）。
 */

/* 进气阀控制引脚定义 */
#define VALVE_INLET_PORT        GPIOA                 // 进气阀 GPIO 端口
#define VALVE_INLET_RCC         RCC_APB2Periph_GPIOA // 进气阀 GPIO 时钟
#define VALVE_INLET_PIN         GPIO_Pin_0           // 进气阀控制引脚

/* 排气阀控制引脚定义 */
#define VALVE_PURGE_PORT        GPIOA                 // 排气阀 GPIO 端口
#define VALVE_PURGE_RCC         RCC_APB2Periph_GPIOA // 排气阀 GPIO 时钟
#define VALVE_PURGE_PIN         GPIO_Pin_1           // 排气阀控制引脚

/* 第三路高边输出定义 */
#define VALVE_HEATER_PORT       GPIOB                 // 第三路输出 GPIO 端口
#define VALVE_HEATER_RCC        RCC_APB2Periph_GPIOB // 第三路输出 GPIO 时钟
#define VALVE_HEATER_PIN        GPIO_Pin_8           // 第三路输出控制引脚

/*
 * 高边 PMOS 经光耦后为低电平有效：
 * 1. 输出低电平表示导通；
 * 2. 输出高电平表示关闭。
 */
#define VALVE_ON(port, pin)     GPIO_ResetBits(port, pin) // 拉低引脚，打开高边输出
#define VALVE_OFF(port, pin)    GPIO_SetBits(port, pin)   // 拉高引脚，关闭高边输出

/**
  * @brief  阀控输出初始化
  * @retval 无
  */
void Valve_Init(void);

/**
  * @brief  控制进气阀开关
  * @param  state 1 表示打开，0 表示关闭
  * @retval 无
  */
void Valve_Control_Inlet(uint8_t state);

/**
  * @brief  控制排气阀开关
  * @param  state 1 表示打开，0 表示关闭
  * @retval 无
  */
void Valve_Control_Purge(uint8_t state);

/**
  * @brief  控制第三路高边输出开关
  * @param  state 1 表示打开，0 表示关闭
  * @retval 无
  */
void Valve_Control_Mout3(uint8_t state);

uint8_t Valve_Get_InletState(void);
uint8_t Valve_Get_PurgeState(void);
uint8_t Valve_Get_Mout3State(void);

#endif
