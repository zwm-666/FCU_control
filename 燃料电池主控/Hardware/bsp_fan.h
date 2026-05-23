#ifndef __BSP_FAN_H
#define __BSP_FAN_H

#include "stm32f10x.h"
#include "bsp_valve.h"

/*
 * 模块名称：bsp_fan
 * 模块功能：提供风扇 PWM 调速、供电控制和转速采集接口。
 * 说明：
 * 1. PWM 输出使用 TIM4_CH4；
 * 2. 转速采集使用 TIM1_CH1 输入捕获；
 * 3. 风扇电源通过 Valve_Control_Mout3 间接控制。
 */

/* ================= 引脚与定时器定义 ================= */
#define FAN_PWM_GPIO_PORT       GPIOB                 // 风扇 PWM 输出 GPIO 端口
#define FAN_PWM_PIN             GPIO_Pin_9           // 风扇 PWM 输出引脚，对应 PB9
#define FAN_PWM_RCC_GPIO        RCC_APB2Periph_GPIOB // 风扇 PWM GPIO 时钟
#define FAN_PWM_RCC_TIM         RCC_APB1Periph_TIM4  // 风扇 PWM 使用 TIM4

#define FAN_CAP_GPIO_PORT       GPIOA                 // 风扇测速输入 GPIO 端口
#define FAN_CAP_PIN             GPIO_Pin_8           // 风扇测速输入引脚，对应 PA8
#define FAN_CAP_RCC_GPIO        RCC_APB2Periph_GPIOA // 风扇测速 GPIO 时钟
#define FAN_CAP_RCC_TIM         RCC_APB2Periph_TIM1  // 风扇测速使用 TIM1

/* ================= 控制宏定义 ================= */
#define FAN_POWER_ON()          Fan_PowerCtrl(1)     // 打开风扇供电
#define FAN_POWER_OFF()         Fan_PowerCtrl(0)     // 关闭风扇供电

/**
  * @brief  初始化风扇相关硬件资源
  * @retval 无
  */
void Fan_Init(void);

/**
  * @brief  控制风扇供电开关
  * @param  on 1 表示上电，0 表示断电
  * @retval 无
  */
void Fan_PowerCtrl(uint8_t on);

/**
  * @brief  设置风扇 PWM 占空比
  * @param  duty 占空比，范围 0~100
  * @retval 无
  */
void Fan_Set_DutyCycle(uint8_t duty);

/**
  * @brief  获取当前风扇转速
  * @retval 风扇转速，单位：RPM
  */
uint16_t Fan_Get_RPM(void);

uint8_t Fan_Get_PowerState(void);
uint8_t Fan_Get_DutyCycle(void);

#endif
