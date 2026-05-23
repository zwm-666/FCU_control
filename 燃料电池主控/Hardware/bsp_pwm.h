#ifndef __BSP_PWM_H
#define __BSP_PWM_H

#include "stm32f10x.h"
#include <stdint.h>

/*
 * 模块名称：bsp_pwm
 * 模块功能：提供 PWM 输出控制功能，用于 DC-DC 转换器的 MOS 管控制
 * 说明：
 * 1. 使用 TIM3 定时器生成 PWM 波形
 * 2. 支持占空比 0-100% 调节
 * 3. 支持 PWM 使能/禁用控制
 * 4. 提供多种频率配置选项
 */

/* PWM 硬件定义 */
#define PWM_TIM                  TIM3                 // PWM 定时器
#define PWM_TIM_RCC              RCC_APB1Periph_TIM3  // 定时器时钟
#define PWM_GPIO_PORT            GPIOB                // PWM 输出端口
#define PWM_GPIO_RCC             RCC_APB2Periph_GPIOB // GPIO 时钟
#define PWM_GPIO_PIN             GPIO_Pin_4           // PWM 输出引脚 (TIM3_CH1)

/* PWM 配置参数 */
#define PWM_FREQ_10KHZ           10000U              // 10kHz PWM 频率
#define PWM_FREQ_20KHZ           20000U              // 20kHz PWM 频率
#define PWM_FREQ_50KHZ           50000U              // 50kHz PWM 频率
#define PWM_FREQ_100KHZ          100000U             // 100kHz PWM 频率

#define PWM_MAX_DUTY             1000U               // 最大占空比（千分比）
#define PWM_MIN_DUTY             0U                  // 最小占空比（千分比）

/**
  * @brief  初始化 PWM 模块
  * @param  freq_hz: PWM 频率，单位 Hz
  * @retval 无
  */
void PWM_Init(uint32_t freq_hz);

/**
  * @brief  设置 PWM 占空比
  * @param  duty: 占空比，千分比 (0-1000)
  * @retval 无
  */
void PWM_SetDuty(uint16_t duty);

/**
  * @brief  使能 PWM 输出
  * @retval 无
  */
void PWM_Enable(void);

/**
  * @brief  禁用 PWM 输出
  * @retval 无
  */
void PWM_Disable(void);

/**
  * @brief  获取当前 PWM 占空比
  * @retval 当前占空比，千分比 (0-1000)
  */
uint16_t PWM_GetDuty(void);

/**
  * @brief  PWM 驱动使用示例
  * 
  * @code
  * // 1. 初始化 PWM，设置为 100kHz 频率
  * PWM_Init(PWM_FREQ_100KHZ);
  * 
  * // 2. 设置占空比为 50%
  * PWM_SetDuty(500);
  * 
  * // 3. 使能 PWM 输出
  * PWM_Enable();
  * 
  * // 4. 运行一段时间后，修改占空比为 75%
  * PWM_SetDuty(750);
  * 
  * // 5. 禁用 PWM 输出
  * PWM_Disable();
  * 
  * // 6. 获取当前占空比
  * uint16_t current_duty = PWM_GetDuty();
  * @endcode
  */

#endif