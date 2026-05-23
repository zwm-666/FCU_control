#include "stm32f10x.h"
#include <stdint.h>

/*
 * 模块名称：delay
 * 模块功能：使用 SysTick 定时器实现阻塞式延时。
 * 说明：
 * 1. 当前实现默认系统时钟为 72MHz；
 * 2. 微秒延时直接按时钟周期装载计数值；
 * 3. 毫秒和秒延时由更小单位循环累加得到。
 */

/**
  * @brief  微秒级延时
  * @param  xus 延时时长，范围：0~233015
  * @retval 无
  */
void Delay_us(uint32_t xus)
{
	uint32_t old_load = SysTick->LOAD;
	uint32_t old_val = SysTick->VAL;
	uint32_t old_ctrl = SysTick->CTRL;

	SysTick->CTRL = 0x00000000;
	SysTick->LOAD = 72 * xus;				// 按 72MHz 主频计算重装值，1us 对应 72 个时钟周期
	SysTick->VAL = 0x00;					// 清空当前计数值，确保从头开始计数
	SysTick->CTRL = 0x00000005;			// 选择 HCLK 作为时钟源并启动 SysTick
	while (!(SysTick->CTRL & 0x00010000));	// 等待计数到 0，COUNTFLAG 置位后退出

	SysTick->CTRL = 0x00000000;
	SysTick->LOAD = old_load;
	SysTick->VAL = old_val;
	SysTick->CTRL = old_ctrl;
}

/**
  * @brief  毫秒级延时
  * @param  xms 延时时长，范围：0~4294967295
  * @retval 无
  */
void Delay_ms(uint32_t xms)
{
	while (xms--)
	{
		Delay_us(1000);	// 每次循环延时 1000us，累计得到 1ms
	}
}

/**
  * @brief  秒级延时
  * @param  xs 延时时长，范围：0~4294967295
  * @retval 无
  */
void Delay_s(uint32_t xs)
{
	while (xs--)
	{
		Delay_ms(1000);	// 每次循环延时 1000ms，累计得到 1s
	}
}
