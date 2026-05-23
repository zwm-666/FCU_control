#ifndef __DELAY_H
#define __DELAY_H

#include <stdint.h>

/*
 * 模块名称：delay
 * 模块功能：提供基于 SysTick 的微秒、毫秒、秒级阻塞延时函数。
 * 说明：
 * 1. 本模块适用于系统主频为 72MHz 的场景；
 * 2. 延时期间 CPU 会一直等待，不适合长时间占用的实时任务；
 * 3. 该头文件声明了对外提供的延时接口。
 */

/**
  * @brief  微秒级延时
  * @param  us 延时时长，单位：微秒
  * @retval 无
  */
void Delay_us(uint32_t us);

/**
  * @brief  毫秒级延时
  * @param  ms 延时时长，单位：毫秒
  * @retval 无
  */
void Delay_ms(uint32_t ms);

/**
  * @brief  秒级延时
  * @param  s 延时时长，单位：秒
  * @retval 无
  */
void Delay_s(uint32_t s);

#endif
