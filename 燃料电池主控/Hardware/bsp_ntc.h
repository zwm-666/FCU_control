#ifndef __BSP_NTC_H
#define __BSP_NTC_H

#include "stm32f10x.h"
#include <math.h>
#include <stdint.h>

/*
 * 模块名称：bsp_ntc
 * 模块功能：采集 NTC 温度、压力输入、电压输入和电流输入，并提供工程量换算接口。
 * 说明：
 * 1. 使用 ADC1 + DMA 扫描多个模拟通道；
 * 2. 支持两路 NTC 温度、两路压力、六路电压以及两路电流换算；
 * 3. 上层通过统一接口获取温度、电压、电流和压力值。
 */

/* ================= 热敏电阻与传感器参数 ================= */
#define NTC_NOMINAL_R       10000.0f    // NTC 在 25℃ 时的标称阻值 R25
#define NTC_NOMINAL_T       25.0f       // NTC 标称温度，单位：℃
#define NTC_B_VALUE         3435.0f     // NTC B 值，用于温度换算
#define PULLUP_R            10000.0f    // 分压上拉电阻阻值，单位：Ω

/* ADC 基础参数 */
#define ADC_REF_VOLT        3.3f        // ADC 参考电压，单位：V
#define ADC_MAX_VAL         4095.0f     // 12 位 ADC 满量程值

/* 压力传感器换算参数（0.5V~4.5V 映射到 MPa） */
#define PRESS_SENSOR_MIN_VOLT   0.5f    // 压力量程下限对应电压
#define PRESS_SENSOR_MAX_VOLT   4.5f    // 压力量程上限对应电压
#define PRESS_SENSOR_RANGE_MPA  0.2f    // 压力传感器满量程，单位：MPa
#define PRESS_RESISTOR_TOP_K    18.0f   // 压力采样分压上电阻，单位：kΩ
#define PRESS_RESISTOR_BOT_K    35.0f   // 压力采样分压下电阻，单位：kΩ

/* ================= 引脚与 ADC 通道定义 ================= */
#define NTC1_GPIO_PORT      GPIOB           // NTC1 所在 GPIO 端口
#define NTC1_GPIO_PIN       GPIO_Pin_0      // NTC1 输入引脚
#define NTC2_GPIO_PORT      GPIOB           // NTC2 所在 GPIO 端口
#define NTC2_GPIO_PIN       GPIO_Pin_1      // NTC2 输入引脚
#define NTC_ADC_CH_1        ADC_Channel_8   // NTC1 对应 ADC1_IN8
#define NTC_ADC_CH_2        ADC_Channel_9   // NTC2 对应 ADC1_IN9

#define VSENSE_GPIO_PORT    GPIOA           // 电压采样端口
#define VSENSE_GPIO_RCC     RCC_APB2Periph_GPIOA // 电压采样 GPIO 时钟
#define VSENSE_PIN_MASK     (GPIO_Pin_2 | GPIO_Pin_3 | GPIO_Pin_4 | GPIO_Pin_5 | GPIO_Pin_6 | GPIO_Pin_7) // AD1~AD6 引脚掩码

#define VSENSE_ADC_CH_0     ADC_Channel_2   // AD1 通道
#define VSENSE_ADC_CH_1     ADC_Channel_3   // AD2 通道
#define VSENSE_ADC_CH_2     ADC_Channel_4   // AD3 通道
#define VSENSE_ADC_CH_3     ADC_Channel_5   // AD4 通道
#define VSENSE_ADC_CH_4     ADC_Channel_6   // AD5 通道
#define VSENSE_ADC_CH_5     ADC_Channel_7   // AD6 通道

#define VSENSE_CH_NUM       6               // 电压采样通道总数

/* 每路电压采样的增益系数：实际电压 = 引脚电压 × gain */
#define VSENSE_CH0_GAIN     20.0f   // 燃料电池输入母线电压
#define VSENSE_CH1_GAIN     20.0f   // 48V 输出母线电压
#define VSENSE_CH2_GAIN     4.0f    // 12V 输出电压
#define VSENSE_CH3_GAIN     2.0f    // 5V 电压
#define VSENSE_CH4_GAIN     1.0f    // 预留 / 电流传感器输入
#define VSENSE_CH5_GAIN     1.0f    // 预留 / 电流传感器输入

/* 电流换算所用通道索引与参数 */
#define CURR_IN_ADC_INDEX       4U      // 输入电流对应 VSENSE 索引
#define CURR_OUT_ADC_INDEX      5U      // 输出电流对应 VSENSE 索引
#define CURR_ZERO_V             1.65f   // 电流传感器零点电压
#define CURR_SENS_V_PER_A       0.066f  // 每安培对应电压变化量

/* DMA 扫描缓冲区中的索引映射 */
#define ADC_IDX_NTC1            0U      // NTC1 在 DMA 缓冲区中的索引
#define ADC_IDX_NTC2            1U      // NTC2 在 DMA 缓冲区中的索引
#define ADC_IDX_VSENSE0         2U      // AD1 在 DMA 缓冲区中的索引
#define ADC_IDX_VSENSE1         3U      // AD2 在 DMA 缓冲区中的索引
#define ADC_IDX_VSENSE2         4U      // AD3 在 DMA 缓冲区中的索引
#define ADC_IDX_VSENSE3         5U      // AD4 在 DMA 缓冲区中的索引
#define ADC_IDX_VSENSE4         6U      // AD5 在 DMA 缓冲区中的索引
#define ADC_IDX_VSENSE5         7U      // AD6 在 DMA 缓冲区中的索引

#define NTC_CH_NUM          8           // DMA 扫描总通道数：2 路 NTC + 6 路电压

/**
  * @brief  初始化 NTC/电压采样模块
  * @retval 无
  */
void NTC_Init(void);

/**
  * @brief  获取指定 NTC 温度
  * @param  channel_index 通道索引，0 表示 NTC1，1 表示 NTC2
  * @retval 温度值，单位：℃
  */
float NTC_Get_Temperature(uint8_t channel_index);

/**
  * @brief  获取入口压力
  * @retval 压力值，单位：MPa
  */
float NTC_Get_InletPressure_MPa(void);

/**
  * @brief  获取瓶压
  * @retval 压力值，单位：MPa
  */
float NTC_Get_BottlePressure_MPa(void);

/**
  * @brief  获取指定电压采样通道的实际电压
  * @param  vsense_index 电压采样索引
  * @retval 实际电压值，单位：V
  */
float NTC_Get_VSense_V(uint8_t vsense_index);

/**
  * @brief  获取输入电流
  * @retval 电流值，单位：A
  */
float NTC_Get_InputCurrent_A(void);

/**
  * @brief  获取输出电流
  * @retval 电流值，单位：A
  */
float NTC_Get_OutputCurrent_A(void);

#endif
