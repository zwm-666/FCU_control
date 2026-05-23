#ifndef __BSP_SIMULATOR_H
#define __BSP_SIMULATOR_H

#include <stdint.h>

/* 模拟输入输出模块 */

/**
 * @brief 初始化模拟器
 * @param input_voltage_start_ms 输入电压接入时间（毫秒）
 * @param input_voltage_value 输入电压值（V）
 * @param output_voltage_target 输出电压目标值（V）
 * @param output_current_target 输出电流目标值（A）
 */
void Simulator_Init(uint32_t input_voltage_start_ms, float input_voltage_value, 
                   float output_voltage_target, float output_current_target);

/**
 * @brief 获取模拟输入电压
 * @return 模拟输入电压值（V）
 */
float Simulator_GetInputVoltage(void);

/**
 * @brief 模拟输入电压接入
 */
void Simulator_ConnectInput(void);

/**
 * @brief 获取模拟输出电压
 * @param pwm_duty 当前PWM占空比
 * @return 模拟输出电压值（V）
 */
float Simulator_GetOutputVoltage(uint16_t pwm_duty);

/**
 * @brief 获取模拟输入电流
 * @param pwm_duty 当前PWM占空比
 * @return 模拟输入电流值（A）
 */
float Simulator_GetInputCurrent(uint16_t pwm_duty);

/**
 * @brief 获取模拟输出电流
 * @param pwm_duty 当前PWM占空比
 * @return 模拟输出电流值（A）
 */
float Simulator_GetOutputCurrent(uint16_t pwm_duty);

/**
 * @brief 更新模拟器状态
 * @param pwm_duty 当前PWM占空比
 */
void Simulator_Update(uint16_t pwm_duty);

#endif /* __BSP_SIMULATOR_H */
