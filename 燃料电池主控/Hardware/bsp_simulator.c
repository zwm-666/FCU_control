#include "bsp_simulator.h"
#include "delay.h"
#include "main.h"

/* 模拟器参数 */
typedef struct {
    float input_voltage_value;         // 输入电压值
    float output_voltage_target;       // 输出电压目标值
    float output_current_target;       // 输出电流目标值
    float input_voltage;               // 当前输入电压
    float output_voltage;              // 当前输出电压
    float input_current;               // 当前输入电流
    float output_current;              // 当前输出电流
    uint8_t input_voltage_on;          // 输入电压是否已接入
    uint32_t start_time_ms;           // 启动时间戳（毫秒）
    uint32_t connect_delay_ms;         // 连接延迟时间（毫秒）
} Simulator_Params_t;

extern volatile uint32_t g_ms;

static Simulator_Params_t g_sim_params = {
    .input_voltage_value = 12.0f,
    .output_voltage_target = 24.0f,
    .output_current_target = 2.0f,
    .input_voltage = 0.0f,
    .output_voltage = 0.0f,
    .input_current = 0.0f,
    .output_current = 0.0f,
    .input_voltage_on = 0
};

/**
 * @brief  初始化模拟器
 */
void Simulator_Init(uint32_t input_voltage_start_ms, float input_voltage_value, 
                   float output_voltage_target, float output_current_target)
{
    g_sim_params.input_voltage_value = input_voltage_value;
    g_sim_params.output_voltage_target = output_voltage_target;
    g_sim_params.output_current_target = output_current_target;
    g_sim_params.input_voltage = 0.0f;
    g_sim_params.output_voltage = 0.0f;
    g_sim_params.input_current = 0.0f;
    g_sim_params.output_current = 0.0f;
    g_sim_params.input_voltage_on = 0;
    g_sim_params.start_time_ms = 0;
    g_sim_params.connect_delay_ms = input_voltage_start_ms;
}

/**
 * @brief  获取模拟输入电压
 */
float Simulator_GetInputVoltage(void)
{
    if (g_sim_params.start_time_ms == 0)
    {
        g_sim_params.start_time_ms = g_ms;
    }
    
    if (!g_sim_params.input_voltage_on)
    {
        if (g_sim_params.connect_delay_ms > 0)
        {
            if ((g_ms - g_sim_params.start_time_ms) >= g_sim_params.connect_delay_ms)
            {
                g_sim_params.input_voltage_on = 1;
                g_sim_params.input_voltage = g_sim_params.input_voltage_value;
            }
        }
    }
    return g_sim_params.input_voltage;
}

/**
 * @brief 模拟输入电压接入
 */
void Simulator_ConnectInput(void)
{
    g_sim_params.input_voltage_on = 1;
    g_sim_params.input_voltage = g_sim_params.input_voltage_value;
}

/**
 * @brief 获取模拟输出电压
 */
float Simulator_GetOutputVoltage(uint16_t pwm_duty)
{
    if (!g_sim_params.input_voltage_on)
    {
        return 0.0f;
    }
    
    // 基于占空比计算输出电压
    float duty_ratio = pwm_duty / 1000.0f;
    float ideal_output = g_sim_params.input_voltage / (1 - duty_ratio * 0.95f);
    
    // 限制输出电压
    if (ideal_output > g_sim_params.output_voltage_target * 1.1f)
    {
        ideal_output = g_sim_params.output_voltage_target * 1.1f;
    }
    
    // 模拟输出电压的动态响应
    g_sim_params.output_voltage = g_sim_params.output_voltage * 0.95f + ideal_output * 0.05f;
    
    return g_sim_params.output_voltage;
}

/**
 * @brief 获取模拟输入电流
 */
float Simulator_GetInputCurrent(uint16_t pwm_duty)
{
    if (!g_sim_params.input_voltage_on || g_sim_params.input_voltage < 0.1f)
    {
        g_sim_params.input_current = 0.0f;
        return 0.0f;
    }
    
    // 基于输出电流和效率计算输入电流
    float output_power = g_sim_params.output_voltage * g_sim_params.output_current;
    float efficiency = 0.85f; // 假设效率为85%
    float input_power = output_power / efficiency;
    g_sim_params.input_current = input_power / g_sim_params.input_voltage;
    
    // 限制输入电流
    if (g_sim_params.input_current > 5.0f) // 最大输入电流5A
    {
        g_sim_params.input_current = 5.0f;
    }
    if (g_sim_params.input_current < 0.01f)
    {
        g_sim_params.input_current = 0.0f;
    }
    
    return g_sim_params.input_current;
}

/**
 * @brief 获取模拟输出电流
 */
float Simulator_GetOutputCurrent(uint16_t pwm_duty)
{
    if (!g_sim_params.input_voltage_on || g_sim_params.input_voltage < 0.1f)
    {
        g_sim_params.output_current = 0.0f;
        return 0.0f;
    }
    
    // 模拟输出电流
    float duty_ratio = pwm_duty / 1000.0f;
    float current = g_sim_params.output_current_target * duty_ratio * 0.8f;
    
    // 模拟电流的动态响应
    g_sim_params.output_current = g_sim_params.output_current * 0.9f + current * 0.1f;
    
    // 限制输出电流
    if (g_sim_params.output_current > g_sim_params.output_current_target * 1.2f)
    {
        g_sim_params.output_current = g_sim_params.output_current_target * 1.2f;
    }
    if (g_sim_params.output_current < 0.01f)
    {
        g_sim_params.output_current = 0.0f;
    }
    
    return g_sim_params.output_current;
}

/**
 * @brief  更新模拟器状态
 */
void Simulator_Update(uint16_t pwm_duty)
{
    // 获取当前输入电压（这会检查是否应该连接）
    Simulator_GetInputVoltage();
    
    // 获取当前输出电压
    Simulator_GetOutputVoltage(pwm_duty);
    
    // 获取当前输出电流
    Simulator_GetOutputCurrent(pwm_duty);
    
    // 获取当前输入电流
    Simulator_GetInputCurrent(pwm_duty);
}
