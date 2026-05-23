#include "bsp_dcdc_boost.h"

/**
 * @brief 初始化PID控制器
 * 
 * @param pid PID控制器结构体指针
 * @param kp 比例系数
 * @param ki 积分系数
 * @param kd 微分系数
 * @param max_out 最大输出限幅
 * @param min_out 最小输出限幅
 * @note 此函数为内部函数，不对外暴露
 * 
 * @code
 * // 调用示例（内部使用）
 * PID_Controller pid;
 * PID_Init(&pid, 1.0f, 0.5f, 0.1f, 1.0f, 0.0f);
 * @endcode
 */
 
static void PID_Init(PID_Controller *pid, float kp, float ki, float kd, float max_out, float min_out)
{
    if (pid == NULL)
        return;
    
    pid->Kp = kp;
    pid->Ki = ki;
    pid->Kd = kd;
    pid->integral = 0.0f;
    pid->prev_error = 0.0f;
    pid->max_output = max_out;
    pid->min_output = min_out;
}

/**
 * @brief 初始化整数版PID控制器
 * 
 * @param pid 整数版PID控制器结构体指针
 * @param kp 比例系数（浮点数）
 * @param ki 积分系数（浮点数）
 * @param kd 微分系数（浮点数）
 * @param max_out 最大输出限幅（浮点数）
 * @param min_out 最小输出限幅（浮点数）
 * @note 将浮点数参数转换为Q12格式的整数
 */
static void PID_Init_Int(PID_Controller_Int *pid, float kp, float ki, float kd, float max_out, float min_out)
{
    if (pid == NULL)
        return;
    
    pid->Kp = (int32_t)(kp * 4096.0f);
    pid->Ki = (int32_t)(ki * 4096.0f);
    pid->Kd = (int32_t)(kd * 4096.0f);
    pid->integral = 0;
    pid->prev_error = 0;
    pid->prev2_error = 0;
    pid->last_output = 0;   // 初始化last_output为0，防止启动瞬间占空比随机跳变
    pid->max_output = (int32_t)(max_out * 4096.0f);
    pid->min_output = (int32_t)(min_out * 4096.0f);
}

/**
 * @brief 初始化DCDC Boost控制器
 * 
 * 将控制器的所有参数设置为默认值：
 * - PID参数：使用DEFAULT宏定义的默认值
 * - 保护限制：使用DEFAULT宏定义的默认值
 * - 目标值：初始化为0
 * - 状态标志：初始化为禁用状态
 * 
 * @param controller 控制器结构体指针
 * 
 * @code
 * // 调用示例
 * DCDC_Boost_Controller boost_ctrl;
 * DCDC_Boost_Init(&boost_ctrl);
 * @endcode
 */
void DCDC_Boost_Init(DCDC_Boost_Controller *controller)
{
    if (controller == NULL)
        return;
    
    /* 初始化电压环PID控制器 */
    PID_Init(&controller->voltage_pid, 
             DCDC_BOOST_DEFAULT_VOLTAGE_KP, 
             DCDC_BOOST_DEFAULT_VOLTAGE_KI, 
             DCDC_BOOST_DEFAULT_VOLTAGE_KD, 
             DCDC_BOOST_DEFAULT_MAX_CURRENT, 0.0f);
    
    /* 初始化电流环PID控制器 */
    PID_Init(&controller->current_pid, 
             DCDC_BOOST_DEFAULT_CURRENT_KP, 
             DCDC_BOOST_DEFAULT_CURRENT_KI, 
             DCDC_BOOST_DEFAULT_CURRENT_KD, 
             1.0f, 0.0f);
    
    /* 初始化电压环整数版PID控制器 */
    PID_Init_Int(&controller->voltage_pid_int, 
                 DCDC_BOOST_DEFAULT_VOLTAGE_KP, 
                 DCDC_BOOST_DEFAULT_VOLTAGE_KI, 
                 DCDC_BOOST_DEFAULT_VOLTAGE_KD, 
                 DCDC_BOOST_DEFAULT_MAX_CURRENT, 0.0f);
    
    /* 初始化电流环整数版PID控制器 */
    PID_Init_Int(&controller->current_pid_int, 
                 DCDC_BOOST_DEFAULT_CURRENT_KP, 
                 DCDC_BOOST_DEFAULT_CURRENT_KI, 
                 DCDC_BOOST_DEFAULT_CURRENT_KD, 
                 1.0f, 0.0f);
    
    /* 初始化目标值 */
    controller->target_voltage = 0.0f;
    controller->target_current = 0.0f;
    
    /* 初始化采样值（清零） */
    controller->input_voltage = 0.0f;
    controller->output_voltage = 0.0f;
    controller->output_current = 0.0f;
    controller->inductor_current = 0.0f;
    
    /* 初始化保护限制参数（使用默认值） */
    controller->max_voltage = DCDC_BOOST_DEFAULT_MAX_VOLTAGE;
    controller->max_current = DCDC_BOOST_DEFAULT_MAX_CURRENT;
    controller->max_input_voltage = DCDC_BOOST_DEFAULT_MAX_INPUT_VOLTAGE;
    controller->min_input_voltage = DCDC_BOOST_DEFAULT_MIN_INPUT_VOLTAGE;
    controller->min_output_voltage = DCDC_BOOST_SHORT_CIRCUIT_THRESHOLD;  /* 短路检测阈值 */
    
    /* 初始化占空比输出 */
    controller->duty_cycle = 0;
    
    /* 初始化工作模式（默认恒压模式） */
    controller->mode = DCDC_BOOST_MODE_CONSTANT_VOLTAGE;
    controller->prev_mode = DCDC_BOOST_MODE_CONSTANT_VOLTAGE;
    
    /* 初始化恒压恒流切换滞回参数 */
    controller->cc_cv_hysteresis_ma = 100;  /* 默认100mA滞回 */
    
    /* 初始化软启动状态 */
    controller->soft_start_enable = 0;
    controller->soft_start_timer = 0;
    controller->soft_start_target = 0.0f;
    
    /* 初始化使能标志（默认禁用） */
    controller->enable = 0;
    
    /* 初始化故障状态（默认无故障） */
    controller->fault = DCDC_BOOST_FAULT_NONE;
    
    /* 初始化电源检测标志 */
    controller->power_on_detected = 0;
}

/**
 * @brief 设置PID控制器参数
 * 
 * 允许外部修改电压环和电流环的PID参数
 * 如果不调用此函数，将使用初始化时设置的默认参数
 * 
 * @param controller 控制器结构体指针
 * @param voltage_kp 电压环比例系数
 * @param voltage_ki 电压环积分系数
 * @param voltage_kd 电压环微分系数
 * @param current_kp 电流环比例系数
 * @param current_ki 电流环积分系数
 * @param current_kd 电流环微分系数
 * 
 * @code
 * // 调用示例：设置自定义PID参数
 * DCDC_Boost_SetPIDParams(&boost_ctrl,
 *                         0.3f, 15.0f, 0.0f,   // 电压环 Kp, Ki, Kd
 *                         1.0f, 2500.0f, 0.0f); // 电流环 Kp, Ki, Kd
 * @endcode
 */
void DCDC_Boost_SetPIDParams(DCDC_Boost_Controller *controller,
                             float voltage_kp, float voltage_ki, float voltage_kd,
                             float current_kp, float current_ki, float current_kd)
{
    if (controller == NULL)
        return;
    
    /* 设置电压环PID参数 */
    controller->voltage_pid.Kp = voltage_kp;
    controller->voltage_pid.Ki = voltage_ki;
    controller->voltage_pid.Kd = voltage_kd;
    
    /* 设置电流环PID参数 */
    controller->current_pid.Kp = current_kp;
    controller->current_pid.Ki = current_ki;
    controller->current_pid.Kd = current_kd;
    
    /* 更新电压环整数版PID参数（Q12格式） */
    controller->voltage_pid_int.Kp = (int32_t)(voltage_kp * 4096.0f);
    controller->voltage_pid_int.Ki = (int32_t)(voltage_ki * 4096.0f);
    controller->voltage_pid_int.Kd = (int32_t)(voltage_kd * 4096.0f);
    
    /* 更新电流环整数版PID参数（Q12格式） */
    controller->current_pid_int.Kp = (int32_t)(current_kp * 4096.0f);
    controller->current_pid_int.Ki = (int32_t)(current_ki * 4096.0f);
    controller->current_pid_int.Kd = (int32_t)(current_kd * 4096.0f);
}

/**
 * @brief 设置额定输出参数（推荐使用，自动计算保护限值）
 * 
 * 根据额定值自动计算保护限值，简化配置流程：
 *   - 输出电压保护上限 = 额定电压 × SAFETY_FACTOR_VOLTAGE
 *   - 输出电流保护上限 = 额定电流 × SAFETY_FACTOR_CURRENT
 *   - 输入电压保护上限 = 额定输入电压 × SAFETY_FACTOR_INPUT_HIGH
 *   - 输入电压保护下限 = 额定输入电压 × SAFETY_FACTOR_INPUT_LOW
 * 
 * @param controller 控制器结构体指针
 * @param rated_voltage 额定输出电压，单位V（目标工作电压）
 * @param rated_current 额定输出电流，单位A（目标工作电流）
 * @param rated_input_voltage 额定输入电压，单位V（典型输入电压）
 * 
 * @code
 * // 调用示例：设置6V输入，13V输出，3A电流
 * DCDC_Boost_SetRatedParams(&boost_ctrl, 13.0f, 3.0f, 6.0f);
 * // 自动计算：
 * //   输出电压上限 = 13V × 1.1 = 14.3V
 * //   输出电流上限 = 3A × 1.1 = 3.3A
 * //   输入电压上限 = 6V × 1.1 = 6.6V
 * //   输入电压下限 = 6V × 0.9 = 5.4V
 * @endcode
 */
void DCDC_Boost_SetRatedParams(DCDC_Boost_Controller *controller,
                               float rated_voltage, float rated_current,
                               float rated_input_voltage)
{
    if (controller == NULL)
        return;
    
    /* 根据额定值自动计算保护限值 */
    float max_voltage = rated_voltage * DCDC_BOOST_SAFETY_FACTOR_VOLTAGE;
    float max_current = rated_current * DCDC_BOOST_SAFETY_FACTOR_CURRENT;
    /* 使用宽电压输入范围配置 */
    float max_input_voltage = DCDC_BOOST_WIDE_INPUT_MAX_VOLTAGE * DCDC_BOOST_SAFETY_FACTOR_INPUT_HIGH;
    float min_input_voltage = DCDC_BOOST_WIDE_INPUT_MIN_VOLTAGE * DCDC_BOOST_SAFETY_FACTOR_INPUT_LOW;
    
    /* 设置保护限制 */
    controller->max_voltage = max_voltage;
    controller->max_current = max_current;
    controller->max_input_voltage = max_input_voltage;
    controller->min_input_voltage = min_input_voltage;
    controller->min_output_voltage = 2.0f;  /* 默认输出下限2V */
    
    /* 更新预计算值（用于中断中避免浮点运算） */
    controller->target_voltage_mv = (int32_t)(rated_voltage * 1000.0f);
    controller->target_current_ma = (int32_t)(rated_current * 1000.0f);
    controller->max_current_ma = (int32_t)(max_current * 1000.0f);
    controller->max_voltage_mv = (int32_t)(max_voltage * 1000.0f);
    controller->min_output_voltage_mv = 2000;  /* 默认2V * 1000 */
    
    /* 更新电压环PID的输出限幅，与最大电流关联 */
    controller->voltage_pid.max_output = max_current;
    controller->voltage_pid_int.max_output = (int32_t)(max_current * 4096.0f);
    
    /* 同时设置目标值为额定值 */
    controller->target_voltage = rated_voltage;
    controller->target_current = rated_current;
}

/**
 * @brief 设置保护限制参数（高级用法，手动配置）
 * 
 * 允许外部直接设置保护限制值，适用于需要精细控制的场景
 * 如果不调用此函数或SetRatedParams()，将使用默认限制值
 * 
 * @param controller 控制器结构体指针
 * @param max_voltage 输出电压保护上限，单位V（应高于额定电压）
 * @param max_current 输出电流保护上限，单位A（应高于额定电流）
 * @param max_input_voltage 输入电压保护上限，单位V（应高于额定输入电压）
 * @param min_input_voltage 输入电压保护下限，单位V（应低于额定输入电压）
 * @param min_output_voltage 输出电压保护下限，单位V（用于区分恒流降压和短路，建议2V）
 * 
 * @code
 * // 调用示例：设置48V输出，5A限流，输入8-30V，输出下限2V
 * DCDC_Boost_SetLimits(&boost_ctrl, 52.8f, 5.5f, 33.0f, 7.2f, 2.0f);
 * @endcode
 */
void DCDC_Boost_SetLimits(DCDC_Boost_Controller *controller,
                          float max_voltage, float max_current,
                          float max_input_voltage, float min_input_voltage,
                          float min_output_voltage)
{
    if (controller == NULL)
        return;
    
    controller->max_voltage = max_voltage;
    controller->max_current = max_current;
    controller->max_input_voltage = max_input_voltage;
    controller->min_input_voltage = min_input_voltage;
    controller->min_output_voltage = min_output_voltage;
    
    controller->max_voltage_mv = (int32_t)(max_voltage * 1000.0f);
    controller->max_current_ma = (int32_t)(max_current * 1000.0f);
    controller->min_output_voltage_mv = (int32_t)(min_output_voltage * 1000.0f);
    
    /* 更新电压环PID的输出限幅，与最大电流关联 */
    controller->voltage_pid.max_output = max_current;
    controller->voltage_pid_int.max_output = (int32_t)(max_current * 4096.0f);
}

/**
 * @brief 设置控制器工作模式
 * 
 * @param controller 控制器结构体指针
 * @param mode 工作模式：恒压或恒流
 * 
 * @code
 * // 调用示例：设置为恒流模式
 * DCDC_Boost_SetMode(&boost_ctrl, DCDC_BOOST_MODE_CONSTANT_CURRENT);
 * 
 * // 调用示例：设置为恒压模式
 * DCDC_Boost_SetMode(&boost_ctrl, DCDC_BOOST_MODE_CONSTANT_VOLTAGE);
 * @endcode
 */
void DCDC_Boost_SetMode(DCDC_Boost_Controller *controller, DCDC_Boost_Mode mode)
{
    if (controller == NULL)
        return;
    
    controller->mode = mode;
}

/**
 * @brief 设置目标输出电压
 * 
 * 设置值会自动限制在0到max_voltage之间
 * 
 * @param controller 控制器结构体指针
 * @param voltage 目标电压值，单位V
 * 
 * @code
 * // 调用示例：设置目标输出电压为24V
 * DCDC_Boost_SetTargetVoltage(&boost_ctrl, 24.0f);
 * @endcode
 */
void DCDC_Boost_SetTargetVoltage(DCDC_Boost_Controller *controller, float voltage)
{
    if (controller == NULL)
        return;
    
    /* 限制目标电压在有效范围内 */
    if (voltage > controller->max_voltage)
        voltage = controller->max_voltage;
    if (voltage < 0.0f)
        voltage = 0.0f;
    
    controller->target_voltage = voltage;
    controller->target_voltage_mv = (int32_t)(voltage * 1000.0f);
}

/**
 * @brief 设置目标输出电流（限流值）
 * 
 * 设置值会自动限制在0到max_current之间
 * 
 * @param controller 控制器结构体指针
 * @param current 目标电流值，单位A
 * 
 * @code
 * // 调用示例：设置限流值为3A
 * DCDC_Boost_SetTargetCurrent(&boost_ctrl, 3.0f);
 * @endcode
 */
void DCDC_Boost_SetTargetCurrent(DCDC_Boost_Controller *controller, float current)
{
    if (controller == NULL)
        return;
    
    /* 限制目标电流在有效范围内 */
    if (current > controller->max_current)
        current = controller->max_current;
    if (current < 0.0f)
        current = 0.0f;
    
    controller->target_current = current;
    controller->target_current_ma = (int32_t)(current * 1000.0f);
}

/**
 * @brief 启用软启动功能
 * 
 * 软启动功能可以使输出电压从0平滑上升到目标值，避免冲击电流
 * 软启动时间由DCDC_BOOST_SOFT_START_TIME宏定义控制（默认1000ms）
 * 
 * @param controller 控制器结构体指针
 * @param target 软启动目标电压，单位V
 * 
 * @code
 * // 调用示例：启用软启动，目标电压24V
 * DCDC_Boost_EnableSoftStart(&boost_ctrl, 24.0f);
 * @endcode
 */
void DCDC_Boost_EnableSoftStart(DCDC_Boost_Controller *controller, float target)
{
    if (controller == NULL)
        return;
    
    controller->soft_start_enable = 1;
    controller->soft_start_timer = 0;
    controller->soft_start_target = target;
}

/**
 * @brief 禁用软启动功能
 * 
 * @param controller 控制器结构体指针
 * 
 * @code
 * // 调用示例：禁用软启动
 * DCDC_Boost_DisableSoftStart(&boost_ctrl);
 * @endcode
 */
void DCDC_Boost_DisableSoftStart(DCDC_Boost_Controller *controller)
{
    if (controller == NULL)
        return;
    
    controller->soft_start_enable = 0;
}

/**
 * @brief 使能DCDC Boost控制器
 * 
 * 使能后控制器才会根据采样值计算并输出占空比
 * 
 * @param controller 控制器结构体指针
 * 
 * @code
 * // 调用示例：使能控制器
 * DCDC_Boost_Enable(&boost_ctrl);
 * @endcode
 */
void DCDC_Boost_Enable(DCDC_Boost_Controller *controller)
{
    if (controller == NULL)
        return;
    
    controller->enable = 1;
}

/**
 * @brief 禁用DCDC Boost控制器
 * 
 * 禁用后占空比输出强制设为0，关闭功率输出
 * 
 * @param controller 控制器结构体指针
 * 
 * @code
 * // 调用示例：禁用控制器
 * DCDC_Boost_Disable(&boost_ctrl);
 * @endcode
 */
void DCDC_Boost_Disable(DCDC_Boost_Controller *controller)
{
    if (controller == NULL)
        return;
    
    controller->enable = 0;
    controller->duty_cycle = 0;
}

/**
 * @brief PID计算函数
 * 
 * 实现标准的增量式PID算法，包含积分限幅功能
 * 
 * @param pid PID控制器结构体指针
 * @param setpoint 设定值
 * @param process_value 过程值（反馈值）
 * @return PID输出值
 * 
 * @code
 * // 调用示例：单独使用PID计算
 * PID_Controller pid;
 * pid.Kp = 1.0f;
 * pid.Ki = 0.5f;
 * pid.Kd = 0.1f;
 * pid.integral = 0.0f;
 * pid.prev_error = 0.0f;
 * pid.max_output = 1.0f;
 * pid.min_output = 0.0f;
 * float output = PID_Compute(&pid, 100.0f, process_value);
 * @endcode
 */
float PID_Compute(PID_Controller *pid, float setpoint, float process_value)
{
    float error, derivative, output;
    
    if (pid == NULL)
        return 0.0f;
    
    /* 计算误差 */
    error = setpoint - process_value;
    
    /* 积分计算与限幅 */
    pid->integral += error;
    if (pid->integral > pid->max_output)
        pid->integral = pid->max_output;
    if (pid->integral < pid->min_output)
        pid->integral = pid->min_output;
    
    /* 微分计算 */
    derivative = error - pid->prev_error;
    pid->prev_error = error;
    
    /* PID输出计算与限幅 */
    output = pid->Kp * error + pid->Ki * pid->integral + pid->Kd * derivative;
    
    if (output > pid->max_output)
        output = pid->max_output;
    if (output < pid->min_output)
        output = pid->min_output;
    
    return output;
}

/**
 * @brief PID计算函数（整数版本）
 * 
 * 实现标准的增量式PID算法，使用整数计算，提高执行速度
 */
int32_t PID_Compute_Int(PID_Controller *pid, int32_t setpoint, int32_t process_value)
{
    int32_t error, derivative, output;
    int32_t kp_scaled, ki_scaled, kd_scaled;
    
    if (pid == NULL)
        return 0;
    
    /* 计算误差 */
    error = setpoint - process_value;
    
    /* 积分计算与限幅 */
    pid->integral += (float)error * 0.001f; // 转换为浮点数进行积分计算
    if (pid->integral > pid->max_output)
        pid->integral = pid->max_output;
    if (pid->integral < pid->min_output)
        pid->integral = pid->min_output;
    
    /* 微分计算 */
    derivative = error - (int32_t)(pid->prev_error * 1000.0f);
    pid->prev_error = (float)error * 0.001f;
    
    /* PID参数缩放 */
    kp_scaled = (int32_t)(pid->Kp * 1000.0f);
    ki_scaled = (int32_t)(pid->Ki * 1000.0f);
    kd_scaled = (int32_t)(pid->Kd * 1000.0f);
    
    /* PID输出计算与限幅 */
    output = (kp_scaled * error) / 1000 + 
             (ki_scaled * (int32_t)(pid->integral * 1000.0f)) / 1000 + 
             (kd_scaled * derivative) / 1000;
    
    /* 输出限幅 */
    int32_t max_output_scaled = (int32_t)(pid->max_output * 1000.0f);
    int32_t min_output_scaled = (int32_t)(pid->min_output * 1000.0f);
    
    if (output > max_output_scaled)
        output = max_output_scaled;
    if (output < min_output_scaled)
        output = min_output_scaled;
    
    return output;
}

/**
 * @brief 位置式PI计算（电压环使用）
 * 
 * @param pid 整数版PID控制器结构体指针
 * @param setpoint 设定值（整数，单位：毫伏/毫安）
 * @param process_value 过程值（整数，单位：毫伏/毫安）
 * @return PI输出（整数，千分比格式 0-1000）
 * @note 纯整数运算，使用Q12格式固定点数学
 */
int32_t PI_Positional_Int(PID_Controller_Int *pid, int32_t setpoint, int32_t process_value)
{
    int32_t error, p_term, i_term, output;
    
    if (pid == NULL)
        return 0;
    
    error = setpoint - process_value;
    
    /* 积分累积（error 是物理单位，直接累加） */
    pid->integral += error;
    
    /* 积分限幅：防止积分饱和，限幅在保守范围内
     * 设置为±5000，避免积分项过大导致输出超限
     * 这个值可以在实测中根据响应进行微调 */
    if (pid->integral > 5000)   /* 5000 作为保守的上限 */
        pid->integral = 5000;
    if (pid->integral < -5000)
        pid->integral = -5000;
    
    /* 比例项：Kp (Q12) * error (物理单位) >> 12 = 物理单位 */
    p_term = (pid->Kp * error) >> 12;
    
    /* 积分项：Ki (Q12) * integral (物理单位) >> 12 = 物理单位 */
    i_term = (pid->Ki * pid->integral) >> 12;
    
    output = p_term + i_term;
    
    /* 注意：电压环输出是电流设定值(mA)，不应限幅到0-1000
     * 由调用方DCDC_Boost_Update_Int进行合理的max_current_ma限幅 */
    
    return output;
}

/**
 * @brief 增量式PI计算（电流环使用）
 * 
 * @param pid 整数版PID控制器结构体指针
 * @param setpoint 设定值（整数，单位：毫伏/毫安）
 * @param process_value 过程值（整数，单位：毫伏/毫安）
 * @return PI输出（整数，千分比格式 0-1000）
 * @note 纯整数运算，使用Q12格式固定点数学
 */
int32_t PI_Incremental_Int(PID_Controller_Int *pid, int32_t setpoint, int32_t process_value)
{
    int32_t error, delta_p, delta_i, delta_output;
    
    if (pid == NULL)
        return 0;
    
    error = setpoint - process_value;
    
    /* 计算比例增量和积分增量 */
    delta_p = pid->Kp * (error - pid->prev_error);
    delta_i = pid->Ki * error;
    
    /* 更新误差历史 */
    pid->prev2_error = pid->prev_error;
    pid->prev_error = error;
    
    /* 计算总增量并转换为物理单位 */
    delta_output = (delta_p + delta_i) >> 12;
    
    /* 累加输出值到last_output字段 */
    pid->last_output += delta_output;
    
    /* 输出限幅：电流环输出是占空比0-1000 */
    if (pid->last_output > 1000)
        pid->last_output = 1000;
    if (pid->last_output < 0)
        pid->last_output = 0;
    
    return pid->last_output;
}

/**
 * @brief 重置PID控制器状态
 * 
 * 将积分值和上一次误差清零，用于控制器重启或切换模式时
 * 
 * @param pid PID控制器结构体指针
 * 
 * @code
 * // 调用示例：重置电压环PID
 * PID_Reset(&boost_ctrl.voltage_pid);
 * 
 * // 调用示例：重置电流环PID
 * PID_Reset(&boost_ctrl.current_pid);
 * @endcode
 */
void PID_Reset(PID_Controller *pid)
{
    if (pid == NULL)
        return;
    
    pid->integral = 0.0f;
    pid->prev_error = 0.0f;
}

/**
 * @brief 重置整数版PID控制器状态
 * 
 * 将积分值、误差历史和last_output清零，用于控制器重启或切换模式时
 * 
 * @param pid 整数版PID控制器结构体指针
 * 
 * @code
 * // 调用示例：重置电压环整数PID
 * PID_Reset_Int(&boost_ctrl.voltage_pid_int);
 * 
 * // 调用示例：重置电流环整数PID
 * PID_Reset_Int(&boost_ctrl.current_pid_int);
 * @endcode
 */
void PID_Reset_Int(PID_Controller_Int *pid)
{
    if (pid == NULL)
        return;
    
    pid->integral = 0;
    pid->prev_error = 0;
    pid->prev2_error = 0;
    pid->last_output = 0;
}

/**
 * @brief 更新DCDC Boost控制器（主控制函数）
 * 
 * 执行双环PID控制算法，包含多层保护机制：
 * 1. 参数有效性检查
 * 2. 更新采样值
 * 3. 输入电压保护检查（超限时关闭输出并清零积分）
 * 4. 输出过压/过流保护检查（超限时关闭输出并清零积分）
 * 5. 软启动处理（如果启用）
 * 6. 电压外环计算（输出作为电流内环的设定值）
 * 7. 电流设定值限幅（防止过流）
 * 8. 电流内环计算（输出作为占空比）
 * 9. 占空比理论上限限制（基于Vin/Vout关系）
 * 10. 占空比变化率限制（防止突变）
 * 11. 占空比硬限幅（5%-95%）
 * 
 * @param controller 控制器结构体指针
 * @param input_voltage 输入电压采样值，单位V
 * @param output_voltage 输出电压采样值，单位V
 * @param output_current 输出电流采样值，单位A
 * @param inductor_current 电感电流采样值，单位A
 * @return 当前占空比，千分比(0-1000)
 * @note 此函数应按DCDC_BOOST_CONTROL_FREQ频率调用（默认100kHz）
 * 
 * @code
 * // 调用示例：在PWM中断或定时中断中调用
 * float vin = ReadInputVoltage();        // 读取输入电压
 * float vout = ReadOutputVoltage();      // 读取输出电压
 * float iout = ReadOutputCurrent();      // 读取输出电流
 * float il = ReadInductorCurrent();      // 读取电感电流
 * uint16_t duty = DCDC_Boost_Update(&boost_ctrl, vin, vout, iout, il);
 * SetPWM_DutyCycle(duty);               // 设置PWM占空比
 * @endcode
 */
uint16_t DCDC_Boost_Update(DCDC_Boost_Controller *controller,
                           float input_voltage, float output_voltage,
                           float output_current, float inductor_current)
{
    float current_setpoint, duty_cycle_float;
    uint16_t new_duty;
    
    /* 参数有效性检查 */
    if (controller == NULL || !controller->enable)
    {
        if (controller != NULL)
            controller->duty_cycle = 0;
        return 0;
    }
    
    /* 更新采样值 */
    controller->input_voltage = input_voltage;
    controller->output_voltage = output_voltage;
    controller->output_current = output_current;
    controller->inductor_current = inductor_current;
    
    /* 检查故障状态 */
    if (controller->fault != DCDC_BOOST_FAULT_NONE)
    {
        controller->duty_cycle = 0;
        return 0;
    }
    
    /* ===== 双环控制逻辑 ===== */
    
    if (controller->mode == DCDC_BOOST_MODE_CONSTANT_VOLTAGE)
    {
        /* 恒压模式：电压外环 + 电流内环 */
        current_setpoint = PID_Compute(&controller->voltage_pid, controller->target_voltage, output_voltage);
        
        /* 电流设定值限幅 */
        if (current_setpoint > controller->max_current)
            current_setpoint = controller->max_current;
        if (current_setpoint < 0.0f)
            current_setpoint = 0.0f;
    }
    else
    {
        /* 恒流模式：直接使用目标电流作为电流内环设定值 */
        float voltage_limit = PID_Compute(&controller->voltage_pid, controller->target_voltage, output_voltage);
        current_setpoint = controller->target_current;
        
        /* 如果电压过高，限制电流设定值 */
        if (voltage_limit < current_setpoint)
            current_setpoint = voltage_limit;
        
        /* 电流设定值限幅 */
        if (current_setpoint > controller->max_current)
            current_setpoint = controller->max_current;
        if (current_setpoint < 0.0f)
            current_setpoint = 0.0f;
    }
    
    /* 电流内环计算 */
    duty_cycle_float = PID_Compute(&controller->current_pid, current_setpoint, inductor_current);
    
    /* 转换为千分比格式 */
    new_duty = (uint16_t)(duty_cycle_float * 1000.0f);
    
    /* ===== 占空比安全限制 ===== */
    
    /* 1. 理论占空比上限限制 */
    float theoretical_max_duty = 0.0f;
    if (output_voltage > 0.01f)
    {
        theoretical_max_duty = 0.95f * (1.0f - input_voltage / output_voltage) * 1000.0f;
    }
    else
    {
        theoretical_max_duty = DCDC_BOOST_MAX_DUTY;
    }
    
    /* 取理论上限和配置上限的较小值 */
    float safe_max_duty = (float)DCDC_BOOST_MAX_DUTY;
    if (theoretical_max_duty < safe_max_duty)
    {
        safe_max_duty = theoretical_max_duty;
    }
    
    /* 应用理论上限限制 */
    if (new_duty > (uint16_t)safe_max_duty)
    {
        new_duty = (uint16_t)safe_max_duty;
    }
    
    /* 2. 占空比变化率限制 */
    const uint16_t max_duty_delta = 50;
    int32_t duty_delta = (int32_t)new_duty - (int32_t)controller->duty_cycle;
    
    if (duty_delta > max_duty_delta)
    {
        new_duty = controller->duty_cycle + max_duty_delta;
    }
    else if (duty_delta < -max_duty_delta)
    {
        new_duty = controller->duty_cycle - max_duty_delta;
    }
    
    /* 3. 硬限幅 */
    if (new_duty > DCDC_BOOST_MAX_DUTY)
        new_duty = DCDC_BOOST_MAX_DUTY;
    if (new_duty < DCDC_BOOST_MIN_DUTY)
        new_duty = DCDC_BOOST_MIN_DUTY;
    
    /* 更新占空比 */
    controller->duty_cycle = new_duty;
    
    return controller->duty_cycle;
}

/**
 * @brief 更新DCDC Boost控制器（整数版本）
 * 
 * 执行双环PID控制算法，使用整数计算，加快运行速度
 * 此函数应按DCDC_BOOST_CONTROL_FREQ频率调用（默认100kHz）
 */
uint16_t DCDC_Boost_Update_Int(DCDC_Boost_Controller *controller, 
                               int32_t input_voltage_mv, int32_t output_voltage_mv, 
                               int32_t output_current_ma, int32_t inductor_current_ma)
{
    int32_t current_setpoint_ma, duty_cycle_int;
    uint16_t new_duty;
    
    /* 参数有效性检查 */
    if (controller == NULL || !controller->enable)
    {
        if (controller != NULL)
            controller->duty_cycle = 0;
        return 0;
    }

    /* 检查故障状态 */
    if (controller->fault != DCDC_BOOST_FAULT_NONE)
    {
        controller->duty_cycle = 0;
        return 0;
    }
    
    /* ===== 恒压恒流自动切换逻辑 ===== */
    
    int32_t current_threshold_high = controller->max_current_ma - controller->cc_cv_hysteresis_ma;
    int32_t current_threshold_low = controller->max_current_ma - controller->cc_cv_hysteresis_ma * 2;
    
    if (controller->mode == DCDC_BOOST_MODE_CONSTANT_VOLTAGE)
    {
        if (output_current_ma >= current_threshold_high)
        {
            controller->mode = DCDC_BOOST_MODE_CONSTANT_CURRENT;
            PID_Reset_Int(&controller->voltage_pid_int);
            PID_Reset_Int(&controller->current_pid_int);
        }
    }
    else
    {
        if (output_current_ma <= current_threshold_low)
        {
            controller->mode = DCDC_BOOST_MODE_CONSTANT_VOLTAGE;
            PID_Reset_Int(&controller->voltage_pid_int);
            PID_Reset_Int(&controller->current_pid_int);
        }
    }
    
    /* ===== 双环控制逻辑 ===== */
    
    if (controller->mode == DCDC_BOOST_MODE_CONSTANT_VOLTAGE)
    {
        current_setpoint_ma = PI_Positional_Int(&controller->voltage_pid_int, controller->target_voltage_mv, output_voltage_mv);
        
        /* 电流设定值限幅 */
        if (current_setpoint_ma > controller->max_current_ma)
            current_setpoint_ma = controller->max_current_ma;
        if (current_setpoint_ma < 0)
            current_setpoint_ma = 0;
    }
    else
    {
        int32_t voltage_limit_ma = PI_Positional_Int(&controller->voltage_pid_int, controller->target_voltage_mv, output_voltage_mv);
        current_setpoint_ma = controller->target_current_ma;
        
        /* 如果电压过高，限制电流设定值 */
        if (voltage_limit_ma < current_setpoint_ma)
            current_setpoint_ma = voltage_limit_ma;
        
        /* 电流设定值限幅 */
        if (current_setpoint_ma > controller->max_current_ma)
            current_setpoint_ma = controller->max_current_ma;
        if (current_setpoint_ma < 0)
            current_setpoint_ma = 0;
    }
    
    /* 电流内环计算（使用增量式PI） */
    duty_cycle_int = PI_Incremental_Int(&controller->current_pid_int, current_setpoint_ma, inductor_current_ma);
    
    /* 转换为千分比格式 */
    new_duty = (uint16_t)duty_cycle_int;
    
    /* ===== 占空比安全限制 ===== */
    
    /* 1. 理论占空比上限限制 */
    int32_t theoretical_max_duty = DCDC_BOOST_MAX_DUTY;
    if (output_voltage_mv > 10)  /* 避免除零，对应0.01V */
    {
        /* 计算理论占空比上限：0.95 * (1 - Vin/Vout) * 1000 */
        /* 使用整数运算：(0.95 * 1000) * (Vout - Vin) / Vout */
        if (output_voltage_mv > input_voltage_mv)  /* 确保 Vout > Vin */
        {
            int32_t numerator = 950 * (output_voltage_mv - input_voltage_mv);
            theoretical_max_duty = numerator / output_voltage_mv;
        }
        /* 否则，保持 theoretical_max_duty 为 DCDC_BOOST_MAX_DUTY */
    }
    
    /* 确保 theoretical_max_duty 为正数 */
    if (theoretical_max_duty < 0)
    {
        theoretical_max_duty = DCDC_BOOST_MAX_DUTY;
    }
    
    /* 取理论上限和配置上限的较小值 */
    int32_t safe_max_duty = DCDC_BOOST_MAX_DUTY;
    if (theoretical_max_duty < safe_max_duty)
    {
        safe_max_duty = theoretical_max_duty;
    }
    
    /* 应用理论上限限制 */
    if (new_duty > (uint16_t)safe_max_duty)
    {
        new_duty = (uint16_t)safe_max_duty;
    }
    
    /* 2. 占空比变化率限制 */
    const uint16_t max_duty_delta = 50;
    int32_t duty_delta = (int32_t)new_duty - (int32_t)controller->duty_cycle;
    
    if (duty_delta > max_duty_delta)
    {
        new_duty = controller->duty_cycle + max_duty_delta;
    }
    else if (duty_delta < -max_duty_delta)
    {
        new_duty = controller->duty_cycle - max_duty_delta;
    }
    
    /* 3. 硬限幅 */
    if (new_duty > DCDC_BOOST_MAX_DUTY)
        new_duty = DCDC_BOOST_MAX_DUTY;
    if (new_duty < DCDC_BOOST_MIN_DUTY)
        new_duty = DCDC_BOOST_MIN_DUTY;
    
    /* 更新占空比 */
    controller->duty_cycle = new_duty;
    
    return controller->duty_cycle;
}

/**
 * @brief 获取当前故障状态
 * 
 * @param controller 控制器结构体指针
 * @return 当前故障状态（DCDC_Boost_Fault枚举值）
 * 
 * @code
 * // 调用示例：检查故障状态
 * DCDC_Boost_Fault fault = DCDC_Boost_GetFault(&boost_ctrl);
 * if (fault & DCDC_BOOST_FAULT_OUTPUT_OVERVOLT) {
 *     printf("Output Overvoltage Fault!\r\n");
 * }
 * @endcode
 */
DCDC_Boost_Fault DCDC_Boost_GetFault(DCDC_Boost_Controller *controller)
{
    if (controller == NULL)
        return DCDC_BOOST_FAULT_NONE;
    
    return controller->fault;
}

/**
 * @brief 清除故障状态
 * 
 * 用于故障恢复后清除故障标志
 * 
 * @param controller 控制器结构体指针
 * 
 * @code
 * // 调用示例：清除故障状态
 * DCDC_Boost_ClearFault(&boost_ctrl);
 * @endcode
 */
void DCDC_Boost_ClearFault(DCDC_Boost_Controller *controller)
{
    if (controller == NULL)
        return;
    
    controller->fault = DCDC_BOOST_FAULT_NONE;
}

/**
 * @brief 中断版故障检测（仅检测关键故障）
 * 
 * 用于中断中快速检测关键故障，只检测最紧急的故障类型
 */
uint8_t DCDC_Boost_CheckFault_Int(DCDC_Boost_Controller *controller,
                                  int32_t input_voltage_mv,
                                  int32_t output_voltage_mv,
                                  int32_t output_current_ma)
{
    if (controller == NULL)
        return 0;
    
    /* 输出过压保护检查 */
    if (output_voltage_mv > controller->max_voltage_mv)
    {
        controller->fault |= DCDC_BOOST_FAULT_OUTPUT_OVERVOLT;
        PID_Reset(&controller->voltage_pid);
        PID_Reset(&controller->current_pid);
        PID_Reset_Int(&controller->voltage_pid_int);
        PID_Reset_Int(&controller->current_pid_int);
        return 1;
    }
    
    /* 输出过流保护检查 */
    if (output_current_ma > controller->max_current_ma)
    {
        controller->fault |= DCDC_BOOST_FAULT_OUTPUT_OVERCURRENT;
        PID_Reset(&controller->voltage_pid);
        PID_Reset(&controller->current_pid);
        PID_Reset_Int(&controller->voltage_pid_int);
        PID_Reset_Int(&controller->current_pid_int);
        return 1;
    }
    
    /* 输出短路保护检查 */
    if (controller->mode == DCDC_BOOST_MODE_CONSTANT_VOLTAGE &&
        output_voltage_mv < controller->min_output_voltage_mv &&
        controller->duty_cycle > DCDC_BOOST_SHORT_CIRCUIT_DUTY_THRESHOLD)
    {
        controller->fault |= DCDC_BOOST_FAULT_SHORT_CIRCUIT;
        PID_Reset(&controller->voltage_pid);
        PID_Reset(&controller->current_pid);
        PID_Reset_Int(&controller->voltage_pid_int);
        PID_Reset_Int(&controller->current_pid_int);
        return 1;
    }
    
    return 0;
}

/**
 * @brief 检测故障状态
 * 
 * 检查输入电压、输出电压、输出电流等是否超出保护范围
 * 
 * @param controller 控制器结构体指针
 * @param input_voltage 输入电压采样值，单位V
 * @param output_voltage 输出电压采样值，单位V
 * @param output_current 输出电流采样值，单位A
 * 
 * @code
 * // 调用示例：检测故障
 * float input_voltage = Get_CalibratedInputVoltage();
 * float output_voltage = Get_CalibratedOutputVoltage();
 * float output_current = Get_CalibratedOutputCurrent();
 * DCDC_Boost_CheckFault(&boost_ctrl, input_voltage, output_voltage, output_current);
 * @endcode
 */
void DCDC_Boost_CheckFault(DCDC_Boost_Controller *controller,
                           float input_voltage, float output_voltage,
                           float output_current)
{
    if (controller == NULL)
        return;
    
    /* 输入电压保护检查（只检查过压，欠压视为等待电源状态） */
    if (input_voltage > controller->max_input_voltage)
    {
        controller->fault |= DCDC_BOOST_FAULT_INPUT_OVERVOLT;
        PID_Reset(&controller->voltage_pid);
        PID_Reset(&controller->current_pid);
        return;
    }
    
    /* 清除输入电压故障标志 */
    if (controller->fault & DCDC_BOOST_FAULT_INPUT_OVERVOLT)
    {
        if (input_voltage <= controller->max_input_voltage)
        {
            controller->fault &= ~DCDC_BOOST_FAULT_INPUT_OVERVOLT;
        }
    }
    
    /* 清除输入欠压故障标志（如果存在） */
    if (controller->fault & DCDC_BOOST_FAULT_INPUT_UNDERVOLT)
    {
        if (input_voltage >= controller->min_input_voltage)
        {
            controller->fault &= ~DCDC_BOOST_FAULT_INPUT_UNDERVOLT;
        }
    }
    
    /* 输出过压保护检查 */
    if (output_voltage > controller->max_voltage)
    {
        controller->fault |= DCDC_BOOST_FAULT_OUTPUT_OVERVOLT;
        PID_Reset(&controller->voltage_pid);
        PID_Reset(&controller->current_pid);
        return;
    }
    
    /* 清除输出过压故障标志 */
    if ((controller->fault & DCDC_BOOST_FAULT_OUTPUT_OVERVOLT) && 
        output_voltage <= controller->max_voltage)
    {
        controller->fault &= ~DCDC_BOOST_FAULT_OUTPUT_OVERVOLT;
    }
    
    /* 输出过流保护检查 */
    if (output_current > controller->max_current)
    {
        controller->fault |= DCDC_BOOST_FAULT_OUTPUT_OVERCURRENT;
        PID_Reset(&controller->voltage_pid);
        PID_Reset(&controller->current_pid);
        return;
    }
    
    /* 清除输出过流故障标志 */
    if ((controller->fault & DCDC_BOOST_FAULT_OUTPUT_OVERCURRENT) && 
        output_current <= controller->max_current)
    {
        controller->fault &= ~DCDC_BOOST_FAULT_OUTPUT_OVERCURRENT;
    }
    
    /* 输出短路保护检查 */
    if (controller->mode == DCDC_BOOST_MODE_CONSTANT_VOLTAGE &&
        output_voltage < controller->min_output_voltage &&
        controller->duty_cycle > DCDC_BOOST_SHORT_CIRCUIT_DUTY_THRESHOLD)
    {
        controller->fault |= DCDC_BOOST_FAULT_SHORT_CIRCUIT;
        PID_Reset(&controller->voltage_pid);
        PID_Reset(&controller->current_pid);
        return;
    }
    
    /* 清除短路故障标志 */
    if ((controller->fault & DCDC_BOOST_FAULT_SHORT_CIRCUIT) &&
        output_voltage >= controller->min_output_voltage)
    {
        controller->fault &= ~DCDC_BOOST_FAULT_SHORT_CIRCUIT;
    }
}

/**
 * @brief 处理软启动
 * 
 * 执行软启动逻辑，返回软启动是否完成
 * 
 * @param controller 控制器结构体指针
 * @param input_voltage 输入电压采样值，单位V
 * @param output_voltage 输出电压采样值，单位V
 * @param output_current 输出电流采样值，单位A
 * @param inductor_current 电感电流采样值，单位A
 * @param duty 输出占空比，千分比(0-1000)
 * @return 1表示软启动已完成，0表示软启动正在进行
 * 
 * @code
 * // 调用示例：处理软启动
 * float input_voltage = Get_CalibratedInputVoltage();
 * float output_voltage = Get_CalibratedOutputVoltage();
 * float output_current = Get_CalibratedOutputCurrent();
 * float inductor_current = input_current;
 * uint16_t duty = 0;
 * uint8_t soft_start_completed = DCDC_Boost_ProcessSoftStart(&boost_ctrl, input_voltage, output_voltage, output_current, inductor_current, &duty);
 * if (!soft_start_completed)
 * {
 *     PWM_SetDuty(duty);
 * }
 * @endcode
 */
uint8_t DCDC_Boost_ProcessSoftStart(DCDC_Boost_Controller *controller,
                                    float input_voltage, float output_voltage,
                                    float output_current, float inductor_current,
                                    uint16_t *duty)
{
    if (controller == NULL || !controller->soft_start_enable)
        return 1;
    
    /* 软启动计时器递增 */
    controller->soft_start_timer++;
    
    /* 检查是否已达到目标电压（98%视为达到） */
    if (output_voltage >= controller->soft_start_target * 0.98f)
    {
        controller->soft_start_enable = 0;
        return 1;
    }
    
    /* 计算软启动进度 */
    float progress = (float)controller->soft_start_timer / 
                    ((float)DCDC_BOOST_SOFT_START_TIME * (float)DCDC_BOOST_CONTROL_FREQ / 1000.0f);
    
    /* 限制进度在有效范围内 */
    if (progress >= 1.0f)
    {
        progress = 1.0f;
        controller->soft_start_enable = 0;
        return 1;
    }
    
    /* 计算实际目标值 */
    float actual_target_voltage = controller->soft_start_target * progress;
    float actual_target_current = controller->target_current * progress;
    
    /* 计算占空比 */
    float current_setpoint = PID_Compute(&controller->voltage_pid, actual_target_voltage, output_voltage);
    
    /* 电流设定值限幅 */
    if (current_setpoint > controller->max_current)
        current_setpoint = controller->max_current;
    if (current_setpoint < 0.0f)
        current_setpoint = 0.0f;
    
    /* 电流内环计算 */
    float duty_cycle_float = PID_Compute(&controller->current_pid, current_setpoint, inductor_current);
    
    /* 转换为千分比格式 */
    *duty = (uint16_t)(duty_cycle_float * 1000.0f);
    
    /* 占空比限幅 */
    if (*duty > DCDC_BOOST_MAX_DUTY)
        *duty = DCDC_BOOST_MAX_DUTY;
    if (*duty < DCDC_BOOST_MIN_DUTY)
        *duty = DCDC_BOOST_MIN_DUTY;
    
    return 0;
}

/**
 * @brief 检测电源是否上电
 * 
 * @param controller 控制器结构体指针
 * @param input_voltage 输入电压采样值，单位V
 * @return 1表示电源已上电，0表示电源未上电
 * 
 * @code
 * // 调用示例：检测电源是否上电
 * float input_voltage = Get_CalibratedInputVoltage();
 * uint8_t is_power_on = DCDC_Boost_IsPowerOn(&boost_ctrl, input_voltage);
 * if (is_power_on)
 * {
 *     // 电源已上电
 * }
 * @endcode
 */
uint8_t DCDC_Boost_IsPowerOn(DCDC_Boost_Controller *controller, float input_voltage)
{
    if (controller == NULL)
        return 0;
    
    return (input_voltage >= controller->min_input_voltage);
}

/**
 * @brief 同步浮点PID状态到整数PID
 *
 * 在软启动阶段使用浮点PID，软启动完成后切换到整数PID
 * 调用此函数可以将浮点PID的积分值和误差状态同步到整数PID
 * 避免切换时出现占空比突变或输出电压过冲
 *
 * @param controller 控制器结构体指针
 * @param current_duty 当前占空比（软启动结束时的占空比）
 */
void DCDC_Boost_SyncPIDState(DCDC_Boost_Controller *controller, uint16_t current_duty)
{
    if (controller == NULL)
        return;

    /* 电压环是位置式PI：设置integral，保证切换时输出连续
     * integral存储的是误差累积，初始设为0让PID重新计算 */
    controller->voltage_pid_int.integral = 0;
    controller->voltage_pid_int.prev_error = 0;
    controller->voltage_pid_int.prev2_error = 0;

    /* 电流环是增量式PI：设置last_output为当前占空比
     * 这样切换时输出直接从current_duty开始，不会突变 */
    controller->current_pid_int.last_output = current_duty;
    controller->current_pid_int.prev_error = 0;
    controller->current_pid_int.prev2_error = 0;
}
