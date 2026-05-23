#ifndef __BSP_DCDC_BOOST_H
#define __BSP_DCDC_BOOST_H

#include <stdint.h>
#include <stddef.h>
#include "stm32f10x.h"

/**
 * @brief DCDC Boost 控制器配置参数宏定义
 * 
 * 这些宏定义了控制器的基本配置参数，可根据实际硬件需求进行修改
 */
#define DCDC_BOOST_SOFT_START_TIME    100     /* 软启动时间，单位ms，默认100ms */
#define DCDC_BOOST_CONTROL_FREQ       100000  /* 控制频率，单位Hz，与PWM开关频率一致(100kHz) */
#define DCDC_BOOST_MAX_DUTY           750     /* 最大占空比，千分比(0-1000)，75% - 保守值 */
                                              /* 说明：比额定80%高5%，留保护余量 */
#define DCDC_BOOST_MIN_DUTY           50      /* 最小占空比，千分比(0-1000)，默认5% */

/**
 * @brief 安全系数配置（保护限值 = 额定值 × 安全系数）
 * 
 * 参考工业级DCDC设计规范：
 * - 输出过压保护(OVP)：通常设置为额定电压的105%-110%
 * - 输出过流保护(OCP)：通常设置为额定电流的110%-120%
 * - 输入过压保护(IVP)：通常设置为额定输入的110%
 * - 输入欠压保护(UVP)：通常设置为额定输入的85%-90%
 */
#define DCDC_BOOST_SAFETY_FACTOR_VOLTAGE    1.10f   /* 输出电压保护系数（+10%）- OVP */
#define DCDC_BOOST_SAFETY_FACTOR_CURRENT    1.10f   /* 输出电流保护系数（+10%）- OCP */
#define DCDC_BOOST_SAFETY_FACTOR_INPUT_HIGH 1.10f   /* 输入电压上限保护系数（+10%）- IVP */
#define DCDC_BOOST_SAFETY_FACTOR_INPUT_LOW  0.85f   /* 输入电压下限保护系数（-15%）- UVP */

/**
 * @brief 短路检测阈值（参考业界标准）
 * 
 * 短路检测原理：当输出电压低于阈值且占空比接近最大时，判定为短路
 * 通常设置为2-3V，确保与恒流模式下的正常电压降低区分开
 */
#define DCDC_BOOST_SHORT_CIRCUIT_THRESHOLD  2.0f    /* 短路检测电压阈值，单位V */
#define DCDC_BOOST_SHORT_CIRCUIT_DUTY_THRESHOLD 750 /* 短路检测占空比阈值，千分比(75%) */

/**
 * @brief 电源检测参数
 * 
 * 用于检测输入电源是否上电，以及软启动相关参数
 */
#define DCDC_BOOST_POWER_ON_THRESHOLD_V 8.0f  /* 电源上电阈值，单位V */
#define DCDC_BOOST_SOFT_START_TIME_MS 100    /* 软启动时间，单位ms */
#define DCDC_BOOST_SOFT_START_STEP_MS 10      /* 软启动步长，单位ms */

/**
 * @brief 默认参数定义
 * 
 * 控制器初始化时使用的默认参数值
 * 根据DCDC Boost电路参数设计：
 *   Vin=13V, Vout=48V, L=10μH, C=387μF, fs=100kHz
 *   RL=0.015Ω, ESR=0.02Ω
 */
/* 电压环PI参数 */
#define DCDC_BOOST_DEFAULT_VOLTAGE_KP   0.2f    /* 默认电压环比例系数 (Kp_v = 0.2) - 保守值 */
#define DCDC_BOOST_DEFAULT_VOLTAGE_KI   20.0f   /* 默认电压环积分系数 (Ki_v = 20) - 保守值 */
#define DCDC_BOOST_DEFAULT_VOLTAGE_KD   0.0f    /* 默认电压环微分系数（不使用微分） */

/* 电流环PI参数 */
#define DCDC_BOOST_DEFAULT_CURRENT_KP   0.05f   /* 默认电流环比例系数 (Kp_i = 0.05) - 保守值 */
#define DCDC_BOOST_DEFAULT_CURRENT_KI   50.0f   /* 默认电流环积分系数 (Ki_i = 50) - 保守值 */
#define DCDC_BOOST_DEFAULT_CURRENT_KD   0.0f    /* 默认电流环微分系数（不使用微分） */

/* 默认额定参数（宽电压输入：13-16V） */
#define DCDC_BOOST_DEFAULT_RATED_VOLTAGE     24.0f   /* 默认额定输出电压，单位V */
#define DCDC_BOOST_DEFAULT_RATED_CURRENT     6.0f    /* 默认额定输出电流，单位A */
#define DCDC_BOOST_DEFAULT_RATED_INPUT_VOLTAGE 16.0f  /* 默认额定输入电压，单位V（宽范围13-16V） */

/**
 * @brief 宽电压输入范围配置（8V-16V）
 * 
 * 针对宽电压输入应用场景的推荐配置：
 * - 典型输入范围：8V ~ 16V
 * - 保护上限：17.6V（16V × 1.1）
 * - 保护下限：6.8V（8V × 0.85）
 */
#define DCDC_BOOST_WIDE_INPUT_MIN_VOLTAGE    8.0f   /* 宽输入电压下限，单位V */
#define DCDC_BOOST_WIDE_INPUT_MAX_VOLTAGE    16.0f   /* 宽输入电压上限，单位V */

/* 保护限制参数（自动计算：保护限值 = 额定值 × 安全系数） */
#define DCDC_BOOST_DEFAULT_MAX_VOLTAGE       (DCDC_BOOST_DEFAULT_RATED_VOLTAGE * DCDC_BOOST_SAFETY_FACTOR_VOLTAGE)
#define DCDC_BOOST_DEFAULT_MAX_CURRENT       (DCDC_BOOST_DEFAULT_RATED_CURRENT * DCDC_BOOST_SAFETY_FACTOR_CURRENT)
#define DCDC_BOOST_DEFAULT_MAX_INPUT_VOLTAGE (DCDC_BOOST_DEFAULT_RATED_INPUT_VOLTAGE * DCDC_BOOST_SAFETY_FACTOR_INPUT_HIGH)
#define DCDC_BOOST_DEFAULT_MIN_INPUT_VOLTAGE ((DCDC_BOOST_WIDE_INPUT_MIN_VOLTAGE) * DCDC_BOOST_SAFETY_FACTOR_INPUT_LOW)

/**
 * @brief PID控制器结构体
 * 
 * 用于存储PID控制器的参数和状态信息
 */
typedef struct
{
    float Kp;          /* 比例系数，可通过DCDC_Boost_SetPIDParams()设置 */
    float Ki;          /* 积分系数，可通过DCDC_Boost_SetPIDParams()设置 */
    float Kd;          /* 微分系数，可通过DCDC_Boost_SetPIDParams()设置 */
    float integral;    /* 积分值，内部状态变量 */
    float prev_error;  /* 上一次误差，内部状态变量 */
    float max_output;  /* 最大输出限幅，默认1.0 */
    float min_output;  /* 最小输出限幅，默认0.0 */
} PID_Controller;

/**
 * @brief 整数版PID控制器结构体（用于纯整数运算）
 * 
 * 所有值均为Q12格式（整数部分8位，小数部分12位）
 * 支持增量式PI（电流环）和位置式PI（电压环）两种算法
 */
typedef struct
{
    int32_t Kp;        /* 比例系数（Q12格式） */
    int32_t Ki;        /* 积分系数（Q12格式） */
    int32_t Kd;        /* 微分系数（Q12格式） */
    int32_t integral;  /* 积分值（Q12格式） */
    int32_t prev_error;/* 上一次误差（Q12格式） */
    int32_t prev2_error;/* 上上次误差（Q12格式，用于增量式PI） */
    int32_t last_output;/* 最后一次输出（物理单位，用于增量式PI） */
    int32_t max_output;/* 最大输出限幅（Q12格式） */
    int32_t min_output;/* 最小输出限幅（Q12格式） */
} PID_Controller_Int;

/**
 * @brief 控制器工作模式枚举
 */
typedef enum {
    DCDC_BOOST_MODE_CONSTANT_VOLTAGE = 0,  /* 恒压模式（默认） */
    DCDC_BOOST_MODE_CONSTANT_CURRENT = 1   /* 恒流模式 */
} DCDC_Boost_Mode;

/**
 * @brief 保护状态标志枚举
 * 
 * 用于指示当前触发的保护类型，便于外部诊断和故障排查
 */
typedef enum {
    DCDC_BOOST_FAULT_NONE = 0,        /* 无故障 */
    DCDC_BOOST_FAULT_INPUT_OVERVOLT = 1,   /* 输入过压 */
    DCDC_BOOST_FAULT_INPUT_UNDERVOLT = 2,  /* 输入欠压 */
    DCDC_BOOST_FAULT_OUTPUT_OVERVOLT = 4,  /* 输出过压 */
    DCDC_BOOST_FAULT_OUTPUT_OVERCURRENT = 8, /* 输出过流 */
    DCDC_BOOST_FAULT_SHORT_CIRCUIT = 16    /* 输出短路 */
} DCDC_Boost_Fault;

/**
 * @brief DCDC Boost双环控制器结构体
 * 
 * 包含电压外环和电流内环的完整控制器状态
 * 所有参数均可通过对应的Set函数进行外部配置
 */
typedef struct
{
    PID_Controller voltage_pid;           /* 电压外环PID控制器，默认参数见DEFAULT宏定义 */
    PID_Controller current_pid;           /* 电流内环PID控制器，默认参数见DEFAULT宏定义 */
    PID_Controller_Int voltage_pid_int;   /* 电压外环整数版PID控制器 */
    PID_Controller_Int current_pid_int;   /* 电流内环整数版PID控制器 */
    
    float target_voltage;         /* 目标输出电压，单位V，默认0V，可通过SetTargetVoltage()设置 */
    float target_current;         /* 目标输出电流，单位A，默认0A，可通过SetTargetCurrent()设置 */
    
    int32_t target_voltage_mv;    /* 目标电压预计算值（mV），中断中直接使用，无浮点运算 */
    int32_t target_current_ma;    /* 目标电流预计算值（mA），中断中直接使用，无浮点运算 */
    int32_t max_current_ma;       /* 最大电流预计算值（mA），中断中直接使用，无浮点运算 */
    
    float input_voltage;          /* 输入电压采样值，单位V，由Update()函数传入 */
    float output_voltage;         /* 输出电压采样值，单位V，由Update()函数传入 */
    float output_current;         /* 输出电流采样值，单位A，由Update()函数传入 */
    float inductor_current;       /* 电感电流采样值，单位A，由Update()函数传入 */
    
    /* 保护限制参数（参考工业级DCDC设计规范） */
    float max_voltage;            /* 输出电压保护上限(OVP)，单位V，应高于额定电压 */
    float max_current;            /* 输出电流保护上限(OCP)，单位A，应高于额定电流 */
    float max_input_voltage;      /* 输入电压保护上限(IVP)，单位V，应高于额定输入电压 */
    float min_input_voltage;      /* 输入电压保护下限(UVP)，单位V，应低于额定输入电压 */
    float min_output_voltage;     /* 输出电压短路检测阈值，单位V，建议2-3V */
    
    /* 预计算的保护值（mV/mA），中断中直接使用，无浮点运算 */
    int32_t max_voltage_mv;       /* 最大输出电压预计算值（mV） */
    int32_t min_output_voltage_mv; /* 最小输出电压预计算值（mV） */
    
    uint16_t duty_cycle;          /* 当前占空比输出，千分比(0-1000)，由Update()函数计算得出 */
    
    DCDC_Boost_Mode mode;         /* 工作模式：恒压/恒流，默认恒压 */
    
    /* 恒压恒流自动切换参数 */
    int32_t cc_cv_hysteresis_ma; /* 模式切换滞回电流值（mA），默认100mA */
    DCDC_Boost_Mode prev_mode;    /* 上一次的工作模式，用于检测切换 */
    uint8_t soft_start_enable;    /* 软启动使能标志，0=禁用，1=启用，可通过EnableSoftStart()设置 */
    uint32_t soft_start_timer;    /* 软启动计时器，内部状态变量 */
    float soft_start_target;      /* 软启动目标电压，单位V，由EnableSoftStart()设置 */
    
    uint8_t enable;               /* 控制器使能标志，0=禁用，1=启用，可通过Enable/Disable()设置 */
    
    /* 电源检测标志 */
    uint8_t power_on_detected;     /* 输入电源检测标志，0=未检测到，1=已检测到 */
    
    /* 保护状态标志 */
    DCDC_Boost_Fault fault;       /* 当前故障状态，可通过GetFault()获取 */
} DCDC_Boost_Controller;

/**
 * @brief 初始化DCDC Boost控制器
 * 
 * 将控制器的所有参数设置为默认值，必须在使用前调用
 * 
 * @param controller 控制器结构体指针
 * 
 * @code
 * // 调用示例
 * DCDC_Boost_Controller boost_ctrl;
 * DCDC_Boost_Init(&boost_ctrl);
 * @endcode
 */
void DCDC_Boost_Init(DCDC_Boost_Controller *controller);

/**
 * @brief 设置PID控制器参数
 * 
 * 允许外部修改电压环和电流环的PID参数，如果不调用此函数，将使用默认参数值
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
                             float current_kp, float current_ki, float current_kd);

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
                               float rated_input_voltage);

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
                          float min_output_voltage);

/**
 * @brief 设置控制器工作模式
 * 
 * @param controller 控制器结构体指针
 * @param mode 工作模式：DCDC_BOOST_MODE_CONSTANT_VOLTAGE（恒压）或DCDC_BOOST_MODE_CONSTANT_CURRENT（恒流）
 * 
 * @code
 * // 调用示例：设置为恒流模式
 * DCDC_Boost_SetMode(&boost_ctrl, DCDC_BOOST_MODE_CONSTANT_CURRENT);
 * 
 * // 调用示例：设置为恒压模式
 * DCDC_Boost_SetMode(&boost_ctrl, DCDC_BOOST_MODE_CONSTANT_VOLTAGE);
 * @endcode
 */
void DCDC_Boost_SetMode(DCDC_Boost_Controller *controller, DCDC_Boost_Mode mode);

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
void DCDC_Boost_SetTargetVoltage(DCDC_Boost_Controller *controller, float voltage);

/**
 * @brief 设置目标输出电流（限流值）
 * 
 * 设置值会自动限制在0到max_current之间，用于限流保护
 * 
 * @param controller 控制器结构体指针
 * @param current 目标电流值，单位A
 * 
 * @code
 * // 调用示例：设置限流值为3A
 * DCDC_Boost_SetTargetCurrent(&boost_ctrl, 3.0f);
 * @endcode
 */
void DCDC_Boost_SetTargetCurrent(DCDC_Boost_Controller *controller, float current);

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
void DCDC_Boost_EnableSoftStart(DCDC_Boost_Controller *controller, float target);

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
void DCDC_Boost_DisableSoftStart(DCDC_Boost_Controller *controller);

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
void DCDC_Boost_Enable(DCDC_Boost_Controller *controller);

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
void DCDC_Boost_Disable(DCDC_Boost_Controller *controller);

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
DCDC_Boost_Fault DCDC_Boost_GetFault(DCDC_Boost_Controller *controller);

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
void DCDC_Boost_ClearFault(DCDC_Boost_Controller *controller);

/**
 * @brief 更新DCDC Boost控制器（主控制函数）
 * 
 * 执行双环PID控制算法，此函数应按DCDC_BOOST_CONTROL_FREQ频率调用（默认100kHz）
 * 
 * @param controller 控制器结构体指针
 * @param input_voltage 输入电压采样值，单位V
 * @param output_voltage 输出电压采样值，单位V
 * @param output_current 输出电流采样值，单位A
 * @param inductor_current 电感电流采样值，单位A
 * @return 当前占空比，千分比(0-1000)
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
                           float output_current, float inductor_current);

/**
 * @brief 更新DCDC Boost控制器（整数版本）
 * 
 * 执行双环PID控制算法，使用整数计算，加快运行速度
 * 此函数应按DCDC_BOOST_CONTROL_FREQ频率调用（默认100kHz）
 * 
 * @param controller 控制器结构体指针
 * @param input_voltage_mv 输入电压采样值，单位mV
 * @param output_voltage_mv 输出电压采样值，单位mV
 * @param output_current_ma 输出电流采样值，单位mA
 * @param inductor_current_ma 电感电流采样值，单位mA
 * @return 当前占空比，千分比(0-1000)
 * 
 * @code
 * // 调用示例：在PWM中断或定时中断中调用
 * int32_t vin_mv = Get_CalibratedInputVoltage_Int();        // 读取输入电压（毫伏）
 * int32_t vout_mv = Get_CalibratedOutputVoltage_Int();      // 读取输出电压（毫伏）
 * int32_t iout_ma = Get_CalibratedOutputCurrent_Int();      // 读取输出电流（毫安）
 * int32_t il_ma = Get_CalibratedInputCurrent_Int();        // 读取电感电流（毫安）
 * uint16_t duty = DCDC_Boost_Update_Int(&boost_ctrl, vin_mv, vout_mv, iout_ma, il_ma);
 * SetPWM_DutyCycle(duty);                                 // 设置PWM占空比
 * @endcode
 */
uint16_t DCDC_Boost_Update_Int(DCDC_Boost_Controller *controller, 
                               int32_t input_voltage_mv, int32_t output_voltage_mv, 
                               int32_t output_current_ma, int32_t inductor_current_ma);

/**
 * @brief PID计算函数（浮点版本）
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
float PID_Compute(PID_Controller *pid, float setpoint, float process_value);

/**
 * @brief PID计算函数（整数版本）
 * 
 * 实现标准的增量式PID算法，使用整数计算，提高执行速度
 * 
 * @param pid PID控制器结构体指针
 * @param setpoint 设定值（毫单位）
 * @param process_value 过程值（毫单位）
 * @return PID输出值（千分比）
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
 * int32_t output = PID_Compute_Int(&pid, 100000, process_value); // 100.0V 转换为 100000mV
 * @endcode
 */
int32_t PID_Compute_Int(PID_Controller *pid, int32_t setpoint, int32_t process_value);

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
void PID_Reset(PID_Controller *pid);
void PID_Reset_Int(PID_Controller_Int *pid);

/**
 * @brief 检测电源是否上电
 * 
 * @param controller 控制器结构体指针
 * @param input_voltage 输入电压采样值，单位V
 * @return 1表示电源已上电，0表示电源未上电
 * 
 * @code
 * // 调用示例
 * float input_voltage = ReadInputVoltage();
 * uint8_t is_power_on = DCDC_Boost_IsPowerOn(&boost_ctrl, input_voltage);
 * if (is_power_on) {
 *     // 电源已上电
 * }
 * @endcode
 */
uint8_t DCDC_Boost_IsPowerOn(DCDC_Boost_Controller *controller, float input_voltage);

/**
 * @brief 执行软启动
 * 
 * 软启动功能可以使输出电压从0平滑上升到目标值，避免冲击电流
 * 
 * @param controller 控制器结构体指针
 * @param input_voltage 输入电压采样值，单位V
 * @param output_voltage 输出电压采样值，单位V
 * @param output_current 输出电流采样值，单位A
 * @param inductor_current 电感电流采样值，单位A
 * @return 当前占空比，千分比(0-1000)
 * 
 * @code
 * // 调用示例
 * float input_voltage = ReadInputVoltage();
 * float output_voltage = ReadOutputVoltage();
 * float output_current = ReadOutputCurrent();
 * float inductor_current = ReadInductorCurrent();
 * uint16_t duty = DCDC_Boost_ExecuteSoftStart(&boost_ctrl, input_voltage, output_voltage, output_current, inductor_current);
 * SetPWM_DutyCycle(duty);
 * @endcode
 */
uint16_t DCDC_Boost_ExecuteSoftStart(DCDC_Boost_Controller *controller, 
                                    float input_voltage, float output_voltage, 
                                    float output_current, float inductor_current);

/**
 * @brief 检查软启动是否完成
 * 
 * @param controller 控制器结构体指针
 * @return 1表示软启动已完成，0表示软启动正在进行
 * 
 * @code
 * // 调用示例
 * if (DCDC_Boost_IsSoftStartCompleted(&boost_ctrl)) {
 *     // 软启动已完成，开始正常控制
 * }
 * @endcode
 */
uint8_t DCDC_Boost_IsSoftStartCompleted(DCDC_Boost_Controller *controller);

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
                           float output_current);

/**
 * @brief 中断版故障检测（仅检测关键故障）
 * 
 * 用于中断中快速检测关键故障，只检测最紧急的故障类型
 * 
 * @param controller 控制器结构体指针
 * @param input_voltage_mv 输入电压（毫伏）
 * @param output_voltage_mv 输出电压（毫伏）
 * @param output_current_ma 输出电流（毫安）
 * @return 1表示检测到故障，0表示正常
 * @note 仅检测：输出过压、输出过流、短路
 */
uint8_t DCDC_Boost_CheckFault_Int(DCDC_Boost_Controller *controller,
                                  int32_t input_voltage_mv,
                                  int32_t output_voltage_mv,
                                  int32_t output_current_ma);

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
                                    uint16_t *duty);

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
uint8_t DCDC_Boost_IsPowerOn(DCDC_Boost_Controller *controller, float input_voltage);

/**
 * @brief 同步浮点PID状态到整数PID
 *
 * 在软启动阶段使用浮点PID，软启动完成后切换到整数PID
 * 调用此函数可以将浮点PID的积分值和误差状态同步到整数PID
 * 避免切换时出现占空比突变或输出电压过冲
 *
 * @param controller 控制器结构体指针
 * @param current_duty 当前占空比（软启动结束时的占空比）
 *
 * @code
 * // 调用示例：软启动完成后同步PID状态
 * uint16_t final_duty = 400; // 软启动结束时的占空比
 * DCDC_Boost_SyncPIDState(&boost_ctrl, final_duty);
 * @endcode
 */
void DCDC_Boost_SyncPIDState(DCDC_Boost_Controller *controller, uint16_t current_duty);

#endif /* __BSP_DCDC_BOOST_H */
