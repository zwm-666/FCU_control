#include "bsp_pwm.h"

/*
 * 模块名称：bsp_pwm
 * 模块功能：提供 PWM 输出控制功能，用于 DC-DC 转换器的 MOS 管控制
 * 说明：
 * 1. 使用 TIM3 定时器生成 PWM 波形
 * 2. 支持占空比 0-100% 调节
 * 3. 支持 PWM 使能/禁用控制
 * 4. 提供多种频率配置选项
 * 
 * 使用示例：
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

/* 全局变量 */
static uint16_t g_pwm_period = 0;    /* PWM 周期 */
static uint16_t g_current_duty = 0;  /* 当前占空比 */

/**
  * @brief  初始化 PWM 模块
  * @param  freq_hz: PWM 频率，单位 Hz
  * @retval 无
  */
void PWM_Init(uint32_t freq_hz)
{
    GPIO_InitTypeDef GPIO_InitStructure;
    TIM_TimeBaseInitTypeDef TIM_TimeBaseStructure;
    TIM_OCInitTypeDef TIM_OCInitStructure;

    /* 计算 PWM 周期 */
    g_pwm_period = (72000000UL / freq_hz) - 1;

    /* 使能时钟 */
    RCC_APB1PeriphClockCmd(PWM_TIM_RCC, ENABLE);
    RCC_APB2PeriphClockCmd(PWM_GPIO_RCC, ENABLE);
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO, ENABLE);
    
    /* 禁用 JTAG，释放 PB4 引脚 */
    GPIO_PinRemapConfig(GPIO_Remap_SWJ_JTAGDisable, ENABLE);
    /* 部分重映射：TIM3_CH1 → PB4 */
    GPIO_PinRemapConfig(GPIO_PartialRemap_TIM3, ENABLE);

    /* 配置 GPIO */
    GPIO_InitStructure.GPIO_Pin = PWM_GPIO_PIN;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(PWM_GPIO_PORT, &GPIO_InitStructure);

    /* 配置定时器 */
    TIM_TimeBaseStructure.TIM_Period = g_pwm_period;
    TIM_TimeBaseStructure.TIM_Prescaler = 0;
    TIM_TimeBaseStructure.TIM_ClockDivision = 0;
    TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
    TIM_TimeBaseInit(PWM_TIM, &TIM_TimeBaseStructure);

    /* 配置 PWM 输出 */
    TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1;
    TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
    TIM_OCInitStructure.TIM_Pulse = 0;
    TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;
    TIM_OC1Init(PWM_TIM, &TIM_OCInitStructure);

    /* 使能预装载 */
    TIM_OC1PreloadConfig(PWM_TIM, TIM_OCPreload_Enable);
    TIM_ARRPreloadConfig(PWM_TIM, ENABLE);

    /* 启动定时器 */
    TIM_Cmd(PWM_TIM, ENABLE);
    
    /* 配置 TRGO（用于触发 ADC 转换） */
    TIM_SelectOutputTrigger(PWM_TIM, TIM_TRGOSource_Update);
    
    /* 初始化占空比 */
    g_current_duty = 0;
    PWM_SetDuty(0);
}

/**
  * @brief  设置 PWM 占空比
  * @param  duty: 占空比，千分比 (0-1000)
  * @retval 无
  */
void PWM_SetDuty(uint16_t duty)
{
    /* 限制占空比范围 */
    if (duty > PWM_MAX_DUTY)
    {
        duty = PWM_MAX_DUTY;
    }
    if (duty < PWM_MIN_DUTY)
    {
        duty = PWM_MIN_DUTY;
    }

    if (duty == 0)
    {
        // 占空比为 0 时，禁用 PWM 输出通道
        TIM_CCxCmd(PWM_TIM, TIM_Channel_1, TIM_CCx_Disable);
    }
    else
    {
        // 占空比不为 0 时，启用 PWM 输出通道
        TIM_CCxCmd(PWM_TIM, TIM_Channel_1, TIM_CCx_Enable);
        
        /* 计算比较值 */
        uint16_t compare = (uint16_t)((uint32_t)g_pwm_period * duty / PWM_MAX_DUTY);

        /* 设置占空比 */
        TIM_SetCompare1(PWM_TIM, compare);
    }
    
    g_current_duty = duty;
}

/**
  * @brief  使能 PWM 输出
  * @retval 无
  */
void PWM_Enable(void)
{
    TIM_Cmd(PWM_TIM, ENABLE);
}

/**
  * @brief  禁用 PWM 输出
  * @retval 无
  */
void PWM_Disable(void)
{
    TIM_Cmd(PWM_TIM, DISABLE);
}

/**
  * @brief  获取当前 PWM 占空比
  * @retval 当前占空比，千分比 (0-1000)
  */
uint16_t PWM_GetDuty(void)
{
    return g_current_duty;
}
