#include "bsp_fan.h"

volatile uint16_t Fan_Period_Capture = 0U;
volatile uint8_t Fan_Capture_Updated = 0U;

static uint8_t s_fan_power_on = 0U;
static uint8_t s_fan_duty = 0U;
static uint8_t s_rpm_stale_ticks = 0U;

#define FAN_RPM_TIMEOUT_TICKS 6U

void Fan_Init(void)
{
    GPIO_InitTypeDef gpio_init;
    TIM_TimeBaseInitTypeDef tim_base_init;
    TIM_OCInitTypeDef tim_oc_init;
    TIM_ICInitTypeDef tim_ic_init;
    NVIC_InitTypeDef nvic_init;

    RCC_APB2PeriphClockCmd(FAN_PWM_RCC_GPIO | FAN_CAP_RCC_GPIO | FAN_CAP_RCC_TIM | RCC_APB2Periph_AFIO, ENABLE);
    RCC_APB1PeriphClockCmd(FAN_PWM_RCC_TIM, ENABLE);

    gpio_init.GPIO_Pin = FAN_PWM_PIN;
    gpio_init.GPIO_Mode = GPIO_Mode_AF_PP;
    gpio_init.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(FAN_PWM_GPIO_PORT, &gpio_init);

    tim_base_init.TIM_Period = 100U - 1U;
    tim_base_init.TIM_Prescaler = 72U - 1U;
    tim_base_init.TIM_ClockDivision = 0U;
    tim_base_init.TIM_CounterMode = TIM_CounterMode_Up;
    TIM_TimeBaseInit(TIM4, &tim_base_init);

    tim_oc_init.TIM_OCMode = TIM_OCMode_PWM1;
    tim_oc_init.TIM_OutputState = TIM_OutputState_Enable;
    tim_oc_init.TIM_Pulse = 0U;
    tim_oc_init.TIM_OCPolarity = TIM_OCPolarity_High;
    TIM_OC4Init(TIM4, &tim_oc_init);
    TIM_OC4PreloadConfig(TIM4, TIM_OCPreload_Enable);
    TIM_Cmd(TIM4, ENABLE);

    gpio_init.GPIO_Pin = FAN_CAP_PIN;
    gpio_init.GPIO_Mode = GPIO_Mode_IPU;
    GPIO_Init(FAN_CAP_GPIO_PORT, &gpio_init);

    tim_base_init.TIM_Period = 0xFFFFU;
    tim_base_init.TIM_Prescaler = 72U - 1U;
    TIM_TimeBaseInit(TIM1, &tim_base_init);

    tim_ic_init.TIM_Channel = TIM_Channel_1;
    tim_ic_init.TIM_ICPolarity = TIM_ICPolarity_Falling;
    tim_ic_init.TIM_ICSelection = TIM_ICSelection_DirectTI;
    tim_ic_init.TIM_ICPrescaler = TIM_ICPSC_DIV1;
    tim_ic_init.TIM_ICFilter = 0x04U;
    TIM_ICInit(TIM1, &tim_ic_init);

    nvic_init.NVIC_IRQChannel = TIM1_CC_IRQn;
    nvic_init.NVIC_IRQChannelPreemptionPriority = 1U;
    nvic_init.NVIC_IRQChannelSubPriority = 1U;
    nvic_init.NVIC_IRQChannelCmd = ENABLE;
    NVIC_Init(&nvic_init);

    TIM_ITConfig(TIM1, TIM_IT_CC1, ENABLE);
    TIM_Cmd(TIM1, ENABLE);
    TIM_CtrlPWMOutputs(TIM1, ENABLE);

    Fan_Period_Capture = 0U;
    Fan_Capture_Updated = 0U;
    s_rpm_stale_ticks = 0U;
    Fan_PowerCtrl(0U);
    Fan_Set_DutyCycle(0U);
}

void Fan_PowerCtrl(uint8_t on)
{
    s_fan_power_on = on ? 1U : 0U;
    Valve_Control_Mout3(s_fan_power_on);

    if (s_fan_power_on != 0U) {
        TIM_SetCompare4(TIM4, s_fan_duty);
    } else {
        TIM_SetCompare4(TIM4, 0U);
    }
}

void Fan_Set_DutyCycle(uint8_t duty)
{
    if (duty > 100U) {
        duty = 100U;
    }

    s_fan_duty = duty;

    if (s_fan_power_on != 0U) {
        TIM_SetCompare4(TIM4, s_fan_duty);
    } else {
        TIM_SetCompare4(TIM4, 0U);
    }
}

uint16_t Fan_Get_RPM(void)
{
    if (Fan_Capture_Updated != 0U) {
        Fan_Capture_Updated = 0U;
        s_rpm_stale_ticks = 0U;
    } else {
        if (s_rpm_stale_ticks < 0xFFU) {
            ++s_rpm_stale_ticks;
        }
        if (s_rpm_stale_ticks > FAN_RPM_TIMEOUT_TICKS) {
            Fan_Period_Capture = 0U;
            return 0U;
        }
    }

    if (Fan_Period_Capture == 0U) {
        return 0U;
    }

    {
        uint32_t rpm = 30000000UL / Fan_Period_Capture;
        if (rpm > 20000UL) {
            rpm = 0UL;
        }
        return (uint16_t)rpm;
    }
}

uint8_t Fan_Get_PowerState(void)
{
    return s_fan_power_on;
}

uint8_t Fan_Get_DutyCycle(void)
{
    return s_fan_duty;
}

void TIM1_CC_IRQHandler(void)
{
    if (TIM_GetITStatus(TIM1, TIM_IT_CC1) != RESET) {
        TIM_ClearITPendingBit(TIM1, TIM_IT_CC1);
        Fan_Period_Capture = TIM_GetCapture1(TIM1);
        TIM_SetCounter(TIM1, 0U);
        Fan_Capture_Updated = 1U;
    }
}
