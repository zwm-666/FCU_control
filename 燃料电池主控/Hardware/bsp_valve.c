#include "bsp_valve.h"

static uint8_t s_inlet_state = 0U;
static uint8_t s_purge_state = 0U;
static uint8_t s_mout3_state = 0U;

void Valve_Init(void)
{
    GPIO_InitTypeDef gpio_init;

    RCC_APB2PeriphClockCmd(VALVE_INLET_RCC | VALVE_HEATER_RCC, ENABLE);

    gpio_init.GPIO_Pin = VALVE_INLET_PIN | VALVE_PURGE_PIN;
    gpio_init.GPIO_Mode = GPIO_Mode_Out_PP;
    gpio_init.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(VALVE_INLET_PORT, &gpio_init);

    gpio_init.GPIO_Pin = VALVE_HEATER_PIN;
    GPIO_Init(VALVE_HEATER_PORT, &gpio_init);

    VALVE_OFF(VALVE_INLET_PORT, VALVE_INLET_PIN);
    VALVE_OFF(VALVE_PURGE_PORT, VALVE_PURGE_PIN);
    VALVE_OFF(VALVE_HEATER_PORT, VALVE_HEATER_PIN);

    s_inlet_state = 0U;
    s_purge_state = 0U;
    s_mout3_state = 0U;
}

void Valve_Control_Inlet(uint8_t state)
{
    s_inlet_state = state ? 1U : 0U;

    if (s_inlet_state != 0U) {
        VALVE_ON(VALVE_INLET_PORT, VALVE_INLET_PIN);
    } else {
        VALVE_OFF(VALVE_INLET_PORT, VALVE_INLET_PIN);
    }
}

void Valve_Control_Purge(uint8_t state)
{
    s_purge_state = state ? 1U : 0U;

    if (s_purge_state != 0U) {
        VALVE_ON(VALVE_PURGE_PORT, VALVE_PURGE_PIN);
    } else {
        VALVE_OFF(VALVE_PURGE_PORT, VALVE_PURGE_PIN);
    }
}

void Valve_Control_Mout3(uint8_t state)
{
    s_mout3_state = state ? 1U : 0U;

    if (s_mout3_state != 0U) {
        VALVE_ON(VALVE_HEATER_PORT, VALVE_HEATER_PIN);
    } else {
        VALVE_OFF(VALVE_HEATER_PORT, VALVE_HEATER_PIN);
    }
}

uint8_t Valve_Get_InletState(void)
{
    return s_inlet_state;
}

uint8_t Valve_Get_PurgeState(void)
{
    return s_purge_state;
}

uint8_t Valve_Get_Mout3State(void)
{
    return s_mout3_state;
}
