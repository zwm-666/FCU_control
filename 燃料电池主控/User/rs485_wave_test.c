#include "485.h"
#include "bsp_dcdc_boost.h"
#include "main.h"
#include "stm32f10x.h"

/*
 * RS485 waveform test firmware.
 *
 * Usage in Keil:
 * 1. Exclude User/main.c from build.
 * 2. Add this file to the User group.
 * 3. Build and download.
 *
 * Expected waveform:
 * - PA9 / SCITXD485: continuous 0x55 UART waveform.
 * - 485A/485B: corresponding differential waveform through U3.
 */

volatile uint32_t g_ms = 0U;
float g_input_voltage = 0.0f;
float g_output_voltage = 0.0f;
float g_input_current = 0.0f;
float g_output_current = 0.0f;
DCDC_Boost_Controller g_boost_ctrl;
uint16_t g_fault_code = 0U;
uint8_t g_fault_level = 0U;
SystemState_t g_state = SYS_OFF;

static void WaveTest_Delay(volatile uint32_t cycles)
{
    while (cycles > 0U) {
        --cycles;
    }
}

int main(void)
{
    const uint8_t pattern = 0x55U;

    SystemInit();
    SysTick_Config(SystemCoreClock / 1000U);
    RS485_Init();

    while (1) {
        RS485_Send_Data(&pattern, 1U);

        /*
         * Leave a small gap so the oscilloscope can trigger on repeated bytes
         * instead of seeing a fully continuous stream.
         */
        WaveTest_Delay(2000U);
    }
}
