#ifndef __BSP_ADC_H
#define __BSP_ADC_H

#include "stm32f10x.h"
#include <stdint.h>

#define ADC_CHANNEL_PA2             ADC_Channel_2
#define ADC_CHANNEL_PA3             ADC_Channel_3
#define ADC_CHANNEL_PA4             ADC_Channel_4
#define ADC_CHANNEL_PA5             ADC_Channel_5
#define ADC_CHANNEL_PA6             ADC_Channel_6
#define ADC_CHANNEL_PA7             ADC_Channel_7
#define ADC_CHANNEL_PB0             ADC_Channel_8
#define ADC_CHANNEL_PB1             ADC_Channel_9

#define ADC_INDEX_INLET_PRESSURE    0U
#define ADC_INDEX_BOTTLE_PRESSURE   1U
#define ADC_INDEX_PRESSURE          ADC_INDEX_BOTTLE_PRESSURE
#define ADC_INDEX_OUTPUT_VOLT       2U
#define ADC_INDEX_INPUT_VOLT        3U
#define ADC_INDEX_OUTPUT_CURR       4U
#define ADC_INDEX_INPUT_CURR        5U
#define ADC_INDEX_NTC1              6U
#define ADC_INDEX_NTC2              7U

#define ADC_CHANNEL_COUNT           8U

#define NTC_NOMINAL_R               10000.0f
#define NTC_NOMINAL_T               25.0f
#define NTC_B_VALUE                 3435.0f
#define NTC_PULLUP_R                10000.0f

#define CALIBRATION_SAMPLE_COUNT    100U

typedef struct {
    float input_current_offset;
    float output_current_offset;
    uint8_t is_calibrated;
    uint32_t calibration_count;
    float calibration_sum_in_curr;
    float calibration_sum_out_curr;
} VICalibration_t;

extern __IO uint16_t ADC_ConvertedValue[ADC_CHANNEL_COUNT];
extern __IO uint16_t g_adc_shadow[ADC_CHANNEL_COUNT];

void ADC1_Mode_Config(void);
void ADC_StartContinuousMode(void);
void ADC_StartTriggeredMode(void);

void VICalibration_Start(void);
uint8_t VICalibration_Update(void);
uint8_t VICalibration_IsCalibrated(void);

uint16_t Get_ADC_RawValue(uint8_t channel_index);
float Get_ADC_Voltage(uint8_t channel_index);

float Get_Pressure_Value(void);
float Get_Inlet_Pressure_Value(void);
float Get_Bottle_Pressure_Value(void);

float Get_ADC3_OutputVoltage(void);
float Get_ADC4_InputVoltage(void);
float Get_ADC5_OutputCurrent(void);
float Get_ADC6_InputCurrent(void);

float Get_CalibratedInputVoltage(void);
int32_t Get_CalibratedInputVoltage_Int(void);
float Get_CalibratedOutputVoltage(void);
int32_t Get_CalibratedOutputVoltage_Int(void);
float Get_CalibratedInputCurrent(void);
int32_t Get_CalibratedInputCurrent_Int(void);
float Get_CalibratedOutputCurrent(void);
int32_t Get_CalibratedOutputCurrent_Int(void);

float Get_NTC1_Temperature(void);
float Get_NTC2_Temperature(void);

#endif
