#include "bsp_adc.h"
#include "delay.h"
#include "stm32f10x_adc.h"

#include <math.h>
#include <string.h>

#define FILTER_SIZE                     10U

#define ADC_REF_VOLTAGE                 3.3f
#define ADC_MAX_CODE                    4095.0f
#define ADC_REF_VOLTAGE_MV             3300L
#define ADC_12BIT_RESOLUTION           4095L
#define ADC_REF_DIV_4095_RECIP         ((ADC_REF_VOLTAGE_MV << 16) / ADC_12BIT_RESOLUTION)
#define ADC_TO_MV(adc_val)             (((int32_t)(adc_val) * ADC_REF_DIV_4095_RECIP) >> 16)

#define PRESSURE_DIVIDER_TOP_K          18.0f
#define PRESSURE_DIVIDER_BOTTOM_K       35.0f
#define PRESSURE_SENSOR_MIN_VOLT        0.5f
#define PRESSURE_SENSOR_MAX_VOLT        4.5f
#define INLET_PRESSURE_FULL_SCALE_MPA   2.55f
#define BOTTLE_PRESSURE_FULL_SCALE_MPA  35.0f

__IO uint16_t ADC_ConvertedValue[ADC_CHANNEL_COUNT];
__IO uint16_t g_adc_shadow[ADC_CHANNEL_COUNT];

static float g_filter_buffer[ADC_CHANNEL_COUNT][FILTER_SIZE];
static uint8_t g_filter_index[ADC_CHANNEL_COUNT];
static uint8_t g_filter_count[ADC_CHANNEL_COUNT];

static int32_t g_calib_iin_offset_ma = 0;
static int32_t g_calib_iout_offset_ma = 0;

static VICalibration_t g_vi_calib;

static void ADC_ResetFilters(void);
static void ADC_ConfigGPIO(void);
static void ADC_ConfigDMA(void);
static void ADC_ConfigNVIC(void);
static void ADC_ConfigScanSequence(void);
static void ADC_ApplyMode(uint8_t continuous_mode);
static uint16_t Get_ADC_BatchAverage(uint8_t channel_index);
static float MovingAverageFilter(uint8_t channel_index, float value);
static float ADC_ConvertPressure(uint8_t channel_index, float full_scale_mpa);
static float NTC_Calculate_Temperature(uint8_t channel_index);

static void ADC_ResetFilters(void)
{
    memset(g_filter_buffer, 0, sizeof(g_filter_buffer));
    memset(g_filter_index, 0, sizeof(g_filter_index));
    memset(g_filter_count, 0, sizeof(g_filter_count));
}

static void ADC_ConfigGPIO(void)
{
    GPIO_InitTypeDef gpio_init;

    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA | RCC_APB2Periph_GPIOB, ENABLE);

    gpio_init.GPIO_Pin = GPIO_Pin_2 | GPIO_Pin_3 | GPIO_Pin_4 |
                         GPIO_Pin_5 | GPIO_Pin_6 | GPIO_Pin_7;
    gpio_init.GPIO_Mode = GPIO_Mode_AIN;
    GPIO_Init(GPIOA, &gpio_init);

    gpio_init.GPIO_Pin = GPIO_Pin_0 | GPIO_Pin_1;
    gpio_init.GPIO_Mode = GPIO_Mode_AIN;
    GPIO_Init(GPIOB, &gpio_init);
}

static void ADC_ConfigDMA(void)
{
    DMA_InitTypeDef dma_init;

    DMA_Cmd(DMA1_Channel1, DISABLE);
    DMA_DeInit(DMA1_Channel1);

    dma_init.DMA_PeripheralBaseAddr = (uint32_t)&(ADC1->DR);
    dma_init.DMA_MemoryBaseAddr = (uint32_t)ADC_ConvertedValue;
    dma_init.DMA_DIR = DMA_DIR_PeripheralSRC;
    dma_init.DMA_BufferSize = ADC_CHANNEL_COUNT;
    dma_init.DMA_PeripheralInc = DMA_PeripheralInc_Disable;
    dma_init.DMA_MemoryInc = DMA_MemoryInc_Enable;
    dma_init.DMA_PeripheralDataSize = DMA_PeripheralDataSize_HalfWord;
    dma_init.DMA_MemoryDataSize = DMA_MemoryDataSize_HalfWord;
    dma_init.DMA_Mode = DMA_Mode_Circular;
    dma_init.DMA_Priority = DMA_Priority_High;
    dma_init.DMA_M2M = DMA_M2M_Disable;
    DMA_Init(DMA1_Channel1, &dma_init);

    DMA_ClearITPendingBit(DMA1_IT_TC1);
    DMA_ITConfig(DMA1_Channel1, DMA_IT_TC, ENABLE);
    DMA_Cmd(DMA1_Channel1, ENABLE);
}

static void ADC_ConfigNVIC(void)
{
    NVIC_InitTypeDef nvic_init;

    nvic_init.NVIC_IRQChannel = DMA1_Channel1_IRQn;
    nvic_init.NVIC_IRQChannelPreemptionPriority = 1U;
    nvic_init.NVIC_IRQChannelSubPriority = 0U;
    nvic_init.NVIC_IRQChannelCmd = ENABLE;
    NVIC_Init(&nvic_init);
}

static void ADC_ConfigScanSequence(void)
{
    ADC_RegularChannelConfig(ADC1, ADC_CHANNEL_PA2, 1U, ADC_SampleTime_7Cycles5);
    ADC_RegularChannelConfig(ADC1, ADC_CHANNEL_PA3, 2U, ADC_SampleTime_7Cycles5);
    ADC_RegularChannelConfig(ADC1, ADC_CHANNEL_PA4, 3U, ADC_SampleTime_7Cycles5);
    ADC_RegularChannelConfig(ADC1, ADC_CHANNEL_PA5, 4U, ADC_SampleTime_7Cycles5);
    ADC_RegularChannelConfig(ADC1, ADC_CHANNEL_PA6, 5U, ADC_SampleTime_7Cycles5);
    ADC_RegularChannelConfig(ADC1, ADC_CHANNEL_PA7, 6U, ADC_SampleTime_7Cycles5);
    ADC_RegularChannelConfig(ADC1, ADC_CHANNEL_PB0, 7U, ADC_SampleTime_239Cycles5);
    ADC_RegularChannelConfig(ADC1, ADC_CHANNEL_PB1, 8U, ADC_SampleTime_239Cycles5);
}

static void ADC_ApplyMode(uint8_t continuous_mode)
{
    ADC_InitTypeDef adc_init;

    ADC_SoftwareStartConvCmd(ADC1, DISABLE);
    ADC_ExternalTrigConvCmd(ADC1, DISABLE);
    ADC_DMACmd(ADC1, DISABLE);
    ADC_Cmd(ADC1, DISABLE);

    ADC_ConfigDMA();

    adc_init.ADC_Mode = ADC_Mode_Independent;
    adc_init.ADC_ScanConvMode = ENABLE;
    adc_init.ADC_ContinuousConvMode = (continuous_mode != 0U) ? ENABLE : DISABLE;
    adc_init.ADC_ExternalTrigConv = (continuous_mode != 0U) ?
        ADC_ExternalTrigConv_None : ADC_ExternalTrigConv_T3_TRGO;
    adc_init.ADC_DataAlign = ADC_DataAlign_Right;
    adc_init.ADC_NbrOfChannel = ADC_CHANNEL_COUNT;
    ADC_Init(ADC1, &adc_init);

    ADC_ConfigScanSequence();
    ADC_DMACmd(ADC1, ENABLE);
    ADC_Cmd(ADC1, ENABLE);

    if (continuous_mode != 0U) {
        ADC_ExternalTrigConvCmd(ADC1, DISABLE);
        ADC_SoftwareStartConvCmd(ADC1, ENABLE);
    } else {
        ADC_ExternalTrigConvCmd(ADC1, ENABLE);
    }
}

void ADC1_Mode_Config(void)
{
    RCC_AHBPeriphClockCmd(RCC_AHBPeriph_DMA1, ENABLE);
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_ADC1, ENABLE);
    RCC_ADCCLKConfig(RCC_PCLK2_Div6);

    ADC_ResetFilters();
    ADC_ConfigGPIO();
    ADC_ConfigNVIC();
    ADC_ApplyMode(0U);

    ADC_ResetCalibration(ADC1);
    while (ADC_GetResetCalibrationStatus(ADC1) != RESET) {
    }

    ADC_StartCalibration(ADC1);
    while (ADC_GetCalibrationStatus(ADC1) != RESET) {
    }
}

void DMA1_Channel1_IRQHandler(void)
{
    uint8_t i;

    if (DMA_GetITStatus(DMA1_IT_TC1) != RESET) {
        for (i = 0U; i < ADC_CHANNEL_COUNT; ++i) {
            g_adc_shadow[i] = ADC_ConvertedValue[i];
        }
        DMA_ClearITPendingBit(DMA1_IT_TC1);
    }
}

void ADC_StartContinuousMode(void)
{
    ADC_ApplyMode(1U);
}

void ADC_StartTriggeredMode(void)
{
    ADC_ApplyMode(0U);
}

uint16_t Get_ADC_RawValue(uint8_t channel_index)
{
    if (channel_index < ADC_CHANNEL_COUNT) {
        return g_adc_shadow[channel_index];
    }

    return 0U;
}

static uint16_t Get_ADC_BatchAverage(uint8_t channel_index)
{
    return Get_ADC_RawValue(channel_index);
}

static float MovingAverageFilter(uint8_t channel_index, float value)
{
    float sum;
    uint8_t i;
    uint8_t sample_count;

    if (channel_index >= ADC_CHANNEL_COUNT) {
        return value;
    }

    g_filter_buffer[channel_index][g_filter_index[channel_index]] = value;
    g_filter_index[channel_index] = (uint8_t)((g_filter_index[channel_index] + 1U) % FILTER_SIZE);

    if (g_filter_count[channel_index] < FILTER_SIZE) {
        ++g_filter_count[channel_index];
    }

    sample_count = g_filter_count[channel_index];
    if (sample_count == 0U) {
        return value;
    }

    sum = 0.0f;
    for (i = 0U; i < sample_count; ++i) {
        sum += g_filter_buffer[channel_index][i];
    }

    return sum / (float)sample_count;
}

float Get_ADC_Voltage(uint8_t channel_index)
{
    uint16_t adc_val;

    if (channel_index >= ADC_CHANNEL_COUNT) {
        return 0.0f;
    }

    adc_val = Get_ADC_BatchAverage(channel_index);
    return ((float)adc_val * ADC_REF_VOLTAGE) / ADC_MAX_CODE;
}

static float ADC_ConvertPressure(uint8_t channel_index, float full_scale_mpa)
{
    float pin_voltage;
    float sensor_voltage;
    float pressure;
    float gain;

    gain = (PRESSURE_DIVIDER_TOP_K + PRESSURE_DIVIDER_BOTTOM_K) / PRESSURE_DIVIDER_BOTTOM_K;
    pin_voltage = Get_ADC_Voltage(channel_index);
    sensor_voltage = pin_voltage * gain;

    if (sensor_voltage < 0.1f) {
        return 0.0f;
    }

    pressure = (sensor_voltage - PRESSURE_SENSOR_MIN_VOLT) *
               (full_scale_mpa / (PRESSURE_SENSOR_MAX_VOLT - PRESSURE_SENSOR_MIN_VOLT));

    if (pressure < 0.0f) {
        pressure = 0.0f;
    }
    if (pressure > full_scale_mpa) {
        pressure = full_scale_mpa;
    }

    return pressure;
}

float Get_Inlet_Pressure_Value(void)
{
    return ADC_ConvertPressure(ADC_INDEX_INLET_PRESSURE, INLET_PRESSURE_FULL_SCALE_MPA);
}

float Get_Bottle_Pressure_Value(void)
{
    return ADC_ConvertPressure(ADC_INDEX_BOTTLE_PRESSURE, BOTTLE_PRESSURE_FULL_SCALE_MPA);
}

float Get_Pressure_Value(void)
{
    return Get_Bottle_Pressure_Value();
}

float Get_ADC3_OutputVoltage(void)
{
    float voltage;

    voltage = Get_ADC_Voltage(ADC_INDEX_OUTPUT_VOLT) * 19.0f;
    return MovingAverageFilter(ADC_INDEX_OUTPUT_VOLT, voltage);
}

float Get_ADC4_InputVoltage(void)
{
    float voltage;

    voltage = Get_ADC_Voltage(ADC_INDEX_INPUT_VOLT) * 4.9f;
    return MovingAverageFilter(ADC_INDEX_INPUT_VOLT, voltage);
}

float Get_ADC5_OutputCurrent(void)
{
    float current;

    current = Get_ADC_Voltage(ADC_INDEX_OUTPUT_CURR) / 0.25f;
    return MovingAverageFilter(ADC_INDEX_OUTPUT_CURR, current);
}

float Get_ADC6_InputCurrent(void)
{
    float current;

    current = Get_ADC_Voltage(ADC_INDEX_INPUT_CURR) / 0.1f;
    return MovingAverageFilter(ADC_INDEX_INPUT_CURR, current);
}

void VICalibration_Start(void)
{
    memset(&g_vi_calib, 0, sizeof(g_vi_calib));

    g_calib_iin_offset_ma = 0;
    g_calib_iout_offset_ma = 0;

    ADC_ResetFilters();
    Delay_ms(10U);
}

uint8_t VICalibration_Update(void)
{
    uint32_t i;
    float input_voltage;

    if (g_vi_calib.is_calibrated != 0U) {
        return 1U;
    }

    for (i = 0U; i < CALIBRATION_SAMPLE_COUNT; ++i) {
        input_voltage = Get_ADC4_InputVoltage();

        if (input_voltage > 0.3f) {
            g_vi_calib.is_calibrated = 1U;
            return 1U;
        }

        g_vi_calib.calibration_sum_in_curr += Get_ADC6_InputCurrent();
        g_vi_calib.calibration_sum_out_curr += Get_ADC5_OutputCurrent();
        ++g_vi_calib.calibration_count;

        Delay_us(10U);
    }

    g_vi_calib.input_current_offset =
        g_vi_calib.calibration_sum_in_curr / (float)CALIBRATION_SAMPLE_COUNT;
    g_vi_calib.output_current_offset =
        g_vi_calib.calibration_sum_out_curr / (float)CALIBRATION_SAMPLE_COUNT;

    g_calib_iin_offset_ma = (int32_t)(g_vi_calib.input_current_offset * 1000.0f);
    g_calib_iout_offset_ma = (int32_t)(g_vi_calib.output_current_offset * 1000.0f);

    g_vi_calib.is_calibrated = 1U;
    return 1U;
}

float Get_CalibratedInputVoltage(void)
{
    return Get_ADC4_InputVoltage();
}

float Get_CalibratedOutputVoltage(void)
{
    return Get_ADC3_OutputVoltage();
}

float Get_CalibratedInputCurrent(void)
{
    float raw_current;
    float calibrated_current;

    raw_current = Get_ADC6_InputCurrent();
    if (g_vi_calib.is_calibrated == 0U) {
        return raw_current;
    }

    calibrated_current = raw_current - g_vi_calib.input_current_offset;
    if (calibrated_current < 0.0001f) {
        calibrated_current = 0.0f;
    }

    return calibrated_current;
}

float Get_CalibratedOutputCurrent(void)
{
    float raw_current;
    float calibrated_current;

    raw_current = Get_ADC5_OutputCurrent();
    if (g_vi_calib.is_calibrated == 0U) {
        return raw_current;
    }

    calibrated_current = raw_current - g_vi_calib.output_current_offset;
    if (calibrated_current < 0.0001f) {
        calibrated_current = 0.0f;
    }

    return calibrated_current;
}

int32_t Get_CalibratedInputVoltage_Int(void)
{
    uint16_t adc_val;
    int32_t voltage_mv;

    adc_val = Get_ADC_BatchAverage(ADC_INDEX_INPUT_VOLT);
    voltage_mv = ADC_TO_MV(adc_val);
    return (voltage_mv * 49L) / 10L;
}

int32_t Get_CalibratedOutputVoltage_Int(void)
{
    uint16_t adc_val;
    int32_t voltage_mv;

    adc_val = Get_ADC_BatchAverage(ADC_INDEX_OUTPUT_VOLT);
    voltage_mv = ADC_TO_MV(adc_val);
    return voltage_mv * 19L;
}

int32_t Get_CalibratedInputCurrent_Int(void)
{
    uint16_t adc_val;
    int32_t voltage_mv;
    int32_t current_ma;
    int32_t calibrated_current_ma;

    adc_val = Get_ADC_BatchAverage(ADC_INDEX_INPUT_CURR);
    voltage_mv = ADC_TO_MV(adc_val);
    current_ma = voltage_mv * 10L;
    calibrated_current_ma = current_ma - g_calib_iin_offset_ma;

    if (calibrated_current_ma < 1L) {
        calibrated_current_ma = 0L;
    }

    return calibrated_current_ma;
}

int32_t Get_CalibratedOutputCurrent_Int(void)
{
    uint16_t adc_val;
    int32_t voltage_mv;
    int32_t current_ma;
    int32_t calibrated_current_ma;

    adc_val = Get_ADC_BatchAverage(ADC_INDEX_OUTPUT_CURR);
    voltage_mv = ADC_TO_MV(adc_val);
    current_ma = voltage_mv * 4L;
    calibrated_current_ma = current_ma - g_calib_iout_offset_ma;

    if (calibrated_current_ma < 1L) {
        calibrated_current_ma = 0L;
    }

    return calibrated_current_ma;
}

uint8_t VICalibration_IsCalibrated(void)
{
    return g_vi_calib.is_calibrated;
}

static float NTC_Calculate_Temperature(uint8_t channel_index)
{
    uint16_t adc_val;
    float pin_voltage;
    float ntc_resistance;
    float temp_kelvin;
    float temperature;

    adc_val = Get_ADC_BatchAverage(channel_index);
    if ((adc_val < 50U) || (adc_val > 4050U)) {
        return -999.0f;
    }

    pin_voltage = ((float)adc_val * ADC_REF_VOLTAGE) / ADC_MAX_CODE;
    if ((pin_voltage < 0.01f) || (pin_voltage > (ADC_REF_VOLTAGE - 0.01f))) {
        return -999.0f;
    }

    ntc_resistance = NTC_PULLUP_R * pin_voltage / (ADC_REF_VOLTAGE - pin_voltage);
    if (ntc_resistance <= 0.0f) {
        return -999.0f;
    }

    temp_kelvin = 1.0f / ((1.0f / (NTC_NOMINAL_T + 273.15f)) +
                  (log(ntc_resistance / NTC_NOMINAL_R) / NTC_B_VALUE));
    temperature = temp_kelvin - 273.15f;

    if ((temperature < -50.0f) || (temperature > 150.0f)) {
        return -999.0f;
    }

    return temperature;
}

float Get_NTC1_Temperature(void)
{
    return NTC_Calculate_Temperature(ADC_INDEX_NTC1);
}

float Get_NTC2_Temperature(void)
{
    return NTC_Calculate_Temperature(ADC_INDEX_NTC2);
}
