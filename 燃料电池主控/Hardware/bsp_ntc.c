#include "bsp_ntc.h"

/*
 * 模块名称：bsp_ntc
 * 模块功能：完成多通道模拟量采样，并将原始 ADC 值换算为温度、压力、电压和电流。
 * 说明：
 * 1. DMA 缓冲区布局固定为 [NTC1, NTC2, AD1, AD2, AD3, AD4, AD5, AD6]；
 * 2. 温度使用 NTC B 值公式换算；
 * 3. 压力、电压、电流均基于电压采样进一步推导得到。
 */

/* DMA 扫描缓冲区：0/1 为两路 NTC，2~7 为 AD1~AD6 */
__IO uint16_t NTC_ADC_Values[NTC_CH_NUM];

/**
  * @brief  初始化 NTC 相关 ADC1 与 DMA1
  * @retval 无
  */
void NTC_Init(void)
{
    DMA_InitTypeDef DMA_InitStructure;
    ADC_InitTypeDef ADC_InitStructure;
    GPIO_InitTypeDef GPIO_InitStructure;
    static const uint8_t adc_scan_list[NTC_CH_NUM] = {
        NTC_ADC_CH_1,
        NTC_ADC_CH_2,
        VSENSE_ADC_CH_0,
        VSENSE_ADC_CH_1,
        VSENSE_ADC_CH_2,
        VSENSE_ADC_CH_3,
        VSENSE_ADC_CH_4,
        VSENSE_ADC_CH_5
    };
    uint8_t rank;

    RCC_AHBPeriphClockCmd(RCC_AHBPeriph_DMA1, ENABLE);                                   // 使能 DMA1 时钟
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_ADC1 | RCC_APB2Periph_GPIOA | RCC_APB2Periph_GPIOB, ENABLE); 

    GPIO_InitStructure.GPIO_Pin = NTC1_GPIO_PIN;                                          // 配置 NTC1 输入引脚
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;                                      // 模拟输入模式
    GPIO_Init(NTC1_GPIO_PORT, &GPIO_InitStructure);                                       // 初始化 NTC1 对应引脚

    GPIO_InitStructure.GPIO_Pin = NTC2_GPIO_PIN;                                          // 配置 NTC2 输入引脚
                                        GPIO_Init(NTC2_GPIO_PORT, &GPIO_InitStructure);                                       // 初始化 NTC2 对应引脚

    GPIO_InitStructure.GPIO_Pin = VSENSE_PIN_MASK;                                        // 一次性配置 AD1~AD6 对应引脚
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;                                         // 全部设置为模拟输入
    GPIO_Init(VSENSE_GPIO_PORT, &GPIO_InitStructure);                                     // 初始化电压采样引脚组

    DMA_DeInit(DMA1_Channel1);                                                            // 复位 DMA1 通道 1
    DMA_InitStructure.DMA_PeripheralBaseAddr = (uint32_t)&(ADC1->DR);                     // 外设地址指向 ADC1 数据寄存器
    DMA_InitStructure.DMA_MemoryBaseAddr = (uint32_t)NTC_ADC_Values;                      // 内存地址指向 DMA 缓冲区首地址
    DMA_InitStructure.DMA_DIR = DMA_DIR_PeripheralSRC;                                    // 数据方向：外设到内存
    DMA_InitStructure.DMA_BufferSize = NTC_CH_NUM;                                        // 缓冲区长度等于扫描通道总数
    DMA_InitStructure.DMA_PeripheralInc = DMA_PeripheralInc_Disable;                      // 外设地址固定不变
    DMA_InitStructure.DMA_MemoryInc = DMA_MemoryInc_Enable;                               // 每次搬运后内存地址自增
    DMA_InitStructure.DMA_PeripheralDataSize = DMA_PeripheralDataSize_HalfWord;           // 外设数据宽度为 16 位
    DMA_InitStructure.DMA_MemoryDataSize = DMA_MemoryDataSize_HalfWord;                   // 内存数据宽度为 16 位
    DMA_InitStructure.DMA_Mode = DMA_Mode_Circular;                                       // 循环模式，持续刷新所有通道数据
    DMA_InitStructure.DMA_Priority = DMA_Priority_High;                                   // 配置较高优先级
    DMA_InitStructure.DMA_M2M = DMA_M2M_Disable;                                          // 禁止内存到内存模式
    DMA_Init(DMA1_Channel1, &DMA_InitStructure);                                          // 写入 DMA 配置

    DMA_Cmd(DMA1_Channel1, ENABLE);                                                       // 使能 DMA 通道

    ADC_InitStructure.ADC_Mode = ADC_Mode_Independent;                                    // ADC 独立模式
    ADC_InitStructure.ADC_ScanConvMode = ENABLE;                                          // 开启扫描模式，多通道轮流采样
    ADC_InitStructure.ADC_ContinuousConvMode = ENABLE;                                    // 开启连续转换模式
    ADC_InitStructure.ADC_ExternalTrigConv = ADC_ExternalTrigConv_None;                   // 不使用外部触发
    ADC_InitStructure.ADC_DataAlign = ADC_DataAlign_Right;                                // 采样结果右对齐
    ADC_InitStructure.ADC_NbrOfChannel = NTC_CH_NUM;                                      // 扫描总通道数为 8
    ADC_Init(ADC1, &ADC_InitStructure);                                                   // 写入 ADC 配置

    RCC_ADCCLKConfig(RCC_PCLK2_Div6);                                                     // 设置 ADC 时钟为 12MHz

    for (rank = 0; rank < NTC_CH_NUM; rank++) {
        ADC_RegularChannelConfig(ADC1, adc_scan_list[rank], (uint8_t)(rank + 1U), ADC_SampleTime_239Cycles5); // 按预设顺序依次配置每个扫描通道
    }

    ADC_DMACmd(ADC1, ENABLE);                                                             // 打开 ADC 的 DMA 请求

    ADC_Cmd(ADC1, ENABLE);                                                                // 使能 ADC1 外设

    ADC_ResetCalibration(ADC1);                                                           // 复位 ADC 校准状态
    while (ADC_GetResetCalibrationStatus(ADC1));                                          // 等待复位完成
    ADC_StartCalibration(ADC1);                                                           // 启动 ADC 校准
    while (ADC_GetCalibrationStatus(ADC1));                                               // 等待校准完成

    ADC_SoftwareStartConvCmd(ADC1, ENABLE);                                               // 软件触发开始连续扫描
}

/**
  * @brief  获取指定通道的温度值
  * @param  channel_index 0 表示 NTC1，1 表示 NTC2
  * @retval 温度值，单位：℃；异常时返回错误标记值
  */
float NTC_Get_Temperature(uint8_t channel_index)
{
    float adc_val = 0.0f;
    float ntc_r = 0.0f;
    float temp_k = 0.0f;

    if (channel_index >= 2U) return -999.0f;                                             // 仅允许访问两路 NTC 通道

    adc_val = (float)NTC_ADC_Values[channel_index];                                       // 从 DMA 缓冲区读取对应的原始 ADC 值

    if (adc_val < 50.0f || adc_val > 4050.0f) {
        return -99.0f;                                                                    // ADC 值过低或过高时，认为传感器开路/短路或异常
    }

    ntc_r = PULLUP_R * adc_val / (ADC_MAX_VAL - adc_val);                                 // 由分压公式反推当前 NTC 阻值

    temp_k = 1.0f / ((1.0f / (273.15f + NTC_NOMINAL_T)) + (log(ntc_r / NTC_NOMINAL_R) / NTC_B_VALUE)); // 使用 B 值公式求热力学温度

    return temp_k - 273.15f;                                                              // 将 Kelvin 温度转换为摄氏度
}

/**
  * @brief  将 ADC 原始值换算为压力值
  * @param  adc_val ADC 原始采样值
  * @retval 压力值，单位：MPa
  */
static float ADC_To_Pressure_MPa(uint16_t adc_val)
{
    float pin_voltage;
    float sensor_voltage;
    float pressure;
    float gain = (PRESS_RESISTOR_TOP_K + PRESS_RESISTOR_BOT_K) / PRESS_RESISTOR_BOT_K;

    pin_voltage = ((float)adc_val * ADC_REF_VOLT) / ADC_MAX_VAL;                         // 先将 ADC 值换算为 MCU 引脚电压
    sensor_voltage = pin_voltage * gain;                                                 // 根据分压比例还原传感器实际输出电压

    if (sensor_voltage < 0.1f) {
        return 0.0f;                                                                      // 电压极低时按 0 压力处理
    }

    pressure = (sensor_voltage - PRESS_SENSOR_MIN_VOLT) * (PRESS_SENSOR_RANGE_MPA / (PRESS_SENSOR_MAX_VOLT - PRESS_SENSOR_MIN_VOLT)); // 根据传感器量程线性换算压力

    if (pressure < 0.0f) {
        pressure = 0.0f;                                                                  // 低于量程下限时钳位为 0
    }
    if (pressure > PRESS_SENSOR_RANGE_MPA) {
        pressure = PRESS_SENSOR_RANGE_MPA;                                                // 超过量程上限时钳位为满量程
    }
    return pressure;                                                                      // 返回最终压力值
}

/**
  * @brief  获取入口压力
  * @retval 压力值，单位：MPa
  */
float NTC_Get_InletPressure_MPa(void)
{
    return ADC_To_Pressure_MPa(NTC_ADC_Values[ADC_IDX_VSENSE0]);                          // 使用 AD1 对应采样值换算入口压力
}

/**
  * @brief  获取瓶压
  * @retval 压力值，单位：MPa
  */
float NTC_Get_BottlePressure_MPa(void)
{
    return ADC_To_Pressure_MPa(NTC_ADC_Values[ADC_IDX_VSENSE1]);                          // 使用 AD2 对应采样值换算瓶压
}

/**
  * @brief  获取指定电压采样通道的实际电压
  * @param  vsense_index 电压采样索引
  * @retval 实际电压值，单位：V；索引非法时返回 -1.0f
  */
float NTC_Get_VSense_V(uint8_t vsense_index)
{
    static const float gain[VSENSE_CH_NUM] = {
        VSENSE_CH0_GAIN,
        VSENSE_CH1_GAIN,
        VSENSE_CH2_GAIN,
        VSENSE_CH3_GAIN,
        VSENSE_CH4_GAIN,
        VSENSE_CH5_GAIN
    };
    uint16_t raw;
    float pin_v;

    if (vsense_index >= VSENSE_CH_NUM) {
        return -1.0f;                                                                     // 索引越界时返回错误标记值
    }

    raw = NTC_ADC_Values[ADC_IDX_VSENSE0 + vsense_index];                                 // 根据索引取出对应的 ADC 原始值
    pin_v = ((float)raw * ADC_REF_VOLT) / ADC_MAX_VAL;                                    // 先计算 MCU 采样引脚电压
    return pin_v * gain[vsense_index];                                                    // 再按对应通道增益还原实际电压
}

/**
  * @brief  获取输入电流
  * @retval 输入电流，单位：A
  */
float NTC_Get_InputCurrent_A(void)
{
    float sensor_v = NTC_Get_VSense_V((uint8_t)CURR_IN_ADC_INDEX);                        // 获取输入电流采样通道的等效电压
    if (sensor_v < 0.0f) {
        return 0.0f;                                                                       // 索引或采样异常时返回 0
    }
    return (sensor_v - CURR_ZERO_V) / CURR_SENS_V_PER_A;                                  // 按零点电压和灵敏度换算输入电流
}

/**
  * @brief  获取输出电流
  * @retval 输出电流，单位：A
  */
float NTC_Get_OutputCurrent_A(void)
{
    float sensor_v = NTC_Get_VSense_V((uint8_t)CURR_OUT_ADC_INDEX);                       // 获取输出电流采样通道的等效电压
    if (sensor_v < 0.0f) {
        return 0.0f;                                                                       // 索引或采样异常时返回 0
    }
    return (sensor_v - CURR_ZERO_V) / CURR_SENS_V_PER_A;                                  // 按零点电压和灵敏度换算输出电流
}
