#include "485.h"
#include "bsp_adc.h"
#include "bsp_dcdc_boost.h"
#include "bsp_fan.h"
#include "bsp_pwm.h"
#include "bsp_valve.h"
#include "delay.h"
#include "main.h"

#include <stdint.h>
#include <string.h>

#define FAULT_RECOVERY_DELAY_MS         1000U
#define FAULT_RECOVERY_MAX_RETRIES      3U
#define SERIAL_STATUS_TX_STEP_MS        25U

#define SERIAL_CTRL_MODE_MANUAL         0x00U
#define SERIAL_CTRL_MODE_AUTO           0x01U

#define SERIAL_CTRL_CMD_STOP            0x00U
#define SERIAL_CTRL_CMD_START           0x01U
#define SERIAL_CTRL_CMD_RESET           0x02U
#define SERIAL_CTRL_CMD_ESTOP           0x03U
#define SERIAL_CTRL_CMD_SHUTDOWN        0x04U

#define SERIAL_CTRL_FLAG_INLET          (1U << 0)
#define SERIAL_CTRL_FLAG_PURGE          (1U << 1)
#define SERIAL_CTRL_FLAG_FAN_POWER      (1U << 2)
#define SERIAL_CTRL_FLAG_FAN_RUN        (1U << 3)

static void SystemInitLayer(void);
static void HardwareInitLayer(void);
static uint8_t CalibrationLayer(void);
static void InterruptInitLayer(void);
static uint8_t SoftStartLayer(float *input_voltage, float *output_voltage,
                              float *input_current, float *output_current,
                              uint16_t *duty, uint8_t reset);
static void MainScheduler(void);
static void LoadDefaultSerialControlPayload(void);
static void TransmitSerialBootProbe(void);
static void PollSerialControlPayload(void);
static void ApplySerialControl(void);
static void TransmitSerialStatusFrames(float input_voltage, float output_voltage,
                                       float input_current, float output_current,
                                       float stack_temp_c, float ambient_temp_c,
                                       float bottle_pressure_mpa, float inlet_pressure_mpa);
static void ForcePowerStageStop(void);
static void ApplyActuatorOutputs(uint8_t flags, uint8_t fan_pwm);
static uint16_t EncodeScaledU16(float value, float scale);
static uint16_t EncodeTempRaw(float temp_c);
static uint16_t EncodePressureRaw(float pressure_mpa);
static uint16_t EncodeFaultCode(void);
static uint8_t MapSystemStatus(SystemState_t state);
static uint8_t MapFaultLevel(DCDC_Boost_Fault fault, SystemState_t state);
static void PackU16LE(uint8_t *dst, uint16_t value);
static void PackU16BE(uint8_t *dst, uint16_t value);

volatile uint32_t g_ms = 0U;
static volatile uint8_t g_systick_ready = 0U;

float g_input_voltage = 0.0f;
float g_output_voltage = 0.0f;
float g_input_current = 0.0f;
float g_output_current = 0.0f;
float g_ntc1_temperature = 0.0f;
float g_ntc2_temperature = 0.0f;

SystemState_t g_system_state = SYS_OFF;
uint8_t g_calibration_status = 0U;
DCDC_Boost_Fault g_last_fault = DCDC_BOOST_FAULT_NONE;
uint32_t g_state_timestamp = 0U;
uint32_t g_state_duration = 0U;
uint8_t g_soft_start_progress = 0U;
uint16_t g_fault_code = 0U;
uint8_t g_fault_level = 0U;
SystemState_t g_state = SYS_OFF;

static uint8_t g_tim3_initialized = 0U;
static uint8_t g_runtime_ready = 0U;
static uint8_t g_run_request = 0U;
static uint8_t g_emergency_stop = 0U;
static uint8_t g_control_mode = SERIAL_CTRL_MODE_MANUAL;
static uint8_t g_control_flags = 0U;
static uint8_t g_fan_target_pwm = 0U;
static uint8_t g_serial_control_payload[RS485_SERIAL_DATA_LEN];

DCDC_Boost_Controller g_boost_ctrl;

static void PackU16LE(uint8_t *dst, uint16_t value)
{
    dst[0] = (uint8_t)(value & 0x00FFU);
    dst[1] = (uint8_t)(value >> 8U);
}

static void PackU16BE(uint8_t *dst, uint16_t value)
{
    dst[0] = (uint8_t)(value >> 8U);
    dst[1] = (uint8_t)(value & 0x00FFU);
}

static uint16_t EncodeScaledU16(float value, float scale)
{
    float scaled;

    if ((scale <= 0.0f) || (value <= 0.0f)) {
        return 0U;
    }

    scaled = (value / scale) + 0.5f;
    if (scaled >= 65535.0f) {
        return 65535U;
    }

    return (uint16_t)scaled;
}

static uint16_t EncodeTempRaw(float temp_c)
{
    if (temp_c < -40.0f) {
        temp_c = -40.0f;
    }

    if (temp_c > 150.0f) {
        temp_c = 150.0f;
    }

    return (uint16_t)(((temp_c + 40.0f) * 10.0f) + 0.5f);
}

static uint16_t EncodePressureRaw(float pressure_mpa)
{
    if (pressure_mpa <= 0.0f) {
        return 0U;
    }

    if (pressure_mpa > 655.35f) {
        pressure_mpa = 655.35f;
    }

    return (uint16_t)((pressure_mpa * 100.0f) + 0.5f);
}

static uint16_t EncodeFaultCode(void)
{
    if (g_boost_ctrl.fault != DCDC_BOOST_FAULT_NONE) {
        return (uint16_t)g_boost_ctrl.fault;
    }

    return (uint16_t)g_last_fault;
}

static uint8_t MapSystemStatus(SystemState_t state)
{
    switch (state) {
    case SYS_OFF:
        return 0U;

    case SYS_SOFT_START:
    case SYS_RUNNING:
        return 2U;

    case SYS_FAULT:
    case SYS_POWER_LOSS:
        return 3U;

    default:
        return 1U;
    }
}

static uint8_t MapFaultLevel(DCDC_Boost_Fault fault, SystemState_t state)
{
    if ((g_emergency_stop != 0U) || (state == SYS_FAULT) || (state == SYS_POWER_LOSS) || (fault != DCDC_BOOST_FAULT_NONE)) {
        return 3U;
    }

    return 0U;
}

static void ApplyActuatorOutputs(uint8_t flags, uint8_t fan_pwm)
{
    Valve_Control_Inlet((flags & SERIAL_CTRL_FLAG_INLET) != 0U);
    Valve_Control_Purge((flags & SERIAL_CTRL_FLAG_PURGE) != 0U);
    Fan_PowerCtrl((flags & SERIAL_CTRL_FLAG_FAN_POWER) != 0U);

    if ((flags & SERIAL_CTRL_FLAG_FAN_RUN) != 0U) {
        Fan_Set_DutyCycle(fan_pwm);
    } else {
        Fan_Set_DutyCycle(0U);
    }
}

static void LoadDefaultSerialControlPayload(void)
{
    uint8_t payload[8] = {0U};
    uint16_t target_voltage = (uint16_t)((DCF_OUT_VOLT_TARGET_DEF_V * 10.0f) + 0.5f);
    uint16_t target_current = (uint16_t)((DCF_IN_CURRENT_REF_DEF_A * 10.0f) + 0.5f);

    PackU16LE(&payload[3], target_voltage);
    PackU16LE(&payload[5], target_current);
    memcpy(g_serial_control_payload, payload, sizeof(g_serial_control_payload));
}

static void TransmitSerialBootProbe(void)
{
    uint8_t count;
    uint8_t payload[8] = {
        0xA5U, 0x00U, 0x42U, 0x4FU, 0x4FU, 0x54U, 0x00U, 0x00U
    };

    for (count = 0U; count < 3U; ++count) {
        payload[0] = (uint8_t)(0xA5U + count);
        RS485_Send_Frame(CAN_ID_FCU1, payload);
        Delay_ms(20U);
    }
}

static void PollSerialControlPayload(void)
{
    uint32_t frame_id;
    uint8_t payload[RS485_SERIAL_DATA_LEN];

    if (RS485_Poll_Frame(&frame_id, payload) == 0U) {
        return;
    }

    if (frame_id == CAN_ID_HOST_CMD) {
        memcpy(g_serial_control_payload, payload, sizeof(g_serial_control_payload));
    }
}

static void ApplySerialControl(void)
{
    uint8_t payload[RS485_SERIAL_DATA_LEN];
    uint8_t command;
    float target_voltage;
    float target_current;

    if (g_runtime_ready == 0U) {
        return;
    }

    memcpy(payload, g_serial_control_payload, sizeof(payload));

    g_control_mode = (uint8_t)(payload[0] & 0x03U);
    command = (uint8_t)((payload[0] >> 2U) & 0x07U);
    g_control_flags = payload[1];
    g_fan_target_pwm = payload[2];

    if (g_fan_target_pwm > 100U) {
        g_fan_target_pwm = 100U;
    }

    target_voltage = (float)((uint16_t)payload[3] | ((uint16_t)payload[4] << 8U)) / 10.0f;
    target_current = (float)((uint16_t)payload[5] | ((uint16_t)payload[6] << 8U)) / 10.0f;

    DCDC_Boost_SetTargetVoltage(&g_boost_ctrl, target_voltage);
    DCDC_Boost_SetTargetCurrent(&g_boost_ctrl, target_current);

    switch (command) {
    case SERIAL_CTRL_CMD_START:
        g_run_request = 1U;
        g_emergency_stop = 0U;
        if ((g_control_mode == SERIAL_CTRL_MODE_MANUAL) || (g_control_mode == SERIAL_CTRL_MODE_AUTO)) {
            ApplyActuatorOutputs(g_control_flags, g_fan_target_pwm);
        }
        break;

    case SERIAL_CTRL_CMD_RESET:
        g_emergency_stop = 0U;
        DCDC_Boost_ClearFault(&g_boost_ctrl);
        PID_Reset(&g_boost_ctrl.voltage_pid);
        PID_Reset(&g_boost_ctrl.current_pid);
        PID_Reset_Int(&g_boost_ctrl.voltage_pid_int);
        PID_Reset_Int(&g_boost_ctrl.current_pid_int);
        if ((g_system_state == SYS_FAULT) || (g_system_state == SYS_POWER_LOSS)) {
            g_system_state = SYS_WAIT_POWER;
        }
        if ((g_control_mode == SERIAL_CTRL_MODE_MANUAL) || (g_control_mode == SERIAL_CTRL_MODE_AUTO)) {
            ApplyActuatorOutputs(g_control_flags, g_fan_target_pwm);
        }
        break;

    case SERIAL_CTRL_CMD_ESTOP:
        g_run_request = 0U;
        g_emergency_stop = 1U;
        ApplyActuatorOutputs(0U, 0U);
        if ((g_system_state != SYS_OFF) && (g_system_state != SYS_INIT) && (g_system_state != SYS_CALIBRATION)) {
            g_system_state = SYS_SHUTDOWN;
        }
        break;

    case SERIAL_CTRL_CMD_SHUTDOWN:
    case SERIAL_CTRL_CMD_STOP:
    default:
        g_run_request = 0U;
        if (command != SERIAL_CTRL_CMD_ESTOP) {
            g_emergency_stop = 0U;
        }
        ApplyActuatorOutputs(0U, 0U);
        if ((g_system_state != SYS_OFF) && (g_system_state != SYS_INIT) && (g_system_state != SYS_CALIBRATION)) {
            g_system_state = SYS_SHUTDOWN;
        }
        break;
    }
}

static void TransmitSerialStatusFrames(float input_voltage, float output_voltage,
                                       float input_current, float output_current,
                                       float stack_temp_c, float ambient_temp_c,
                                       float bottle_pressure_mpa, float inlet_pressure_mpa)
{
    static uint32_t last_tx_ms = 0U;
    static uint8_t next_frame = 0U;
    static uint8_t heartbeat = 0U;
    uint8_t payload[8];
    uint8_t io_flags = 0U;

    if (g_runtime_ready == 0U) {
        return;
    }

    if ((uint32_t)(g_ms - last_tx_ms) < SERIAL_STATUS_TX_STEP_MS) {
        return;
    }

    last_tx_ms = g_ms;
    g_fault_code = EncodeFaultCode();
    g_fault_level = MapFaultLevel(g_boost_ctrl.fault, g_system_state);
    g_state = g_system_state;

    switch (next_frame) {
    case 0U:
        ++heartbeat;
        payload[0] = heartbeat;
        payload[1] = (uint8_t)((MapSystemStatus(g_system_state) & 0x03U) |
                               ((g_fault_level & 0x03U) << 2U));
        payload[2] = 0U;
        payload[3] = 0U;
        payload[4] = 0U;
        payload[5] = 0U;
        payload[6] = 0U;
        payload[7] = 0U;
        RS485_Send_Frame(CAN_ID_FCU1, payload);
        break;

    case 1U:
        PackU16LE(&payload[0], EncodeScaledU16(input_voltage, 0.01f));
        PackU16LE(&payload[2], EncodeScaledU16(input_current, 0.1f));
        PackU16LE(&payload[4], EncodeScaledU16(output_voltage, 0.01f));
        PackU16LE(&payload[6], EncodeScaledU16(output_current, 0.1f));
        RS485_Send_Frame(CAN_ID_FCU2, payload);
        break;

    case 2U:
        PackU16BE(&payload[0], EncodeTempRaw(stack_temp_c));
        PackU16BE(&payload[2], EncodePressureRaw(bottle_pressure_mpa));
        PackU16BE(&payload[4], EncodePressureRaw(inlet_pressure_mpa));
        payload[6] = 0U;
        payload[7] = 0U;
        RS485_Send_Frame(CAN_ID_FCU3, payload);
        break;

    default:
        if (Valve_Get_InletState() != 0U) {
            io_flags |= (1U << 0);
        }
        if (Valve_Get_PurgeState() != 0U) {
            io_flags |= (1U << 1);
        }
        if (Fan_Get_PowerState() != 0U) {
            io_flags |= (1U << 2);
        }
        if ((Fan_Get_PowerState() != 0U) && (Fan_Get_DutyCycle() > 0U)) {
            io_flags |= (1U << 3);
        }

        payload[0] = io_flags;
        payload[1] = Fan_Get_DutyCycle();
        PackU16BE(&payload[2], EncodeTempRaw(ambient_temp_c));
        PackU16BE(&payload[4], g_fault_code);
        payload[6] = 0U;
        payload[7] = 0U;
        RS485_Send_Frame(CAN_ID_FCU4, payload);
        break;
    }

    next_frame = (uint8_t)((next_frame + 1U) & 0x03U);
}

static void ForcePowerStageStop(void)
{
    PWM_SetDuty(0U);
    DCDC_Boost_Disable(&g_boost_ctrl);
    ADC_StartContinuousMode();
    TIM_Cmd(TIM3, DISABLE);
    TIM_ITConfig(TIM3, TIM_IT_Update, DISABLE);
    TIM_ClearITPendingBit(TIM3, TIM_IT_Update);
    g_tim3_initialized = 0U;
}

static void Timebase_Init(void)
{
    if (SysTick_Config(SystemCoreClock / 1000U) == 0U) {
        g_systick_ready = 1U;
    }
}

static void TIM3_Init(void)
{
    NVIC_InitTypeDef nvic_init;

    if (g_tim3_initialized != 0U) {
        TIM_Cmd(TIM3, ENABLE);
        TIM_ITConfig(TIM3, TIM_IT_Update, ENABLE);
        return;
    }

    RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM3, ENABLE);
    TIM_SelectOutputTrigger(TIM3, TIM_TRGOSource_Update);
    TIM_ITConfig(TIM3, TIM_IT_Update, ENABLE);

    nvic_init.NVIC_IRQChannel = TIM3_IRQn;
    nvic_init.NVIC_IRQChannelPreemptionPriority = 2U;
    nvic_init.NVIC_IRQChannelSubPriority = 0U;
    nvic_init.NVIC_IRQChannelCmd = ENABLE;
    NVIC_Init(&nvic_init);

    TIM_Cmd(TIM3, ENABLE);
    g_tim3_initialized = 1U;
}

static void SystemInitLayer(void)
{
    SystemInit();
    Timebase_Init();
    HardwareInitLayer();
}

static void HardwareInitLayer(void)
{
    RS485_Init();
    TransmitSerialBootProbe();

    ADC1_Mode_Config();
    ADC_StartContinuousMode();

    Valve_Init();
    Fan_Init();

    DCDC_Boost_Init(&g_boost_ctrl);
    DCDC_Boost_SetRatedParams(&g_boost_ctrl, DCF_OUT_VOLT_TARGET_DEF_V, DCF_IN_CURRENT_REF_DEF_A, 13.0f);
    DCDC_Boost_SetTargetVoltage(&g_boost_ctrl, DCF_OUT_VOLT_TARGET_DEF_V);
    DCDC_Boost_SetTargetCurrent(&g_boost_ctrl, DCF_IN_CURRENT_REF_DEF_A);
    DCDC_Boost_Enable(&g_boost_ctrl);

    PWM_Init(PWM_FREQ_100KHZ);
    PWM_Enable();

    LoadDefaultSerialControlPayload();
    g_runtime_ready = 1U;
}

static uint8_t CalibrationLayer(void)
{
    static uint8_t calibration_state = 0U;

    switch (calibration_state) {
    case 0U:
        VICalibration_Start();
        g_calibration_status = 1U;
        calibration_state = 1U;
        break;

    case 1U:
        if (VICalibration_Update() != 0U) {
            g_calibration_status = 2U;
            calibration_state = 2U;
            return 1U;
        }
        break;

    case 2U:
        return 1U;

    default:
        calibration_state = 0U;
        break;
    }

    return 0U;
}

static void InterruptInitLayer(void)
{
    TIM3_Init();
}

static uint8_t SoftStartLayer(float *input_voltage, float *output_voltage,
                              float *input_current, float *output_current,
                              uint16_t *duty, uint8_t reset)
{
    static uint8_t soft_start_state = 0U;
    uint8_t completed;

    if (reset != 0U) {
        soft_start_state = 0U;
    }

    switch (soft_start_state) {
    case 0U:
        DCDC_Boost_EnableSoftStart(&g_boost_ctrl, g_boost_ctrl.target_voltage);
        g_soft_start_progress = 0U;
        soft_start_state = 1U;
        break;

    case 1U:
        *duty = 0U;
        completed = DCDC_Boost_ProcessSoftStart(&g_boost_ctrl, *input_voltage,
                                                *output_voltage, *output_current,
                                                *input_current, duty);

        if (completed != 0U) {
            DCDC_Boost_SyncPIDState(&g_boost_ctrl, *duty);
            TIM3_Init();
            g_soft_start_progress = 100U;
            soft_start_state = 2U;
            return 1U;
        }

        g_soft_start_progress = (uint8_t)(((*duty) - DCDC_BOOST_MIN_DUTY) * 100U /
                                          (DCDC_BOOST_MAX_DUTY - DCDC_BOOST_MIN_DUTY));
        break;

    case 2U:
        return 1U;

    default:
        soft_start_state = 0U;
        break;
    }

    return 0U;
}

static void MainScheduler(void)
{
    static SystemState_t prev_state = SYS_OFF;
    static uint32_t state_start_time = 0U;
    static uint16_t current_duty = 0U;
    float input_voltage = 0.0f;
    float output_voltage = 0.0f;
    float input_current = 0.0f;
    float output_current = 0.0f;
    float ntc1_temp = 0.0f;
    float ntc2_temp = 0.0f;
    float inlet_pressure = 0.0f;
    float bottle_pressure = 0.0f;

    if (g_runtime_ready != 0U) {
        input_voltage = Get_CalibratedInputVoltage();
        output_voltage = Get_CalibratedOutputVoltage();
        input_current = Get_CalibratedInputCurrent();
        output_current = Get_CalibratedOutputCurrent();
        ntc1_temp = Get_NTC1_Temperature();
        ntc2_temp = Get_NTC2_Temperature();
        inlet_pressure = Get_Inlet_Pressure_Value();
        bottle_pressure = Get_Bottle_Pressure_Value();

        g_input_voltage = input_voltage;
        g_output_voltage = output_voltage;
        g_input_current = input_current;
        g_output_current = output_current;

        if (ntc1_temp > -100.0f) {
            g_ntc1_temperature = ntc1_temp;
        }
        if (ntc2_temp > -100.0f) {
            g_ntc2_temperature = ntc2_temp;
        }

        PollSerialControlPayload();
        ApplySerialControl();
        TransmitSerialStatusFrames(input_voltage, output_voltage, input_current, output_current,
                                   ntc1_temp, ntc2_temp, bottle_pressure, inlet_pressure);

        if ((g_system_state == SYS_WAIT_POWER) || (g_system_state == SYS_POWER_LOSS)) {
            PWM_SetDuty(0U);
            current_duty = 0U;
        }

        if ((g_run_request != 0U) || (g_system_state == SYS_RUNNING) ||
            (g_system_state == SYS_SOFT_START) || (g_system_state == SYS_FAULT)) {
            DCDC_Boost_CheckFault(&g_boost_ctrl, input_voltage, output_voltage, output_current);
        }
    }

    switch (g_system_state) {
    case SYS_OFF:
        g_system_state = SYS_INIT;
        state_start_time = g_ms;
        break;

    case SYS_INIT:
        SystemInitLayer();
        g_system_state = SYS_CALIBRATION;
        state_start_time = g_ms;
        break;

    case SYS_CALIBRATION:
        if (CalibrationLayer() != 0U) {
            g_system_state = SYS_WAIT_POWER;
            state_start_time = g_ms;
        }
        break;

    case SYS_WAIT_POWER:
        if ((g_runtime_ready != 0U) && (g_run_request != 0U) &&
            (DCDC_Boost_IsPowerOn(&g_boost_ctrl, input_voltage) != 0U)) {
            ADC_StartContinuousMode();
            DCDC_Boost_Enable(&g_boost_ctrl);
            DCDC_Boost_ClearFault(&g_boost_ctrl);
            PID_Reset(&g_boost_ctrl.voltage_pid);
            PID_Reset(&g_boost_ctrl.current_pid);
            PID_Reset_Int(&g_boost_ctrl.voltage_pid_int);
            PID_Reset_Int(&g_boost_ctrl.current_pid_int);
            g_system_state = SYS_SOFT_START;
            state_start_time = g_ms;
        }
        break;

    case SYS_SOFT_START:
        if (g_run_request == 0U) {
            g_system_state = SYS_SHUTDOWN;
            state_start_time = g_ms;
            break;
        }

        {
            uint8_t reset = (uint8_t)(prev_state != SYS_SOFT_START);
            uint16_t soft_start_duty = 0U;

            if (reset != 0U) {
                ADC_StartContinuousMode();
            }

            if (SoftStartLayer(&input_voltage, &output_voltage, &input_current,
                               &output_current, &soft_start_duty, reset) != 0U) {
                ADC_StartTriggeredMode();
                g_system_state = SYS_RUNNING;
                state_start_time = g_ms;
                current_duty = soft_start_duty;
            } else {
                PWM_SetDuty(soft_start_duty);
                current_duty = soft_start_duty;
            }
        }
        break;

    case SYS_RUNNING:
        if (g_run_request == 0U) {
            g_system_state = SYS_SHUTDOWN;
            state_start_time = g_ms;
        }
        break;

    case SYS_FAULT:
        if (g_run_request == 0U) {
            g_system_state = SYS_SHUTDOWN;
            state_start_time = g_ms;
        } else {
            static uint8_t fault_state = 0U;
            static uint8_t fault_retry_count = 0U;
            static uint32_t fault_delay_start = 0U;

            if (prev_state != SYS_FAULT) {
                fault_state = 0U;
                fault_retry_count = 0U;
                fault_delay_start = g_ms;
                TIM_Cmd(TIM3, DISABLE);
                TIM_ITConfig(TIM3, TIM_IT_Update, DISABLE);
                TIM_ClearITPendingBit(TIM3, TIM_IT_Update);
            }

            switch (fault_state) {
            case 0U:
                PWM_SetDuty(0U);
                fault_state = 1U;
                fault_delay_start = g_ms;
                break;

            case 1U:
                if ((uint32_t)(g_ms - fault_delay_start) >= FAULT_RECOVERY_DELAY_MS) {
                    fault_state = 2U;
                }
                break;

            case 2U:
                if (g_boost_ctrl.fault == DCDC_BOOST_FAULT_NONE) {
                    ++fault_retry_count;
                    PID_Reset(&g_boost_ctrl.voltage_pid);
                    PID_Reset(&g_boost_ctrl.current_pid);

                    if (fault_retry_count <= FAULT_RECOVERY_MAX_RETRIES) {
                        fault_state = 3U;
                    } else {
                        fault_state = 4U;
                    }
                } else {
                    fault_delay_start = g_ms;
                    fault_state = 1U;
                }
                break;

            case 3U:
                if (DCDC_Boost_IsPowerOn(&g_boost_ctrl, g_input_voltage) != 0U) {
                    DCDC_Boost_ClearFault(&g_boost_ctrl);
                    PID_Reset(&g_boost_ctrl.voltage_pid);
                    PID_Reset(&g_boost_ctrl.current_pid);
                    PID_Reset_Int(&g_boost_ctrl.voltage_pid_int);
                    PID_Reset_Int(&g_boost_ctrl.current_pid_int);
                    fault_state = 0U;
                    fault_retry_count = 0U;
                    g_system_state = SYS_SOFT_START;
                    state_start_time = g_ms;
                } else {
                    fault_state = 2U;
                }
                break;

            case 4U:
                if (g_boost_ctrl.fault == DCDC_BOOST_FAULT_NONE) {
                    fault_state = 3U;
                }
                break;

            default:
                fault_state = 0U;
                break;
            }
        }
        break;

    case SYS_POWER_LOSS:
        ForcePowerStageStop();
        g_system_state = SYS_WAIT_POWER;
        state_start_time = g_ms;
        break;

    case SYS_SHUTDOWN:
        ForcePowerStageStop();
        g_system_state = SYS_WAIT_POWER;
        state_start_time = g_ms;
        break;

    default:
        g_system_state = SYS_WAIT_POWER;
        state_start_time = g_ms;
        break;
    }

    if ((g_runtime_ready != 0U) && (g_run_request != 0U) && (g_emergency_stop == 0U) &&
        (g_boost_ctrl.fault != DCDC_BOOST_FAULT_NONE) && (g_system_state != SYS_FAULT)) {
        g_system_state = SYS_FAULT;
        g_last_fault = g_boost_ctrl.fault;
        state_start_time = g_ms;
    }

    if ((g_runtime_ready != 0U) && (g_run_request != 0U) &&
        ((g_system_state == SYS_RUNNING) || (g_system_state == SYS_SOFT_START)) &&
        (input_voltage < g_boost_ctrl.min_input_voltage)) {
        g_system_state = SYS_POWER_LOSS;
        state_start_time = g_ms;
    }

    if (g_system_state != prev_state) {
        prev_state = g_system_state;
        g_state_timestamp = state_start_time;
    }

    g_state_duration = g_ms - g_state_timestamp;
    g_fault_code = EncodeFaultCode();
    g_fault_level = MapFaultLevel(g_boost_ctrl.fault, g_system_state);
    g_state = g_system_state;
}

int main(void)
{
    g_system_state = SYS_OFF;

    while (1) {
        MainScheduler();
    }
}
