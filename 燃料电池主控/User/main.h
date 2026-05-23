#ifndef __MAIN_H
#define __MAIN_H

#include "stm32f10x.h"
#include <stdint.h>

#define CTRL_PERIOD_MS               10U
#define SENSOR_PERIOD_MS             50U
#define FAULT_PERIOD_MS              20U
#define COMM_PERIOD_MS               100U
#define STATUS_PERIOD_MS             500U

#define CAN_ID_FCU1                  0x18FF01F0UL
#define CAN_ID_FCU2                  0x18FF02F0UL
#define CAN_ID_FCU3                  0x18FF03F0UL
#define CAN_ID_FCU4                  0x18FF04F0UL
#define CAN_ID_HOST_CMD              0x18FF10A0UL
#define CAN_STD_ID_STATUS            0x181U
#define CAN_STD_ID_HOST_CMD          0x201U

#define CAN_CMD_TIMEOUT_MS           1000U

#define PURGE_INTERVAL_MS            10000U
#define PURGE_DURATION_MS            200U

#define TEMP_OVER_LIMIT_C            80.0f
#define TEMP_RECOVER_C               70.0f
#define FC_IN_UV_LIMIT_V             10.0f
#define FC_IN_UV_RECOVER_V           12.0f
#define FC_IN_OV_LIMIT_V             60.0f
#define FC_IN_OV_RECOVER_V           58.0f
#define DCF_OUT_OV_LIMIT_V           55.0f
#define DCF_OUT_OV_RECOVER_V         52.0f
#define IN_OC_LIMIT_A                38.0f
#define OUT_OC_LIMIT_A               12.0f

#define DCF_OUT_VOLT_TARGET_DEF_V    48.0f
#define DCF_OUT_VOLT_LIMIT_V         50.0f
#define DCF_IN_CURRENT_REF_DEF_A     10.0f
#define DCL_OUT_VOLT_TARGET_V        12.0f

#define DCF_KP                       0.25f
#define DCF_KI                       0.01f
#define DCL_KP                       0.50f
#define DCL_KI                       0.01f

#define CFG_MAGIC                    0x43465531UL
#define CFG_VERSION                  0x0001U
#define CFG_EE_ADDR                  0U
#define CFG_SAVE_MIN_INTERVAL_MS     1000U

#define ADC_MAP_VIN_IDX              2U
#define ADC_MAP_VOUT48_IDX           3U
#define ADC_MAP_VOUT12_IDX           4U

#define ACC_GPIO_PORT                GPIOB
#define ACC_GPIO_RCC                 RCC_APB2Periph_GPIOB
#define ACC_GPIO_PIN                 GPIO_Pin_11
#define KEY_GPIO_PORT                GPIOB
#define KEY_GPIO_RCC                 RCC_APB2Periph_GPIOB
#define KEY_GPIO_PIN                 GPIO_Pin_10
#define KEY_INPUT_ENABLE             0U

#define PWR_TIM                      TIM3
#define PWR_TIM_RCC                  RCC_APB1Periph_TIM3
#define PWR_GPIO_RCC                 RCC_APB2Periph_GPIOB
#define PWR_AFIO_RCC                 RCC_APB2Periph_AFIO
#define PWR_GPIO_PORT                GPIOB
#define DCF_PWM_PIN                  GPIO_Pin_4
#define DCL_PWM_PIN                  GPIO_Pin_5
#define DCL_EN_PIN                   GPIO_Pin_12
#define PWR_PWM_FREQ_HZ              10000U
#define PWR_PWM_MAX_DUTY             100U

#define SELFTEST_STEP_MS             1000U
#define FAN_STALL_RPM_TH             200U
#define FAN_STALL_DETECT_MS          3000U

#define FE_IN_UV                     (1UL << 0)
#define FE_IN_OV                     (1UL << 1)
#define FE_IN_OC                     (1UL << 2)
#define FE_OUT_OV                    (1UL << 3)
#define FE_OUT_OC                    (1UL << 4)
#define FE_OTP                       (1UL << 5)
#define FE_SENSOR                    (1UL << 6)
#define FE_PRESS_ABN                 (1UL << 7)
#define FE_CAN_TO                    (1UL << 8)
#define FE_FAN_STALL                 (1UL << 9)

typedef enum
{
    SYS_OFF = 0,            // 系统关闭
    SYS_INIT,               // 系统初始化
    SYS_CALIBRATION,        // 零点校准
    SYS_WAIT_POWER,         // 等待电源
    SYS_SOFT_START,         // 软启动
    SYS_RUNNING,            // 正常运行
    SYS_FAULT,              // 故障状态
    SYS_POWER_LOSS,         // 电源丢失
    SYS_SHUTDOWN            // 系统关闭中
} SystemState_t;

typedef enum
{
    FAULT_NONE   = 0,
    FAULT_OVP    = 1 << 0,
    FAULT_OCP    = 1 << 1,
    FAULT_OTP    = 1 << 2,
    FAULT_UVP    = 1 << 3,
    FAULT_SENSOR = 1 << 4,
    FAULT_CAN_TO = 1 << 5
} FaultMask_t;

typedef enum
{
    SELFTEST_IDLE = 0,
    SELFTEST_RUNNING
} SelfTestMode_t;

typedef struct
{
    float temp_stack_c;
    float temp_ambient_c;
    float pressure_inlet_mpa;
    float pressure_bottle_mpa;
    float h2_percent;
    float in_v;
    float out_v;
    float dcl_out_v;
    float in_a;
    float out_a;
    uint16_t fan_rpm;
} SensorData_t;

typedef struct
{
    uint8_t work_mode;
    uint8_t state_cmd;
    uint8_t force_inlet;
    uint8_t force_purge;
    uint8_t force_fan_pwr;
    uint8_t force_fan1;
    uint8_t fan1_target_pwm;
    float dcf_v_target;
    float dcf_i_target;
} HostCommand_t;

typedef struct
{
    uint32_t magic;
    uint16_t version;
    uint16_t length;
    float temp_over_c;
    float temp_recover_c;
    float in_uv_v;
    float in_uv_recover_v;
    float in_ov_v;
    float in_ov_recover_v;
    float out_ov_v;
    float out_ov_recover_v;
    float in_oc_a;
    float out_oc_a;
    float dcf_out_v_target;
    float dcf_in_i_target;
    uint16_t can_cmd_timeout_ms;
    uint16_t reserve;
    uint32_t crc;
} RuntimeConfig_t;

extern volatile uint32_t g_ms;
extern uint16_t g_fault_code;
extern uint8_t g_fault_level;
extern SystemState_t g_state;

int main(void);

#endif
