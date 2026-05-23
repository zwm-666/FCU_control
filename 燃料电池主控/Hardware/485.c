#include "485.h"

#include <string.h>

volatile uint8_t RS485_RX_BUF[RS485_RX_BUF_SIZE];
volatile uint8_t RS485_RX_CNT = 0U;

void RS485_Init(void)
{
    GPIO_InitTypeDef gpio_init;
    USART_InitTypeDef usart_init;
    NVIC_InitTypeDef nvic_init;

    memset((void *)RS485_RX_BUF, 0, sizeof(RS485_RX_BUF));
    RS485_RX_CNT = 0U;

    RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1 | RCC_APB2Periph_GPIOA, ENABLE);

    gpio_init.GPIO_Pin = GPIO_Pin_9;
    gpio_init.GPIO_Speed = GPIO_Speed_50MHz;
    gpio_init.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_Init(GPIOA, &gpio_init);

    gpio_init.GPIO_Pin = GPIO_Pin_10;
    gpio_init.GPIO_Mode = GPIO_Mode_IPU;
    GPIO_Init(GPIOA, &gpio_init);

    usart_init.USART_BaudRate = RS485_BAUDRATE;
    usart_init.USART_WordLength = USART_WordLength_8b;
    usart_init.USART_StopBits = USART_StopBits_1;
    usart_init.USART_Parity = USART_Parity_No;
    usart_init.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
    usart_init.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
    USART_Init(USART1, &usart_init);

    nvic_init.NVIC_IRQChannel = USART1_IRQn;
    nvic_init.NVIC_IRQChannelPreemptionPriority = 3U;
    nvic_init.NVIC_IRQChannelSubPriority = 3U;
    nvic_init.NVIC_IRQChannelCmd = ENABLE;
    NVIC_Init(&nvic_init);

    USART_ITConfig(USART1, USART_IT_RXNE, ENABLE);
    USART_Cmd(USART1, ENABLE);
}

void RS485_Send_Byte(uint8_t byte)
{
    USART_SendData(USART1, byte);
    while (USART_GetFlagStatus(USART1, USART_FLAG_TC) == RESET) {
    }
}

void RS485_Send_String(const char *str)
{
    if (str == 0) {
        return;
    }

    while (*str != '\0') {
        RS485_Send_Byte((uint8_t)(*str));
        ++str;
    }
}

void RS485_Send_Data(const uint8_t *data, uint16_t len)
{
    uint16_t index;

    if (data == 0) {
        return;
    }

    for (index = 0U; index < len; ++index) {
        RS485_Send_Byte(data[index]);
    }
}

uint8_t RS485_Send_Frame(uint32_t frame_id, const uint8_t *payload)
{
    uint8_t frame[RS485_SERIAL_FRAME_LEN];

    if (SerialProtocol_BuildFrame(frame_id, payload, RS485_SERIAL_DATA_LEN,
                                  frame, sizeof(frame)) == 0U) {
        return 0U;
    }

    RS485_Send_Data(frame, sizeof(frame));
    return 1U;
}

uint8_t RS485_Poll_Frame(uint32_t *frame_id, uint8_t *payload)
{
    uint8_t local_buf[RS485_RX_BUF_SIZE];
    uint8_t local_count;
    uint8_t offset;
    uint8_t payload_len = 0U;

    if ((frame_id == 0) || (payload == 0) || (RS485_RX_CNT < RS485_SERIAL_FRAME_LEN)) {
        return 0U;
    }

    __disable_irq();
    local_count = RS485_RX_CNT;
    if (local_count > RS485_RX_BUF_SIZE) {
        local_count = RS485_RX_BUF_SIZE;
    }
    memcpy(local_buf, (const void *)RS485_RX_BUF, local_count);
    RS485_RX_CNT = 0U;
    __enable_irq();

    for (offset = 0U; (uint8_t)(offset + RS485_SERIAL_FRAME_LEN) <= local_count; ++offset) {
        if (SerialProtocol_ParseFrame(&local_buf[offset], RS485_SERIAL_FRAME_LEN,
                                      frame_id, payload, &payload_len) != 0U) {
            return 1U;
        }
    }

    return 0U;
}

void USART1_IRQHandler(void)
{
    if (USART_GetITStatus(USART1, USART_IT_RXNE) != RESET) {
        uint8_t data = (uint8_t)(USART_ReceiveData(USART1) & 0x00FFU);

        if (RS485_RX_CNT < RS485_RX_BUF_SIZE) {
            RS485_RX_BUF[RS485_RX_CNT] = data;
            ++RS485_RX_CNT;
        } else {
            RS485_RX_CNT = 0U;
        }

        USART_ClearITPendingBit(USART1, USART_IT_RXNE);
    }
}
