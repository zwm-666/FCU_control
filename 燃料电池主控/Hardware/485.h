#ifndef __RS485_H
#define __RS485_H

#include "stm32f10x.h"
#include "serial_protocol.h"
#include <stdint.h>

#define RS485_BAUDRATE                 9600U
#define RS485_RX_BUF_SIZE              128U

extern volatile uint8_t RS485_RX_BUF[RS485_RX_BUF_SIZE];
extern volatile uint8_t RS485_RX_CNT;

void RS485_Init(void);
void RS485_Send_Byte(uint8_t byte);
void RS485_Send_String(const char *str);
void RS485_Send_Data(const uint8_t *data, uint16_t len);
uint8_t RS485_Send_Frame(uint32_t frame_id, const uint8_t *payload);
uint8_t RS485_Poll_Frame(uint32_t *frame_id, uint8_t *payload);

#endif
