#ifndef __SERIAL_PROTOCOL_H
#define __SERIAL_PROTOCOL_H

#include <stdint.h>

#define RS485_SERIAL_HEAD0              0xAAU
#define RS485_SERIAL_HEAD1              0x55U
#define RS485_SERIAL_DATA_LEN           8U
#define RS485_SERIAL_FRAME_LEN          16U

uint8_t SerialProtocol_BuildFrame(uint32_t frame_id,
                                  const uint8_t *payload,
                                  uint8_t payload_len,
                                  uint8_t *out_frame,
                                  uint8_t out_size);

uint8_t SerialProtocol_ParseFrame(const uint8_t *frame,
                                  uint8_t frame_len,
                                  uint32_t *frame_id,
                                  uint8_t *payload,
                                  uint8_t *payload_len);

#endif
