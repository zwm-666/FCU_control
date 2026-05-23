#include "serial_protocol.h"

#include <string.h>

static uint8_t SerialProtocol_Checksum(const uint8_t *data, uint8_t len)
{
    uint8_t checksum = 0U;
    uint8_t index;

    for (index = 0U; index < len; ++index) {
        checksum = (uint8_t)(checksum + data[index]);
    }

    return checksum;
}

uint8_t SerialProtocol_BuildFrame(uint32_t frame_id,
                                  const uint8_t *payload,
                                  uint8_t payload_len,
                                  uint8_t *out_frame,
                                  uint8_t out_size)
{
    if ((payload == 0) || (out_frame == 0) ||
        (payload_len != RS485_SERIAL_DATA_LEN) ||
        (out_size < RS485_SERIAL_FRAME_LEN)) {
        return 0U;
    }

    out_frame[0] = RS485_SERIAL_HEAD0;
    out_frame[1] = RS485_SERIAL_HEAD1;
    out_frame[2] = (uint8_t)(frame_id & 0x000000FFUL);
    out_frame[3] = (uint8_t)((frame_id >> 8U) & 0x000000FFUL);
    out_frame[4] = (uint8_t)((frame_id >> 16U) & 0x000000FFUL);
    out_frame[5] = (uint8_t)((frame_id >> 24U) & 0x000000FFUL);
    out_frame[6] = payload_len;
    memcpy(&out_frame[7], payload, RS485_SERIAL_DATA_LEN);
    out_frame[15] = SerialProtocol_Checksum(&out_frame[2], 13U);

    return 1U;
}

uint8_t SerialProtocol_ParseFrame(const uint8_t *frame,
                                  uint8_t frame_len,
                                  uint32_t *frame_id,
                                  uint8_t *payload,
                                  uint8_t *payload_len)
{
    uint32_t parsed_id;

    if ((frame == 0) || (frame_id == 0) || (payload == 0) || (payload_len == 0) ||
        (frame_len != RS485_SERIAL_FRAME_LEN)) {
        return 0U;
    }

    if ((frame[0] != RS485_SERIAL_HEAD0) || (frame[1] != RS485_SERIAL_HEAD1)) {
        return 0U;
    }

    if (frame[6] != RS485_SERIAL_DATA_LEN) {
        return 0U;
    }

    if (frame[15] != SerialProtocol_Checksum(&frame[2], 13U)) {
        return 0U;
    }

    parsed_id = (uint32_t)frame[2] |
                ((uint32_t)frame[3] << 8U) |
                ((uint32_t)frame[4] << 16U) |
                ((uint32_t)frame[5] << 24U);

    *frame_id = parsed_id;
    *payload_len = frame[6];
    memcpy(payload, &frame[7], RS485_SERIAL_DATA_LEN);

    return 1U;
}
