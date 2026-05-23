#include "../Hardware/serial_protocol.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

static void test_build_frame_uses_can_id_payload_and_sum_checksum(void)
{
    uint8_t payload[RS485_SERIAL_DATA_LEN] = {
        0x12U, 0x02U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U
    };
    uint8_t frame[RS485_SERIAL_FRAME_LEN] = {0U};
    uint8_t expected[RS485_SERIAL_FRAME_LEN] = {
        0xAAU, 0x55U,
        0xF0U, 0x01U, 0xFFU, 0x18U,
        0x08U,
        0x12U, 0x02U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
        0x24U
    };

    assert(SerialProtocol_BuildFrame(0x18FF01F0UL, payload, sizeof(payload), frame, sizeof(frame)) == 1U);
    assert(memcmp(frame, expected, sizeof(expected)) == 0);
}

static void test_parse_rejects_bad_checksum(void)
{
    uint8_t frame[RS485_SERIAL_FRAME_LEN] = {
        0xAAU, 0x55U,
        0xA0U, 0x10U, 0xFFU, 0x18U,
        0x08U,
        0x04U, 0x19U, 0x50U, 0xE0U, 0x01U, 0x64U, 0x00U, 0x00U,
        0x00U
    };
    uint32_t frame_id = 0U;
    uint8_t payload[RS485_SERIAL_DATA_LEN] = {0U};
    uint8_t payload_len = 0U;

    assert(SerialProtocol_ParseFrame(frame, sizeof(frame), &frame_id, payload, &payload_len) == 0U);
}

static void test_parse_extracts_valid_control_frame(void)
{
    uint8_t frame[RS485_SERIAL_FRAME_LEN] = {
        0xAAU, 0x55U,
        0xA0U, 0x10U, 0xFFU, 0x18U,
        0x08U,
        0x04U, 0x19U, 0x50U, 0xE0U, 0x01U, 0x64U, 0x00U, 0x00U,
        0x81U
    };
    uint8_t expected_payload[RS485_SERIAL_DATA_LEN] = {
        0x04U, 0x19U, 0x50U, 0xE0U, 0x01U, 0x64U, 0x00U, 0x00U
    };
    uint32_t frame_id = 0U;
    uint8_t payload[RS485_SERIAL_DATA_LEN] = {0U};
    uint8_t payload_len = 0U;

    assert(SerialProtocol_ParseFrame(frame, sizeof(frame), &frame_id, payload, &payload_len) == 1U);
    assert(frame_id == 0x18FF10A0UL);
    assert(payload_len == RS485_SERIAL_DATA_LEN);
    assert(memcmp(payload, expected_payload, sizeof(expected_payload)) == 0);
}

int main(void)
{
    test_build_frame_uses_can_id_payload_and_sum_checksum();
    test_parse_rejects_bad_checksum();
    test_parse_extracts_valid_control_frame();
    return 0;
}
