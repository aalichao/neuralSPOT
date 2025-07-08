#include <stdint.h>
#include <cstring>
#include <cstddef>
#include <stdlib.h>
#include <string.h>
#include "am_util_stdio.h"
#include "ns_peripherals_button.h"
#include "ns_peripherals_power.h"
#include "ns_ambiqsuite_harness.h"
#include "ns_perf_profile.h"
#include "ns_usb.h"
#include "crc32.h"

#include "ae_api.h"
#include "FreeRTOS.h"
#include "task.h"

#include "ns_malloc.h"
#include "ns_model.h"

// Message header structure (13 bytes)
typedef struct {
    uint32_t crc32;
    uint8_t command;
    uint32_t chunk_id;
    uint32_t total_chunks;
} __attribute__((packed)) usb_message_header_t;

// Chunk commands
#define CHUNK_CMD_MODEL_DATA 0x01
#define CHUNK_CMD_ACK        0x02
#define CHUNK_CMD_CONFIG     0x03
#define CHUNK_CMD_RUN_STATS  0x04

// Maximum model sizes for TCM and SRAM
#define TCM_MODEL_SIZE  (256 * 1024) // 256KB
#define SRAM_MODEL_SIZE (512 * 1024) // 512KB

// Statically allocated model arrays
NS_PUT_IN_TCM alignas(16) static uint8_t tcm_model_array[TCM_MODEL_SIZE];
AM_SHARED_RW alignas(16) static uint8_t sram_model_array[SRAM_MODEL_SIZE];

typedef enum { MODEL_LOC_TCM = 0, MODEL_LOC_SRAM = 1 } model_location_t;
static model_location_t selected_model_location = MODEL_LOC_TCM;

// Model upload state
typedef struct {
    uint8_t* model_buffer;
    uint32_t model_size;
    uint32_t total_chunks;
    uint32_t received_chunks;
    bool upload_in_progress;
    bool upload_complete;
} model_upload_state_t;

// Global model upload state
static model_upload_state_t model_state = {0};

// WebUSB Configuration and Datatypes
#define MY_RX_BUFSIZE 4096
#define MY_TX_BUFSIZE 4096

static uint8_t my_rx_ff_buf[MY_RX_BUFSIZE] __attribute__((aligned(16)));
static uint8_t my_tx_ff_buf[MY_TX_BUFSIZE] __attribute__((aligned(16)));

// WebUSB URL
static ns_tusb_desc_webusb_url_t webusb_url;
static ns_usb_config_t webUsbConfig = {
    .api = &ns_usb_V1_0_0,
    .deviceType = NS_USB_VENDOR_DEVICE,
    .rx_buffer = NULL,
    .rx_bufferLength = 0,
    .tx_buffer = NULL,
    .tx_bufferLength = 0,
    .rx_cb = NULL,
    .tx_cb = NULL,
    .service_cb = NULL,
    .desc_url = &webusb_url // Filled in at runtime
};

// Custom power mode for USB only
const ns_power_config_t ns_power_usb = {
    .api = &ns_power_V1_0_0,
    .eAIPowerMode = NS_MAXIMUM_PERF,
    .bNeedAudAdc = false,
    .bNeedSharedSRAM = true,
    .bNeedCrypto = false,
    .bNeedBluetooth = false,
    .bNeedUSB = true,
    .bNeedIOM = false,
    .bNeedAlternativeUART = false,
    .b128kTCM = false,
    .bEnableTempCo = false,
    .bNeedITM = true,
    .bNeedXtal = true};

// Model upload functions
void sendAck(uint32_t chunk_id) {
    // Send 5-byte ACK: 0xAA, chunk_id (LE)
    uint8_t ack[5];
    ack[0] = 0xAA;
    ack[1] = (uint8_t)(chunk_id & 0xFF);
    ack[2] = (uint8_t)((chunk_id >> 8) & 0xFF);
    ack[3] = (uint8_t)((chunk_id >> 16) & 0xFF);
    ack[4] = (uint8_t)((chunk_id >> 24) & 0xFF);
    // ns_lp_printf("Sending ACK bytes: 0x%02X %08X\n", ack[0], chunk_id);
    webusb_send_data(ack, 5);
}

// Handle configuration message to set model location
void handle_model_config(const uint8_t* data, uint32_t length) {
    if (length < 1) return;
    selected_model_location = (data[0] == 1) ? MODEL_LOC_SRAM : MODEL_LOC_TCM;
    ns_lp_printf("Model location set to: %s\n", selected_model_location == MODEL_LOC_SRAM ? "SRAM" : "TCM");
}

void handle_model_chunk(const uint8_t* data, uint32_t length) {
    if (length < 13) {
        ns_lp_printf("Error: Message too short\n");
        return;
    }
    // Parse header
    const usb_message_header_t* header = (const usb_message_header_t*)data;
    const uint8_t* payload = data + 13;
    uint32_t payload_length = length - 13;
    ns_lp_printf("Received model chunk: id=%u, total=%u, length=%u\n", header->chunk_id, header->total_chunks, payload_length);
    // Verify CRC32
    uint32_t calculated_crc = CalcCrc32(0xFFFFFFFF, payload_length, (uint8_t*)payload);
    if (calculated_crc != header->crc32) {
        ns_lp_printf("Error: CRC32 mismatch (calc=0x%08X, recv=0x%08X)\n", calculated_crc, header->crc32);
        return;
    }
    if (header->command == CHUNK_CMD_MODEL_DATA) {
        if (!model_state.upload_in_progress) {
            model_state.total_chunks = header->total_chunks;
            model_state.received_chunks = 0;
            model_state.upload_in_progress = true;
            model_state.upload_complete = false;
            ns_lp_printf("Starting model upload: %d chunks\n", model_state.total_chunks);
            uint32_t estimated_size = header->total_chunks * payload_length;
            ns_lp_printf("Estimated model size: %d bytes\n", estimated_size);
            uint8_t* model_buffer = (selected_model_location == MODEL_LOC_SRAM) ? sram_model_array : tcm_model_array;
            size_t model_buffer_size = (selected_model_location == MODEL_LOC_SRAM) ? SRAM_MODEL_SIZE : TCM_MODEL_SIZE;
            if (estimated_size > model_buffer_size) {
                ns_lp_printf("Error: Model too large for selected memory\n");
                model_state.upload_in_progress = false;
                return;
            }
            model_state.model_buffer = model_buffer;
            model_state.model_size = estimated_size;
        }
        uint32_t offset = header->chunk_id * (model_state.model_size / model_state.total_chunks);
        if (offset + payload_length <= model_state.model_size) {
            memcpy(model_state.model_buffer + offset, payload, payload_length);
            model_state.received_chunks++;
            // ns_lp_printf("Chunk %u stored at offset %u, received %d/%d\n", header->chunk_id, offset, model_state.received_chunks, model_state.total_chunks);
            if (header->chunk_id == header->total_chunks - 1) {
                model_state.model_size = offset + payload_length;
                model_state.upload_complete = true;
                model_state.upload_in_progress = false;
                ns_lp_printf("Model upload complete: %d bytes\n", model_state.model_size);
            }
        }
        // ns_lp_printf("Sending ACK for chunk %u\n", header->chunk_id);
        sendAck(header->chunk_id);
    } else {
        // ns_lp_printf("Unknown command: %d\n", header->command);
    }
}

#define ARENA_SIZE (128 * 1024) // Adjust as needed
alignas(16) static uint8_t tcm_arena[ARENA_SIZE];
AM_SHARED_RW alignas(16) static uint8_t sram_arena[ARENA_SIZE];
static ns_model_state_t model_state_struct;

void run_model_and_send_stats() {
    ns_lp_printf("=== run_model_and_send_stats called ===\n");
    
    if (!model_state.upload_complete) {
        ns_lp_printf("Model not uploaded yet\n");
        // Send error response
        uint8_t error_packet[8] = {0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF}; // 0 cycles, error status
        webusb_send_data(error_packet, 8);
        return;
    }
    
    ns_lp_printf("Model upload complete, size: %d bytes\n", model_state.model_size);
    
    // Select model buffer and arena
    uint8_t* model_data = (selected_model_location == MODEL_LOC_SRAM) ? sram_model_array : tcm_model_array;
    size_t model_len = model_state.model_size;
    uint8_t* arena = (selected_model_location == MODEL_LOC_SRAM) ? sram_arena : tcm_arena;
    
    ns_lp_printf("Using model location: %s\n", selected_model_location == MODEL_LOC_SRAM ? "SRAM" : "TCM");
    ns_lp_printf("Model data pointer: %p, size: %d\n", model_data, model_len);

    // Fill out model state struct
    memset(&model_state_struct, 0, sizeof(model_state_struct));
    model_state_struct.model_array = model_data;
    model_state_struct.arena = arena;
    model_state_struct.arena_size = ARENA_SIZE;
    
    ns_lp_printf("Initializing model...\n");
    int status = ns_model_init(&model_state_struct);
    if (status != 0) {
        ns_lp_printf("Model init failed with status: %d\n", status);
        // Send error response
        uint8_t error_packet[8] = {0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF}; // 0 cycles, error status
        webusb_send_data(error_packet, 8);
        return;
    }
    
    ns_lp_printf("Model init successful\n");
    ns_lp_printf("Number of input tensors: %d\n", model_state_struct.numInputTensors);
    
    // Optionally fill input tensor with zeros
    for (uint32_t i = 0; i < model_state_struct.numInputTensors; i++) {
        memset(model_state_struct.model_input[i]->data.int8, 0, model_state_struct.model_input[i]->bytes);
        ns_lp_printf("Input tensor %d: %d bytes\n", i, model_state_struct.model_input[i]->bytes);
    }
    
    ns_lp_printf("Starting inference...\n");
    // Profile inference

    TfLiteStatus invoke_status = model_state_struct.interpreter->Invoke();
    
    // ns_lp_printf("Inference complete: cycles=%d, status=%d\n", cycles, invoke_status);
    
    // Send stats over USB (cycles and status)
    uint8_t stats_packet[8];
    // memcpy(&stats_packet[0], &cycles, 4);
    memcpy(&stats_packet[4], &invoke_status, 4);
    
    ns_lp_printf("Sending stats packet: ");
    for (int i = 0; i < 8; i++) {
        ns_lp_printf("0x%02X ", stats_packet[i]);
    }
    ns_lp_printf("\n");
    
    webusb_send_data(stats_packet, 8);
    ns_lp_printf("Stats packet sent\n");
    ns_lp_printf("=== run_model_and_send_stats complete ===\n");
}

void msgReceived(const uint8_t *buffer, uint32_t length, void *args) {
    ns_lp_printf("=== msgReceived called ===\n");
    ns_lp_printf("Received %d bytes\n", length);
    if (length >= 13) {
        const usb_message_header_t* header = (const usb_message_header_t*)buffer;
        const uint8_t* payload = buffer + 13;
        uint32_t payload_length = length - 13;
        ns_lp_printf("Header: CRC32=0x%08X, cmd=%d, chunk=%u, total=%u\n", header->crc32, header->command, header->chunk_id, header->total_chunks);
        if (header->command == CHUNK_CMD_MODEL_DATA) {
            ns_lp_printf("Processing model chunk\n");
            handle_model_chunk(buffer, length);
            return;
        } else if (header->command == CHUNK_CMD_CONFIG) {
            ns_lp_printf("Processing configuration\n");
            handle_model_config(payload, payload_length);
            ns_lp_printf("Configuration received\n");
            return;
        } else if (header->command == CHUNK_CMD_RUN_STATS) {
            ns_lp_printf("Processing RUN_STATS command\n");
            run_model_and_send_stats();
            return;
        } else {
            ns_lp_printf("Unknown command: %d\n", header->command);
        }
    } else {
        ns_lp_printf("Packet too short for header (need 13, got %d)\n", length);
    }
    ns_lp_printf("=== end msgReceived ===\n");
}

int main(void) {
    usb_handle_t usb_handle = NULL;

    ns_core_config_t ns_core_cfg = {.api = &ns_core_V1_0_0};
    NS_TRY(ns_core_init(&ns_core_cfg), "Core init failed.\b");

    ns_power_config(&ns_power_usb);

    // Only turn HP while doing codec and AI
    NS_TRY(ns_set_performance_mode(NS_MINIMUM_PERF), "Set CPU Perf mode failed. ");

    ns_itm_printf_enable();
    ns_interrupt_master_enable();

    ns_init_perf_profiler();
    ns_start_perf_profiler();

    // Initialize the URL descriptor
    strcpy(webusb_url.url, "ambiqai.github.io/web-ble-dashboards/nnse-usb/");
    webusb_url.bDescriptorType = 3;
    webusb_url.bScheme = 1;
    webusb_url.bLength = 3 + sizeof(webusb_url.url) - 1;

    // WebUSB Setup - Register callback for raw data
    webusb_register_raw_cb(msgReceived, NULL);
    webUsbConfig.rx_buffer = my_rx_ff_buf;
    webUsbConfig.rx_bufferLength = MY_RX_BUFSIZE;
    webUsbConfig.tx_buffer = my_tx_ff_buf;
    webUsbConfig.tx_bufferLength = MY_TX_BUFSIZE;

    NS_TRY(ns_usb_init(&webUsbConfig, &usb_handle), "USB Init Failed\n");
    ns_lp_printf("USB Init Success\n");

    // Initialize model upload state
    model_state.model_buffer = NULL;
    model_state.model_size = 0;
    model_state.total_chunks = 0;
    model_state.received_chunks = 0;
    model_state.upload_in_progress = false;
    model_state.upload_complete = false;

    ns_lp_printf("Model upload firmware ready\n");
    
    // Send a simple heartbeat message to verify USB is working
    uint8_t heartbeat[] = {0x48, 0x45, 0x4C, 0x4C, 0x4F}; // "HELLO"
    webusb_send_data(heartbeat, 5);
    ns_lp_printf("Sent heartbeat message\n");

    vTaskStartScheduler();
    while (1) {
    };
}
