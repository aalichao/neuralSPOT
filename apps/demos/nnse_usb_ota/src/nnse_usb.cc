#include <stdint.h>
#include <cstring>
#include <cstddef>
#include <stdlib.h>
#include <string.h>
#include "seCntrlClass.h"
#include "am_util_stdio.h"
#include "ns_peripherals_button.h"
#include "ns_peripherals_power.h"
#include "ns_ambiqsuite_harness.h"
#include "ns_timer.h"
#include "ns_perf_profile.h"
#include "ns_usb.h"
#include "crc32.h"

#include "ae_api.h"
#include "FreeRTOS.h"
#include "task.h"

#include "ns_malloc.h"

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

// Model upload state
typedef struct {
    uint8_t* model_buffer;
    uint32_t model_size;
    uint32_t total_chunks;
    uint32_t received_chunks;
    bool upload_in_progress;
    bool upload_complete;
} model_upload_state_t;

// Maximum model size (adjust as needed)
#define MAX_MODEL_SIZE (512 * 1024) // 512KB

// Global model upload state
static model_upload_state_t model_state = {0};

// Speech Enhancement Control Structure
seCntrlClass cntrl_inst;

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

// FreeRTOS Tasks
TaskHandle_t my_xSetupTask;      // Starts the other tasks and suspends itself

// Timer for basic timing
ns_timer_config_t basic_tickTimer = {
    .api = &ns_timer_V1_0_0,
    .timer = NS_TIMER_COUNTER,
    .enableInterrupt = false,
};

// Model upload functions
void sendAck(uint32_t chunk_id) {
    // Send 5-byte ACK: 0xAA, chunk_id (LE)
    uint8_t ack[5];
    ack[0] = 0xAA;
    ack[1] = (uint8_t)(chunk_id & 0xFF);
    ack[2] = (uint8_t)((chunk_id >> 8) & 0xFF);
    ack[3] = (uint8_t)((chunk_id >> 16) & 0xFF);
    ack[4] = (uint8_t)((chunk_id >> 24) & 0xFF);
    ns_lp_printf("Sending ACK bytes: 0x%02X %08X\n", ack[0], chunk_id);
    webusb_send_data(ack, 5);
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
    
    ns_lp_printf("Received model chunk: id=%u, total=%u, length=%u\n", 
                 header->chunk_id, header->total_chunks, payload_length);
    
    // Verify CRC32
    uint32_t calculated_crc = CalcCrc32(0xFFFFFFFF, payload_length, (uint8_t*)payload);
    if (calculated_crc != header->crc32) {
        ns_lp_printf("Error: CRC32 mismatch (calc=0x%08X, recv=0x%08X)\n", calculated_crc, header->crc32);
        return;
    }
    
    if (header->command == CHUNK_CMD_MODEL_DATA) {
        // Handle model data chunk
        if (!model_state.upload_in_progress) {
            // First chunk - initialize upload
            model_state.total_chunks = header->total_chunks;
            model_state.received_chunks = 0;
            model_state.upload_in_progress = true;
            model_state.upload_complete = false;
            
            ns_lp_printf("Starting model upload: %d chunks\n", model_state.total_chunks);
            
            // Allocate buffer for the full model
            if (model_state.model_buffer) {
                // Free existing buffer if any
                free(model_state.model_buffer);
            }
            
            // Estimate model size (will be corrected on last chunk)
            uint32_t estimated_size = header->total_chunks * payload_length;
            ns_lp_printf("Estimated model size: %d bytes\n", estimated_size);

            if (estimated_size > MAX_MODEL_SIZE) {
                ns_lp_printf("Error: Model too large\n");
                model_state.upload_in_progress = false;
                return;
            }
            
            model_state.model_buffer = (uint8_t*)ns_malloc(estimated_size);
            if (!model_state.model_buffer) {
                ns_lp_printf("Error: Failed to allocate model buffer\n");
                model_state.upload_in_progress = false;
                return;
            }
            
            model_state.model_size = estimated_size;
        }
        
        // Copy chunk data to buffer
        uint32_t offset = header->chunk_id * (model_state.model_size / model_state.total_chunks);
        if (offset + payload_length <= model_state.model_size) {
            memcpy(model_state.model_buffer + offset, payload, payload_length);
            model_state.received_chunks++;
            
            ns_lp_printf("Chunk %u stored at offset %u, received %d/%d\n", 
                         header->chunk_id, offset, model_state.received_chunks, model_state.total_chunks);
            
            // Check if this is the last chunk
            if (header->chunk_id == header->total_chunks - 1) {
                // Set the true model size based on the last chunk
                model_state.model_size = offset + payload_length;
                model_state.upload_complete = true;
                model_state.upload_in_progress = false;
                
                ns_lp_printf("Model upload complete: %d bytes\n", model_state.model_size);
                
                // Initialize the model with seCntrlClass
                if (model_state.model_buffer && model_state.model_size > 0) {
                    // Call seCntrlClass_init with the uploaded model
                    seCntrlClass_init(&cntrl_inst);
                    ns_lp_printf("Model initialized successfully\n");
                }
            }
        }
        
        // Send ACK for this chunk
        ns_lp_printf("Sending ACK for chunk %u\n", header->chunk_id);
        sendAck(header->chunk_id);
        
    } else {
        ns_lp_printf("Unknown command: %d\n", header->command);
    }
}

void setup_task(void *pvParameters) {
    ns_lp_printf("Setting up USB Model Upload Task\n");
    vTaskSuspend(NULL);
    while (1)
        ;
}

void msgReceived(const uint8_t *buffer, uint32_t length, void *args) {
    // Log all received data for debugging
    ns_lp_printf("=== msgReceived called ===\n");
    ns_lp_printf("Received %d bytes\n", length);
    
    // Print first few bytes for debugging
    if (length > 0) {
        ns_lp_printf("First bytes: 0x%02X 0x%02X 0x%02X 0x%02X\n", 
                     buffer[0], buffer[1], buffer[2], buffer[3]);
    }
    
    // Send a simple response back to verify communication
    uint8_t response[] = {0x52, 0x45, 0x43, 0x56, 0x00}; // "RECV" + null
    webusb_send_data(response, 5);
    ns_lp_printf("Sent response: RECV\n");
    
    // Check for simple PING test (4 bytes: "PING")
    if (length == 4 && buffer[0] == 'P' && buffer[1] == 'I' && buffer[2] == 'N' && buffer[3] == 'G') {
        ns_lp_printf("[TEST] PING packet received!\n");
        return;
    }
    
    // Check if this is a model chunk (has 13-byte header)
    if (length >= 13) {
        const usb_message_header_t* header = (const usb_message_header_t*)buffer;
        
        ns_lp_printf("Header: CRC32=0x%08X, cmd=%d, chunk=%u, total=%u\n", 
                     header->crc32, header->command, header->chunk_id, header->total_chunks);
        
        // Check if this is a model data chunk
        if (header->command == CHUNK_CMD_MODEL_DATA) {
            ns_lp_printf("Processing model chunk\n");
            handle_model_chunk(buffer, length);
            return;
        } else {
            ns_lp_printf("Not a model chunk (cmd=%d)\n", header->command);
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

    // Initialize the timer
    NS_TRY(ns_timer_init(&basic_tickTimer), "Timer init failed.\n");

    // initialize neural nets controller
    seCntrlClass_init(&cntrl_inst);
    
    // Create setup task
    xTaskCreate(setup_task, "Setup", 512, 0, 1, &my_xSetupTask);

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
