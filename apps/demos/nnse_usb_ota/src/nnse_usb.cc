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

// Add profiling includes
#include "ns_core.h"
#include "ns_energy_monitor.h"
#include "ns_pmu_utils.h"
#include "ns_pmu_map.h"
#include "ns_power_profile.h"

#include "arrhythmia_model_power_example_tensors.h"

#ifdef NS_MLPROFILE
#ifdef AM_PART_APOLLO5B
extern ns_pmu_config_t ns_microProfilerPMU;
extern ns_profiler_sidecar_t ns_microProfilerSidecar;
extern ns_profiler_event_stats_t ns_profiler_events_stats[NS_PROFILER_RPC_EVENTS_MAX];
#endif
#endif

// TFLM Config
static ns_model_state_t model;
volatile int example_status = 0; // Prevent the compiler from optimizing out while loops

static uint32_t g_num_layers = 0;
static uint32_t g_rv_count = 0;

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
#define CHUNK_CMD_PMU_CSV    0x05

// Maximum model sizes for TCM and SRAM
#define TCM_MODEL_SIZE  (250 * 1024) // 256KB
#define SRAM_MODEL_SIZE (512 * 1024) // 512KB

// Statically allocated model arrays
NS_PUT_IN_TCM alignas(16) static uint8_t tcm_model_array[TCM_MODEL_SIZE];
AM_SHARED_RW alignas(16) static uint8_t sram_model_array[SRAM_MODEL_SIZE];

typedef enum { MODEL_LOC_TCM = 0, MODEL_LOC_SRAM = 1 } model_location_t;
typedef enum { ARENA_LOC_TCM = 0, ARENA_LOC_SRAM = 1 } arena_location_t;
static model_location_t selected_model_location = MODEL_LOC_TCM;
static arena_location_t selected_arena_location = ARENA_LOC_TCM;

#define ARENA_SIZE (73 * 1024) // Increased from 16KB to 64KB for model tensor allocation
NS_PUT_IN_TCM alignas(16) static uint8_t tcm_arena[ARENA_SIZE];
AM_SHARED_RW alignas(16) static uint8_t sram_arena[ARENA_SIZE];

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

const ns_power_config_t ns_power_measurement = {
    .api = &ns_power_V1_0_0,
    .eAIPowerMode = NS_MAXIMUM_PERF,
    .bNeedAudAdc = false,
    .bNeedSharedSRAM = true,
    .bNeedCrypto = false,
    .bNeedBluetooth = false,
    .bNeedUSB = false,
    .bNeedIOM = false,
    .bNeedAlternativeUART = false,
    .b128kTCM = false,
    .bEnableTempCo = false,
    .bNeedITM = false};

// Profiling function based on tflm_profiling.cc approach
typedef struct {
    uint32_t num_layers;
    TfLiteStatus invoke_status;
} profiling_result_t;

// Callback for profiling
int tf_invoke() {
    // ns_lp_printf("tf_invoke called\n");
    model.interpreter->Invoke();
    return 0;
}

profiling_result_t profile_model_inference(ns_model_state_t *model) {
    profiling_result_t result = {0, kTfLiteOk};
    
    ns_lp_printf("Starting TFLM profiling...\n");
    
    // Initialize the model, get handle if successful
    ns_init_power_monitor_state();
    ns_set_power_monitor_state(NS_IDLE);
    
    // Dump power-related registers (optional)
    ns_lp_printf("Current power and performance register settings:\n");
    #ifdef AM_PART_APOLLO5B
    capture_snapshot(0);
    print_snapshot(0, false);
    #else
    ns_pp_ap5_snapshot(false, 0, false);
    ns_pp_ap5_snapshot(false, 0, true);
    #endif
    
    ns_set_power_monitor_state(3); // GPIO to signal profiling phase
    
    ns_lp_printf("First Run: basic capture\n");
    ns_set_power_monitor_state(0); // GPIO 00 indicates inference is under way
    
    #ifndef AM_PART_APOLLO5B
        ns_reset_perf_counters(); // Reset performance counters
        ns_start_perf_profiler(); // Start the profiler
    #endif
    
    // Run inference with TFLM profiling
    result.invoke_status = model->interpreter->Invoke();
    
    #ifndef AM_PART_APOLLO5B
        ns_stop_perf_profiler(); // Stop the profiler
    #endif
    
    // Log profiling data if available
    if (model->profiler != nullptr) {
        model->profiler->LogCsv(); // prints and captures events in buffer
        
        // Get number of layers from profiler sidecar
        #ifdef AM_PART_APOLLO5B
        result.num_layers = ns_microProfilerSidecar.captured_event_num;
        g_num_layers = result.num_layers;
        #endif
        
        ns_lp_printf("Number of layers: %d\n", result.num_layers);
    }

    ns_set_power_monitor_state(2); // GPIO 02 indicates profiling complete

    #ifdef AM_PART_APOLLO5B
        ns_lp_printf("Full PMU profiling run - will run model many times to capture all PMU counters\n");
        ns_lp_printf("Each . is an Invoke of the model, capturing 4 distinct PMU counters.\n");
        
        // Run the model repeatedly, capturing different PMU every time. The results
        // will accumulate in the events array (ns_profiler_event_stats_t). Print those 
        // prettily to the console, up until the limit of the event buffer (4096
        // events). 
        //
        // IMPORTANT: this assumes that every run is identical, which is true for most models,
        // but not all. Specifically, some models include a CALL_ONCE layer that will only run
        // the first time the model is invoked. That 'first run' is above, and is already
        // captured in the event buffer. If the model has a CALL_ONCE layer, the number of layers
        // will be different for the runs below, so the mapping of PMU events to layers must be 
        // adjusted.

        ns_characterize_model(tf_invoke);
        ns_lp_printf("\nPMU profiling .\n");
        ns_parse_pmu_stats(result.num_layers, model->rv_count); // Parse the PMU stats and print them out in CSV format
    #endif // AM_PART_APOLLO5B
    
    
    
    return result;
}

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

// Handle configuration message to set model location and arena location
void handle_model_config(const uint8_t* data, uint32_t length) {
    if (length < 1) return;
    selected_model_location = (data[0] == 1) ? MODEL_LOC_SRAM : MODEL_LOC_TCM;
    ns_lp_printf("Model location set to: %s\n", selected_model_location == MODEL_LOC_SRAM ? "SRAM" : "TCM");
    
    if (length >= 2) {
        selected_arena_location = (data[1] == 1) ? ARENA_LOC_SRAM : ARENA_LOC_TCM;
        ns_lp_printf("Arena location set to: %s\n", selected_arena_location == ARENA_LOC_SRAM ? "SRAM" : "TCM");
    } else {
        // Default arena location to match model location for backward compatibility
        selected_arena_location = (selected_model_location == MODEL_LOC_SRAM) ? ARENA_LOC_SRAM : ARENA_LOC_TCM;
        ns_lp_printf("Arena location defaulted to: %s\n", selected_arena_location == ARENA_LOC_SRAM ? "SRAM" : "TCM");
    }
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
            // uint8_t* model_buffer = sram_model_array;
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
#include "arrhythmia_model_power_model_data.h"

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
    // uint8_t* model_data = sram_model_array;
    // size_t model_len = model_state.model_size;
    uint8_t* arena = (selected_arena_location == ARENA_LOC_SRAM) ? sram_arena : tcm_arena;
    
    ns_lp_printf("Using model location: %s\n", selected_model_location == MODEL_LOC_SRAM ? "SRAM" : "TCM");
    // ns_lp_printf("Model data pointer: %p, size: %d\n", model_data, model_len);

    // Fill out model state struct
    memset(&model, 0, sizeof(model));
    model.model_array = model_data;
    // model.model_array = arrhythmia_model_power_model;
    model.arena = arena;
    model.arena_size = ARENA_SIZE;
    
    ns_lp_printf("Initializing model...\n");
    int status = ns_model_minimal_init(&model);
    if (status != 0) {
        ns_lp_printf("Model init failed with status: %d\n", status);
        // Send error response
        uint8_t error_packet[8] = {0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF}; // 0 cycles, error status
        webusb_send_data(error_packet, 8);
        return;
    }
    
    ns_lp_printf("Model init successful\n");
    ns_lp_printf("Number of input tensors: %d\n", model.numInputTensors);
    
    // Enhanced input tensor handling
    for (uint32_t i = 0; i < model.numInputTensors; i++) {
        TfLiteTensor* input_tensor = model.model_input[i];
        
        ns_lp_printf("Input tensor %d details:\n", i);
        ns_lp_printf("  - Data type: %d (0=float32, 1=float16, 2=int32, 3=uint8, 4=int64, 5=string, 6=bool, 7=int16, 8=complex64, 9=int8)\n", 
                     input_tensor->type);
        ns_lp_printf("  - Bytes: %d\n", input_tensor->bytes);
        ns_lp_printf("  - Dimensions: %d\n", input_tensor->dims->size);
        
        // Print shape information
        ns_lp_printf("  - Shape: [");
        for (int j = 0; j < input_tensor->dims->size; j++) {
            ns_lp_printf("%d", input_tensor->dims->data[j]);
            if (j < input_tensor->dims->size - 1) ns_lp_printf(", ");
        }
        ns_lp_printf("]\n");
        
        // Get quantization parameters if applicable
        if (input_tensor->quantization.type == kTfLiteAffineQuantization) {
            TfLiteAffineQuantization* quant = 
                (TfLiteAffineQuantization*)input_tensor->quantization.params;
            if (quant && quant->scale && quant->zero_point) {
                ns_lp_printf("  - Scale: %f\n", quant->scale->data[0]);
                ns_lp_printf("  - Zero point: %d\n", quant->zero_point->data[0]);
            }
        }

        // Initialize input tensors
        int offset = 0;
        for (uint32_t i = 0; i < model.numInputTensors; i++) {
            memcpy(
                model.model_input[i]->data.int8, ((char *)arrhythmia_model_power_example_input_tensors) + offset,
                model.model_input[i]->bytes);
            offset += model.model_input[i]->bytes;
        }
    }
    
    ns_lp_printf("Starting inference...\n");
    // Profile inference

    // Use the profiling function to get detailed profiling data
    profiling_result_t result = profile_model_inference(&model);
    
    // Get the invoke status from the profiling result
    TfLiteStatus invoke_status = result.invoke_status;
    uint32_t num_layers = result.num_layers;
    
    // Capture basic performance metrics
    uint32_t cycles = 0;
    #ifndef AM_PART_APOLLO5B
    cycles = ns_get_perf_cycles();
    #endif
    
    ns_lp_printf("Inference complete: cycles=%d, status=%d, layers=%d\n", cycles, invoke_status, num_layers);
    
    // Send enhanced stats over USB (cycles, status, num_layers, arena_used)
    uint8_t stats_packet[16]; // Increased size to include more data
    memcpy(&stats_packet[0], &cycles, 4);
    // Fix TfLiteStatus size issue by casting to uint32_t
    uint32_t status_value = (uint32_t)invoke_status;
    memcpy(&stats_packet[4], &status_value, 4);
    memcpy(&stats_packet[8], &num_layers, 4);
    memcpy(&stats_packet[12], &model.computed_arena_size, 4);
    
    ns_lp_printf("Sending enhanced stats packet: ");
    for (int i = 0; i < 16; i++) {
        ns_lp_printf("0x%02X ", stats_packet[i]);
    }
    ns_lp_printf("\n");
    ns_lp_printf("Stats: cycles=%d, status=%d, layers=%d, arena_used=%d\n", 
                 cycles, invoke_status, num_layers, model.computed_arena_size);
    
    webusb_send_data(stats_packet, 16);
    ns_lp_printf("Enhanced stats packet sent\n");
    ns_lp_printf("=== run_model_and_send_stats complete ===\n");
}

// Add PMU CSV transfer state

typedef struct {
    uint32_t total_chunks;
    uint32_t current_chunk;
    bool in_progress;
    uint32_t num_layers;
    uint32_t rv_count;
    uint32_t retry_count;
} pmu_csv_transfer_state_t;

static pmu_csv_transfer_state_t pmu_csv_state = {0};
#define PMU_CSV_MAX_RETRIES 5

void send_next_pmu_csv_chunk();

// void msgReceived(const uint8_t *buffer, uint32_t length, void *args) {
//     ns_lp_printf("=== msgReceived called ===\n");
//     ns_lp_printf("Received %d bytes\n", length);
//     if (length >= 13) {
//         const usb_message_header_t* header = (const usb_message_header_t*)buffer;
//         const uint8_t* payload = buffer + 13;
//         uint32_t payload_length = length - 13;
//         ns_lp_printf("Header: CRC32=0x%08X, cmd=%d, chunk=%u, total=%u\n", header->crc32, header->command, header->chunk_id, header->total_chunks);
//         if (header->command == CHUNK_CMD_MODEL_DATA) {
//             ns_lp_printf("Processing model chunk\n");
//             handle_model_chunk(buffer, length);
//             return;
//         } else if (header->command == CHUNK_CMD_CONFIG) {
//             ns_lp_printf("Processing configuration\n");
//             handle_model_config(payload, payload_length);
//             ns_lp_printf("Configuration received\n");
//             return;
//         } else if (header->command == CHUNK_CMD_RUN_STATS) {
//             ns_lp_printf("Processing RUN_STATS command\n");
//             run_model_and_send_stats();
//             return;
//         } else if (header->command == CHUNK_CMD_PMU_CSV && !pmu_csv_state.in_progress) {
//             ns_lp_printf("Starting PMU CSV transfer\n");
//             pmu_csv_state.total_chunks = g_num_layers + 1;
//             pmu_csv_state.current_chunk = 0;
//             pmu_csv_state.in_progress = true;
//             pmu_csv_state.num_layers = g_num_layers;
//             pmu_csv_state.rv_count = g_rv_count;
//             pmu_csv_state.retry_count = 0;
//             send_next_pmu_csv_chunk();
//             return;
//         } else if (header->command == CHUNK_CMD_ACK && pmu_csv_state.in_progress) {
//             uint32_t acked_chunk = header->chunk_id;
//             ns_lp_printf("Received ACK for chunk %u\n", acked_chunk);
//             if (acked_chunk == pmu_csv_state.current_chunk) {
//                 pmu_csv_state.current_chunk++;
//                 pmu_csv_state.retry_count = 0;
//                 if (pmu_csv_state.current_chunk < pmu_csv_state.total_chunks) {
//                     send_next_pmu_csv_chunk();
//                 } else {
//                     pmu_csv_state.in_progress = false;
//                     ns_lp_printf("PMU CSV transfer complete\n");
//                 }
//             } else {
//                 // Resend current chunk if out of order or lost
//                 if (++pmu_csv_state.retry_count < PMU_CSV_MAX_RETRIES) {
//                     ns_lp_printf("Resending chunk %u (retry %u)\n", pmu_csv_state.current_chunk, pmu_csv_state.retry_count);
//                     send_next_pmu_csv_chunk();
//                 } else {
//                     ns_lp_printf("PMU CSV transfer failed: too many retries\n");
//                     pmu_csv_state.in_progress = false;
//                 }
//             }
//             return;
//         } else {
//             ns_lp_printf("Unknown command: %d\n", header->command);
//         }
//     } else {
//         ns_lp_printf("Packet too short for header (need 13, got %d)\n", length);
//     }
//     ns_lp_printf("=== end msgReceived ===\n");
// }

void send_pmu_stats() {
    ns_lp_printf("=== send_pmu_stats called ===\n");
    ns_lp_printf("Sending PMU stats\n");

    for (int i = 0; i < g_num_layers; i++) {
        send_next_pmu_csv_chunk();
    }

    ns_lp_printf("=== end send_pmu_stats ===\n");
}


void send_next_pmu_csv_chunk(int chunk_id) {
    uint8_t packet[512];
    size_t packet_len = 0;

    if (chunk_id == -1) {
        ns_set_pmu_header();
        // Set ns_profiler_csv_header to the PMU header
        ns_lp_printf("ns_profiler_csv_header = %s\n", ns_profiler_csv_header);
    } else {
        ns_lp_printf("Sending PMU CSV row chunk %u\n", chunk_id);
    }

    uint32_t pmu_event_counters[NS_NUM_PMU_MAP_SIZE];

    ns_get_layer_counters(chunk_id, g_num_layers, g_rv_count, pmu_event_counters);

    int row_len = sizeof(pmu_event_counters);

    ns_lp_printf("row_len = %d\n", row_len);

    usb_message_header_t row_header = {
        .crc32 = CalcCrc32(0xFFFFFFFF, row_len, (uint8_t*)pmu_event_counters),
        .command = CHUNK_CMD_PMU_CSV,
        .chunk_id = chunk_id,
        .total_chunks = g_num_layers + 1
    };
    packet[0] = 0x00; packet[1] = 0x02;
    memcpy(packet + 2, &row_header, 13);

    // Add tag and elapsed_us
    memcpy(packet + 15, ns_profiler_events_stats[chunk_id].tag, 20);
    memcpy(packet + 35, &ns_profiler_events_stats[chunk_id].elapsed_us, 4);

    memcpy(packet + 39, pmu_event_counters, row_len);
    packet_len = 2 + 13 + 20 + 4 + row_len;
    ns_lp_printf("Sending PMU CSV row chunk %u, len=%d\n", chunk_id, (int)packet_len);

    // for (uint32_t map_index = 0; map_index < NS_NUM_PMU_MAP_SIZE; map_index++)
    // {
    //     ns_lp_printf("%d, ", pmu_event_counters[map_index]);
    // }

    webusb_send_data(packet, packet_len);
}

volatile int last_acknowledged_chunk = -1; // at file scope

// New function: send all PMU CSV chunks in a blocking loop, waiting for ACKs
void send_all_pmu_csv_chunks() {
    // pmu_csv_state.total_chunks = g_num_layers + 1;
    // pmu_csv_state.in_progress = true;
    // pmu_csv_state.retry_count = 0;
    // last_acknowledged_chunk = -1;
    // int max_retries = PMU_CSV_MAX_RETRIES;
    // for (uint32_t chunk = 0; chunk < pmu_csv_state.total_chunks; ) {
    //     int retries = 0;
    //     while (retries < max_retries) {
    //         send_next_pmu_csv_chunk(chunk);
    //         // Wait for ACK (polling with timeout)
    //         ns_delay_us(100000);
    //         if (last_acknowledged_chunk == (int)chunk) {
    //             ns_lp_printf("ACK received for chunk %u\n", chunk);
    //             // Got ACK, move to next chunk
    //             chunk++;
    //             break;
    //         } else {
    //             // Timeout, resend
    //             retries++;
    //             ns_lp_printf("Timeout waiting for ACK for chunk %u retry %d\n", chunk, retries);
    //         }
    //     }
    //     if (retries == max_retries) {
    //         ns_lp_printf("Failed to send chunk %u after %d retries. Current Chunk =%d Aborting.\n", chunk, max_retries);
    //         pmu_csv_state.in_progress = false;
    //         return;
    //     }
    // }
    // pmu_csv_state.in_progress = false;
    // ns_lp_printf("PMU CSV transfer complete (blocking loop)\n");

    // ns_lp_printf("g_num_layers = %d\n", g_num_layers);

    // for (uint32_t i = 0; i < g_num_layers+1; i++) {
    //     send_next_pmu_csv_chunk(i);
    //     ns_lp_printf("current chunk = %d\n", i);
    //     ns_lp_printf("last awknowledged chunk = %d\n", last_acknowledged_chunk);
    //     ns_delay_us(100000);
    // }
    send_next_pmu_csv_chunk(1);
    ns_delay_us(1000000);
    ns_lp_printf("last_acknowledged_chunk = %d\n", last_acknowledged_chunk);
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
            // return;
        } else if (header->command == CHUNK_CMD_CONFIG) {
            ns_lp_printf("Processing configuration\n");
            handle_model_config(payload, payload_length);
            ns_lp_printf("Configuration received\n");
            // return;
        } else if (header->command == CHUNK_CMD_RUN_STATS) {
            ns_lp_printf("Processing RUN_STATS command\n");
            run_model_and_send_stats();
            // return;
        } else if (header->command == CHUNK_CMD_PMU_CSV && !pmu_csv_state.in_progress) {
            ns_lp_printf("Starting PMU CSV transfer (blocking loop)\n");
            send_next_pmu_csv_chunk(0);
            // ns_delay_us(1000000);
            // return;
        } else if (header->command == CHUNK_CMD_ACK) {
            last_acknowledged_chunk = header->chunk_id;
            ns_lp_printf("Received ACK for chunk %u (set last_acknowledged_chunk)\n", header->chunk_id);

            if (last_acknowledged_chunk < g_num_layers) {
                send_next_pmu_csv_chunk(last_acknowledged_chunk + 1);
            }
            // ns_delay_us(1000000);
            // return;
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
    uint32_t num_layers = 0;
    char name[50];
    uint32_t pmu_profile_start_layer = 0;
    NS_TRY(ns_core_init(&ns_core_cfg), "Core init failed.\n");

    // ns_power_config(&ns_power_usb);

    NS_TRY(ns_power_config(&ns_power_measurement), "Power Init Failed.\n");

    NS_TRY(ns_set_performance_mode(NS_MAXIMUM_PERF), "Set CPU Perf mode failed.");

    ns_itm_printf_enable();
    ns_interrupt_master_enable();

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
        // example_status = NS_STATUS_SUCCESS;
    };
}
