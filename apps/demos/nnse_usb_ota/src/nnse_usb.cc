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

// Heap allocation for ns_malloc - similar to RPC examples
// Apollo510 has 3072KB SRAM, reserve some for system/stack
#define TOTAL_SRAM_SIZE_KB 3072
#define RESERVED_SRAM_SIZE_KB 512  // Reserve 512KB for system/stack/other
#define AVAILABLE_SRAM_SIZE_KB (TOTAL_SRAM_SIZE_KB - RESERVED_SRAM_SIZE_KB)
#define NS_MALLOC_HEAP_SIZE_IN_K 2048  // 2MB heap for dynamic allocation (needed for large models/arenas)

#if (configAPPLICATION_ALLOCATED_HEAP == 1)
// ns_malloc uses this heap for dynamic allocation
size_t ucHeapSize = NS_MALLOC_HEAP_SIZE_IN_K * 1024;
AM_SHARED_RW uint8_t ucHeap[NS_MALLOC_HEAP_SIZE_IN_K * 1024] __attribute__((aligned(4)));
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
#define CHUNK_CMD_DERIVED_ARRAYS 0x06
#define CHUNK_CMD_STRING_ARRAYS 0x07

// Memory allocation approach: Static for TCM, Dynamic for SRAM
#define TCM_MODEL_SIZE  (250 * 1024) // 250KB for TCM model
#define TCM_ARENA_SIZE  (100 * 1024) // 100KB for TCM arena

// Static TCM allocations (ns_malloc can't access TCM)
NS_PUT_IN_TCM alignas(16) static uint8_t tcm_model_array[TCM_MODEL_SIZE];
NS_PUT_IN_TCM alignas(16) static uint8_t tcm_arena[TCM_ARENA_SIZE];

// Dynamic SRAM allocations (using ns_malloc) - no static arrays needed
// These will be allocated dynamically when needed

typedef enum { MODEL_LOC_TCM = 0, MODEL_LOC_SRAM = 1 } model_location_t;
typedef enum { ARENA_LOC_TCM = 0, ARENA_LOC_SRAM = 1 } arena_location_t;
static model_location_t selected_model_location = MODEL_LOC_TCM;
static arena_location_t selected_arena_location = ARENA_LOC_TCM;

// Dynamic allocation state for SRAM
typedef struct {
    uint8_t* model_buffer;
    uint8_t* arena_buffer;
    uint32_t model_size;
    uint32_t arena_size;
    bool allocated;
} dynamic_allocation_state_t;

static dynamic_allocation_state_t sram_allocation = {0};

// Dynamic derived arrays storage (replaces hardcoded arrays)
typedef struct {
    uint32_t* mac_estimates;
    uint32_t* stride_h;
    uint32_t* stride_w;
    uint32_t* dilation_h;
    uint32_t* dilation_w;
    uint32_t* output_magnitudes;
    uint32_t* read_estimates;
    uint32_t* write_estimates;
    uint32_t* input_magnitudes;
    uint32_t num_operators;
    bool allocated;
} dynamic_derived_arrays_t;

typedef struct {
    char** mac_strings;
    char** output_shapes;
    char** filter_shapes;
    uint32_t num_operators;
    bool allocated;
} dynamic_string_arrays_t;

dynamic_derived_arrays_t g_derived_arrays = {0};
dynamic_string_arrays_t g_string_arrays = {0};

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

// Function to allocate model and arena dynamically for SRAM
bool allocate_sram_model_and_arena(uint32_t model_size) {
    ns_lp_printf("allocate_sram_model_and_arena: model_size=%d bytes\n", model_size);
    
    // Free any existing allocation
    if (sram_allocation.allocated) {
        ns_lp_printf("Freeing existing SRAM allocation\n");
        if (sram_allocation.model_buffer) {
            ns_free(sram_allocation.model_buffer);
            sram_allocation.model_buffer = NULL;
        }
        if (sram_allocation.arena_buffer) {
            ns_free(sram_allocation.arena_buffer);
            sram_allocation.arena_buffer = NULL;
        }
        sram_allocation.allocated = false;
    }
    
    // Check if model fits in available SRAM
    uint32_t available_sram_bytes = AVAILABLE_SRAM_SIZE_KB * 1024;
    if (model_size > available_sram_bytes) {
        ns_lp_printf("Error: Model too large for SRAM (%d > %d)\n", model_size, available_sram_bytes);
        return false;
    }
    
    // Allocate model buffer with fallback
    ns_lp_printf("Allocating model buffer: %d bytes\n", model_size);
    sram_allocation.model_buffer = (uint8_t*)ns_malloc(model_size);
    if (!sram_allocation.model_buffer) {
        ns_lp_printf("Error: Failed to allocate model buffer\n");
        return false;
    }
    ns_lp_printf("Model buffer allocated at: %p\n", sram_allocation.model_buffer);
    
    // Try different arena sizes with fallback
    uint32_t arena_sizes_to_try[] = {
        512 * 1024,    // 512KB
        256 * 1024,    // 256KB
        128 * 1024,    // 128KB
        64 * 1024,     // 64KB
        32 * 1024      // 32KB (minimum)
    };
    
    bool arena_allocated = false;
    for (int i = 0; i < sizeof(arena_sizes_to_try) / sizeof(arena_sizes_to_try[0]); i++) {
        uint32_t try_size = arena_sizes_to_try[i];
        ns_lp_printf("Trying arena size: %d bytes (%d KB)\n", try_size, try_size / 1024);
        
        sram_allocation.arena_buffer = (uint8_t*)ns_malloc(try_size);
        if (sram_allocation.arena_buffer) {
            sram_allocation.arena_size = try_size;
            arena_allocated = true;
            ns_lp_printf("Arena buffer allocated at: %p with size %d bytes\n", 
                        sram_allocation.arena_buffer, try_size);
            break;
        } else {
            ns_lp_printf("Failed to allocate arena with size %d bytes\n", try_size);
        }
    }
    
    if (!arena_allocated) {
        ns_lp_printf("Error: Failed to allocate arena buffer with any size\n");
        ns_free(sram_allocation.model_buffer);
        sram_allocation.model_buffer = NULL;
        return false;
    }
    
    sram_allocation.model_size = model_size;
    sram_allocation.allocated = true;
    
    ns_lp_printf("Successfully allocated in SRAM: model=%d bytes, arena=%d bytes\n", 
                 model_size, sram_allocation.arena_size);
    
    return true;
}

// Function to allocate only arena in SRAM (when model is in TCM)
bool allocate_sram_arena_only(uint32_t arena_size) {
    ns_lp_printf("allocate_sram_arena_only: arena_size=%d bytes\n", arena_size);
    
    // Free any existing arena allocation
    if (sram_allocation.arena_buffer) {
        ns_free(sram_allocation.arena_buffer);
        sram_allocation.arena_buffer = NULL;
    }
    
    // Try different arena sizes with fallback (more conservative)
    uint32_t arena_sizes_to_try[] = {
        512 * 1024,    // 512KB
        256 * 1024,    // 256KB
        128 * 1024,    // 128KB
        64 * 1024,     // 64KB
        32 * 1024      // 32KB (minimum)
    };
    
    bool arena_allocated = false;
    for (int i = 0; i < sizeof(arena_sizes_to_try) / sizeof(arena_sizes_to_try[0]); i++) {
        uint32_t try_size = arena_sizes_to_try[i];
        ns_lp_printf("Trying arena size: %d bytes (%d KB)\n", try_size, try_size / 1024);
        
        sram_allocation.arena_buffer = (uint8_t*)ns_malloc(try_size);
        if (sram_allocation.arena_buffer) {
            sram_allocation.arena_size = try_size;
            arena_allocated = true;
            sram_allocation.allocated = true; // Mark as allocated even if model_buffer is NULL
            ns_lp_printf("Arena buffer allocated at: %p with size %d bytes\n", 
                        sram_allocation.arena_buffer, try_size);
            break;
        } else {
            ns_lp_printf("Failed to allocate arena with size %d bytes\n", try_size);
        }
    }
    
    if (!arena_allocated) {
        ns_lp_printf("Error: Failed to allocate arena buffer with any size\n");
        return false;
    }
    
    ns_lp_printf("Successfully allocated arena in SRAM: %d bytes\n", sram_allocation.arena_size);
    
    return true;
}

// Function to free derived arrays
void free_derived_arrays() {
    if (g_derived_arrays.allocated) {
        if (g_derived_arrays.mac_estimates) {
            ns_free(g_derived_arrays.mac_estimates);
            g_derived_arrays.mac_estimates = NULL;
        }
        if (g_derived_arrays.stride_h) {
            ns_free(g_derived_arrays.stride_h);
            g_derived_arrays.stride_h = NULL;
        }
        if (g_derived_arrays.stride_w) {
            ns_free(g_derived_arrays.stride_w);
            g_derived_arrays.stride_w = NULL;
        }
        if (g_derived_arrays.dilation_h) {
            ns_free(g_derived_arrays.dilation_h);
            g_derived_arrays.dilation_h = NULL;
        }
        if (g_derived_arrays.dilation_w) {
            ns_free(g_derived_arrays.dilation_w);
            g_derived_arrays.dilation_w = NULL;
        }
        if (g_derived_arrays.output_magnitudes) {
            ns_free(g_derived_arrays.output_magnitudes);
            g_derived_arrays.output_magnitudes = NULL;
        }
        if (g_derived_arrays.read_estimates) {
            ns_free(g_derived_arrays.read_estimates);
            g_derived_arrays.read_estimates = NULL;
        }
        if (g_derived_arrays.write_estimates) {
            ns_free(g_derived_arrays.write_estimates);
            g_derived_arrays.write_estimates = NULL;
        }
        if (g_derived_arrays.input_magnitudes) {
            ns_free(g_derived_arrays.input_magnitudes);
            g_derived_arrays.input_magnitudes = NULL;
        }
        g_derived_arrays.allocated = false;
        g_derived_arrays.num_operators = 0;
        ns_lp_printf("Freed derived arrays\n");
    }
}

// Function to free SRAM allocations
void free_sram_allocation() {
    if (sram_allocation.allocated) {
        if (sram_allocation.model_buffer) {
            ns_free(sram_allocation.model_buffer);
            sram_allocation.model_buffer = NULL;
        }
        if (sram_allocation.arena_buffer) {
            ns_free(sram_allocation.arena_buffer);
            sram_allocation.arena_buffer = NULL;
        }
        sram_allocation.allocated = false;
        ns_lp_printf("Freed SRAM allocations\n");
    }
}

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
    ns_lp_printf("sendAck: Sending ACK bytes: 0x%02X %08X\n", ack[0], chunk_id);
    webusb_send_data(ack, 5);
    ns_lp_printf("sendAck: ACK sent successfully\n");
}

// Handle derived arrays message (chunked)
void handle_derived_arrays(const uint8_t* data, uint32_t length) {
    if (length < 21) { // 8 bytes for metadata + at least 8 bytes for one operator
        ns_lp_printf("Error: Derived arrays message too short\n");
        return;
    }
    
    // Parse header
    const usb_message_header_t* header = (const usb_message_header_t*)data;
    const uint8_t* payload = data + 13;
    uint32_t payload_length = length - 13;
    
    ns_lp_printf("Received derived arrays chunk %u/%u: payload_length=%u\n", 
                 header->chunk_id, header->total_chunks, payload_length);
    
    // Parse chunk metadata
    uint32_t total_operators = *(uint32_t*)payload;
    uint32_t operators_in_chunk = *(uint32_t*)(payload + 4);
    
    // Calculate chunk start index based on payload size (same as JavaScript)
    const uint32_t CHUNK_SIZE = 480;
    const uint32_t METADATA_SIZE = 8;
    const uint32_t OPERATOR_DATA_SIZE = 36; // Updated to match new payload size
    const uint32_t MAX_OPERATORS_PER_CHUNK = (CHUNK_SIZE - METADATA_SIZE) / OPERATOR_DATA_SIZE;
    uint32_t chunk_start_idx = header->chunk_id * MAX_OPERATORS_PER_CHUNK;
    
    ns_lp_printf("Chunk metadata: total_ops=%u, chunk_ops=%u, start_idx=%u (max_ops_per_chunk=%u)\n", 
                 total_operators, operators_in_chunk, chunk_start_idx, MAX_OPERATORS_PER_CHUNK);
    ns_lp_printf("Chunking: CHUNK_SIZE=%u, METADATA_SIZE=%u, OPERATOR_DATA_SIZE=%u, MAX_OPERATORS_PER_CHUNK=%u\n",
                 CHUNK_SIZE, METADATA_SIZE, OPERATOR_DATA_SIZE, MAX_OPERATORS_PER_CHUNK);
    
    // Calculate expected payload size: 8 bytes for metadata + 36 bytes per operator (9 values * 4 bytes each)
    uint32_t expected_size = 8 + operators_in_chunk * 36;
    if (payload_length < expected_size) {
        ns_lp_printf("Error: Payload too small for %u operators in chunk (need %u, got %u)\n", 
                     operators_in_chunk, expected_size, payload_length);
        return;
    }
    
    // Allocate arrays on first chunk (chunk 0)
    if (header->chunk_id == 0) {
        // Free any existing allocation
        if (g_derived_arrays.allocated) {
            ns_lp_printf("Freeing existing derived arrays allocation\n");
            if (g_derived_arrays.mac_estimates) {
                ns_free(g_derived_arrays.mac_estimates);
                g_derived_arrays.mac_estimates = NULL;
            }
            if (g_derived_arrays.stride_h) {
                ns_free(g_derived_arrays.stride_h);
                g_derived_arrays.stride_h = NULL;
            }
            if (g_derived_arrays.stride_w) {
                ns_free(g_derived_arrays.stride_w);
                g_derived_arrays.stride_w = NULL;
            }
            if (g_derived_arrays.dilation_h) {
                ns_free(g_derived_arrays.dilation_h);
                g_derived_arrays.dilation_h = NULL;
            }
            if (g_derived_arrays.dilation_w) {
                ns_free(g_derived_arrays.dilation_w);
                g_derived_arrays.dilation_w = NULL;
            }
            if (g_derived_arrays.output_magnitudes) {
                ns_free(g_derived_arrays.output_magnitudes);
                g_derived_arrays.output_magnitudes = NULL;
            }
            if (g_derived_arrays.read_estimates) {
                ns_free(g_derived_arrays.read_estimates);
                g_derived_arrays.read_estimates = NULL;
            }
            if (g_derived_arrays.write_estimates) {
                ns_free(g_derived_arrays.write_estimates);
                g_derived_arrays.write_estimates = NULL;
            }
            if (g_derived_arrays.input_magnitudes) {
                ns_free(g_derived_arrays.input_magnitudes);
                g_derived_arrays.input_magnitudes = NULL;
            }
            g_derived_arrays.allocated = false;
        }
        
        // Allocate arrays for total number of operators
        g_derived_arrays.mac_estimates = (uint32_t*)ns_malloc(total_operators * sizeof(uint32_t));
        g_derived_arrays.stride_h = (uint32_t*)ns_malloc(total_operators * sizeof(uint32_t));
        g_derived_arrays.stride_w = (uint32_t*)ns_malloc(total_operators * sizeof(uint32_t));
        g_derived_arrays.dilation_h = (uint32_t*)ns_malloc(total_operators * sizeof(uint32_t));
        g_derived_arrays.dilation_w = (uint32_t*)ns_malloc(total_operators * sizeof(uint32_t));
        g_derived_arrays.output_magnitudes = (uint32_t*)ns_malloc(total_operators * sizeof(uint32_t));
        g_derived_arrays.read_estimates = (uint32_t*)ns_malloc(total_operators * sizeof(uint32_t));
        g_derived_arrays.write_estimates = (uint32_t*)ns_malloc(total_operators * sizeof(uint32_t));
        g_derived_arrays.input_magnitudes = (uint32_t*)ns_malloc(total_operators * sizeof(uint32_t));
    
        // Check if all allocations succeeded
        if (!g_derived_arrays.mac_estimates || !g_derived_arrays.stride_h || 
            !g_derived_arrays.stride_w || !g_derived_arrays.dilation_h || 
            !g_derived_arrays.dilation_w || !g_derived_arrays.output_magnitudes ||
            !g_derived_arrays.read_estimates || !g_derived_arrays.write_estimates ||
            !g_derived_arrays.input_magnitudes) {
            ns_lp_printf("Error: Failed to allocate derived arrays\n");
            // Free any successful allocations
            if (g_derived_arrays.mac_estimates) ns_free(g_derived_arrays.mac_estimates);
            if (g_derived_arrays.stride_h) ns_free(g_derived_arrays.stride_h);
            if (g_derived_arrays.stride_w) ns_free(g_derived_arrays.stride_w);
            if (g_derived_arrays.dilation_h) ns_free(g_derived_arrays.dilation_h);
            if (g_derived_arrays.dilation_w) ns_free(g_derived_arrays.dilation_w);
            if (g_derived_arrays.output_magnitudes) ns_free(g_derived_arrays.output_magnitudes);
            if (g_derived_arrays.read_estimates) ns_free(g_derived_arrays.read_estimates);
            if (g_derived_arrays.write_estimates) ns_free(g_derived_arrays.write_estimates);
            if (g_derived_arrays.input_magnitudes) ns_free(g_derived_arrays.input_magnitudes);
            return;
        }
        
        g_derived_arrays.num_operators = total_operators;
        g_derived_arrays.allocated = true;
        ns_lp_printf("Allocated derived arrays for %u total operators\n", total_operators);
    }
    
    // Parse chunk data into the correct positions
    ns_lp_printf("Parsing chunk data for %u operators\n", operators_in_chunk);
    
    // Check if arrays are properly allocated
    if (!g_derived_arrays.allocated || !g_derived_arrays.mac_estimates || !g_derived_arrays.stride_h) {
        ns_lp_printf("Error: Derived arrays not properly allocated\n");
        return;
    }
    
    uint32_t offset = 8; // Skip metadata (total_ops + chunk_ops)
    
    // Add bounds checking to prevent crashes
    if (offset + operators_in_chunk * 36 > payload_length) {
        ns_lp_printf("Error: Payload too small for parsing %u operators\n", operators_in_chunk);
        return;
    }
    
    for (uint32_t i = 0; i < operators_in_chunk; i++) {
        uint32_t op_idx = chunk_start_idx + i;
        
        // Bounds check for array access
        if (op_idx >= total_operators) {
            ns_lp_printf("Error: Operator index %u out of bounds (total: %u)\n", op_idx, total_operators);
            return;
        }
        
        // Bounds check for payload access
        if (offset + 36 > payload_length) {
            ns_lp_printf("Error: Payload access out of bounds at offset %u\n", offset);
            return;
        }
        
        g_derived_arrays.mac_estimates[op_idx] = *(uint32_t*)(payload + offset);
        g_derived_arrays.stride_h[op_idx] = *(uint32_t*)(payload + offset + 4);
        g_derived_arrays.stride_w[op_idx] = *(uint32_t*)(payload + offset + 8);
        g_derived_arrays.dilation_h[op_idx] = *(uint32_t*)(payload + offset + 12);
        g_derived_arrays.dilation_w[op_idx] = *(uint32_t*)(payload + offset + 16);
        g_derived_arrays.output_magnitudes[op_idx] = *(uint32_t*)(payload + offset + 20);
        g_derived_arrays.read_estimates[op_idx] = *(uint32_t*)(payload + offset + 24);
        g_derived_arrays.write_estimates[op_idx] = *(uint32_t*)(payload + offset + 28);
        g_derived_arrays.input_magnitudes[op_idx] = *(uint32_t*)(payload + offset + 32);
        offset += 36;
        
        // Debug: print first few values
        if (i < 3) {
            ns_lp_printf("  Op %u: MAC=%u, StrideH=%u, StrideW=%u, DilationH=%u, DilationW=%u\n", op_idx, 
                        g_derived_arrays.mac_estimates[op_idx], 
                        g_derived_arrays.stride_h[op_idx],
                        g_derived_arrays.stride_w[op_idx],
                        g_derived_arrays.dilation_h[op_idx],
                        g_derived_arrays.dilation_w[op_idx]);
        }
    }
    ns_lp_printf("Finished parsing chunk data\n");
    
    // All values are now received from the payload, no defaults needed
    
    // Send ACK for this chunk
    ns_lp_printf("About to send ACK for derived arrays chunk %u\n", header->chunk_id);
    sendAck(header->chunk_id);
    ns_lp_printf("ACK sent for derived arrays chunk %u\n", header->chunk_id);
    
    // Check if this is the last chunk
    if (header->chunk_id == header->total_chunks - 1) {
        ns_lp_printf("Successfully received all derived arrays chunks for %u operators\n", total_operators);
        ns_lp_printf("First few MAC estimates: %u, %u, %u\n", 
                     g_derived_arrays.mac_estimates[0], 
                     g_derived_arrays.mac_estimates[1], 
                     g_derived_arrays.mac_estimates[2]);
        
        // Print a summary of the derived arrays for TFLM profiling
        ns_lp_printf("=== DERIVED ARRAYS SUMMARY ===\n");
        ns_lp_printf("Total operators: %u\n", g_derived_arrays.num_operators);
        ns_lp_printf("Arrays allocated: %s\n", g_derived_arrays.allocated ? "YES" : "NO");
        ns_lp_printf("MAC estimates array: %p\n", g_derived_arrays.mac_estimates);
        ns_lp_printf("Stride H array: %p\n", g_derived_arrays.stride_h);
        ns_lp_printf("Stride W array: %p\n", g_derived_arrays.stride_w);
        ns_lp_printf("Dilation H array: %p\n", g_derived_arrays.dilation_h);
        ns_lp_printf("Dilation W array: %p\n", g_derived_arrays.dilation_w);
        ns_lp_printf("Output magnitudes array: %p\n", g_derived_arrays.output_magnitudes);
        ns_lp_printf("Read estimates array: %p\n", g_derived_arrays.read_estimates);
        ns_lp_printf("Write estimates array: %p\n", g_derived_arrays.write_estimates);
        ns_lp_printf("Input magnitudes array: %p\n", g_derived_arrays.input_magnitudes);
        
        // Print first 10 MAC estimates
        ns_lp_printf("First 10 MAC estimates: ");
        for (uint32_t i = 0; i < 10 && i < g_derived_arrays.num_operators; i++) {
            ns_lp_printf("%u ", g_derived_arrays.mac_estimates[i]);
        }
        ns_lp_printf("\n");
        
        // Print first 10 Stride H values
        ns_lp_printf("First 10 Stride H values: ");
        for (uint32_t i = 0; i < 10 && i < g_derived_arrays.num_operators; i++) {
            ns_lp_printf("%u ", g_derived_arrays.stride_h[i]);
        }
        ns_lp_printf("\n");
        
        // Print last 5 MAC estimates
        if (g_derived_arrays.num_operators > 5) {
            ns_lp_printf("Last 5 MAC estimates: ");
            for (uint32_t i = g_derived_arrays.num_operators - 5; i < g_derived_arrays.num_operators; i++) {
                ns_lp_printf("%u ", g_derived_arrays.mac_estimates[i]);
            }
            ns_lp_printf("\n");
        }
        
        ns_lp_printf("=== END DERIVED ARRAYS SUMMARY ===\n");
    }
}

// Handle string arrays message (chunked)
void handle_string_arrays(const uint8_t* data, uint32_t length) {
    if (length < 21) { // 13 bytes for header + 8 bytes for metadata
        ns_lp_printf("Error: String arrays message too short\n");
        return;
    }
    
    // Parse header
    const usb_message_header_t* header = (const usb_message_header_t*)data;
    const uint8_t* payload = data + 13;
    uint32_t payload_length = length - 13;
    
    ns_lp_printf("Received string arrays chunk %u/%u: payload_length=%u\n", 
                 header->chunk_id, header->total_chunks, payload_length);
    
    // Parse chunk metadata
    uint32_t total_operators = *(uint32_t*)payload;
    uint32_t operators_in_chunk = *(uint32_t*)(payload + 4);
    
    ns_lp_printf("String arrays chunk metadata: total_ops=%u, chunk_ops=%u\n", 
                 total_operators, operators_in_chunk);
    
    // Calculate chunk start index (same logic as derived arrays)
    const uint32_t CHUNK_SIZE = 480;
    const uint32_t METADATA_SIZE = 8;
    // Use the same fixed chunking as JavaScript
    const uint32_t MAX_OPERATORS_PER_CHUNK = 8; // Fixed conservative value
    uint32_t chunk_start_idx = header->chunk_id * MAX_OPERATORS_PER_CHUNK;
    
    ns_lp_printf("String arrays chunking: start_idx=%u, max_ops_per_chunk=%u\n", 
                 chunk_start_idx, MAX_OPERATORS_PER_CHUNK);
    
    // Allocate arrays on first chunk (chunk 0)
    if (header->chunk_id == 0) {
        // Free any existing string arrays
        if (g_string_arrays.allocated) {
            for (uint32_t i = 0; i < g_string_arrays.num_operators; i++) {
                if (g_string_arrays.mac_strings[i]) ns_free(g_string_arrays.mac_strings[i]);
                if (g_string_arrays.output_shapes[i]) ns_free(g_string_arrays.output_shapes[i]);
                if (g_string_arrays.filter_shapes[i]) ns_free(g_string_arrays.filter_shapes[i]);
            }
            if (g_string_arrays.mac_strings) ns_free(g_string_arrays.mac_strings);
            if (g_string_arrays.output_shapes) ns_free(g_string_arrays.output_shapes);
            if (g_string_arrays.filter_shapes) ns_free(g_string_arrays.filter_shapes);
            g_string_arrays.allocated = false;
        }
        
        // Allocate string arrays for total number of operators
        g_string_arrays.mac_strings = (char**)ns_malloc(total_operators * sizeof(char*));
        g_string_arrays.output_shapes = (char**)ns_malloc(total_operators * sizeof(char*));
        g_string_arrays.filter_shapes = (char**)ns_malloc(total_operators * sizeof(char*));
        
        if (!g_string_arrays.mac_strings || !g_string_arrays.output_shapes || !g_string_arrays.filter_shapes) {
            ns_lp_printf("Error: Failed to allocate string arrays\n");
            return;
        }
        
        // Initialize pointers
        for (uint32_t i = 0; i < total_operators; i++) {
            g_string_arrays.mac_strings[i] = NULL;
            g_string_arrays.output_shapes[i] = NULL;
            g_string_arrays.filter_shapes[i] = NULL;
        }
        
        g_string_arrays.num_operators = total_operators;
        g_string_arrays.allocated = true;
        ns_lp_printf("Allocated string arrays for %u total operators\n", total_operators);
    }
    
    // Check if arrays are properly allocated
    if (!g_string_arrays.allocated || !g_string_arrays.mac_strings || !g_string_arrays.output_shapes || !g_string_arrays.filter_shapes) {
        ns_lp_printf("Error: String arrays not properly allocated\n");
        return;
    }
    
    // Parse strings from payload
    uint32_t offset = 8; // Skip metadata (total_ops + chunk_ops)
    
    ns_lp_printf("Parsing string arrays chunk data for %u operators\n", operators_in_chunk);
    
    for (uint32_t i = 0; i < operators_in_chunk; i++) {
        uint32_t op_idx = chunk_start_idx + i;
        
        // Bounds check for array access
        if (op_idx >= total_operators) {
            ns_lp_printf("Error: String array operator index %u out of bounds (total: %u)\n", op_idx, total_operators);
            return;
        }
        // Parse MAC string
        if (offset + 4 > payload_length) {
            ns_lp_printf("Error: Payload too small for MAC string %u\n", i);
            return;
        }
        uint32_t mac_len = *(uint32_t*)(payload + offset);
        offset += 4;
        
        if (offset + mac_len > payload_length) {
            ns_lp_printf("Error: MAC string %u out of bounds\n", i);
            return;
        }
        
        g_string_arrays.mac_strings[op_idx] = (char*)ns_malloc(mac_len + 1);
        if (g_string_arrays.mac_strings[op_idx]) {
            memcpy(g_string_arrays.mac_strings[op_idx], payload + offset, mac_len);
            g_string_arrays.mac_strings[op_idx][mac_len] = '\0';
        }
        offset += mac_len;
        
        // Parse output shape string
        if (offset + 4 > payload_length) {
            ns_lp_printf("Error: Payload too small for output shape string %u\n", i);
            return;
        }
        uint32_t output_len = *(uint32_t*)(payload + offset);
        offset += 4;
        
        if (offset + output_len > payload_length) {
            ns_lp_printf("Error: Output shape string %u out of bounds\n", i);
            return;
        }
        
        g_string_arrays.output_shapes[op_idx] = (char*)ns_malloc(output_len + 1);
        if (g_string_arrays.output_shapes[op_idx]) {
            memcpy(g_string_arrays.output_shapes[op_idx], payload + offset, output_len);
            g_string_arrays.output_shapes[op_idx][output_len] = '\0';
        }
        offset += output_len;
        
        // Parse filter shape string
        if (offset + 4 > payload_length) {
            ns_lp_printf("Error: Payload too small for filter shape string %u\n", i);
            return;
        }
        uint32_t filter_len = *(uint32_t*)(payload + offset);
        offset += 4;
        
        if (offset + filter_len > payload_length) {
            ns_lp_printf("Error: Filter shape string %u out of bounds\n", i);
            return;
        }
        
        g_string_arrays.filter_shapes[op_idx] = (char*)ns_malloc(filter_len + 1);
        if (g_string_arrays.filter_shapes[op_idx]) {
            memcpy(g_string_arrays.filter_shapes[op_idx], payload + offset, filter_len);
            g_string_arrays.filter_shapes[op_idx][filter_len] = '\0';
        }
        offset += filter_len;
        
        // Debug: print first few strings
        if (i < 3) {
            ns_lp_printf("  Op %u: MAC='%s', Output='%s', Filter='%s'\n", op_idx,
                        g_string_arrays.mac_strings[op_idx] ? g_string_arrays.mac_strings[op_idx] : "NULL",
                        g_string_arrays.output_shapes[op_idx] ? g_string_arrays.output_shapes[op_idx] : "NULL",
                        g_string_arrays.filter_shapes[op_idx] ? g_string_arrays.filter_shapes[op_idx] : "NULL");
        }
    }
    
    ns_lp_printf("Finished parsing string arrays chunk data\n");
    
    // Send ACK for this chunk
    ns_lp_printf("About to send ACK for string arrays chunk %u\n", header->chunk_id);
    sendAck(header->chunk_id);
    ns_lp_printf("ACK sent for string arrays chunk %u\n", header->chunk_id);
    
    // Check if this is the last chunk
    if (header->chunk_id == header->total_chunks - 1) {
        ns_lp_printf("Successfully received all string arrays chunks for %u operators\n", total_operators);
        
        // Print a summary of the string arrays for TFLM profiling
        ns_lp_printf("=== STRING ARRAYS SUMMARY ===\n");
        ns_lp_printf("Total operators: %u\n", g_string_arrays.num_operators);
        ns_lp_printf("Arrays allocated: %s\n", g_string_arrays.allocated ? "YES" : "NO");
        ns_lp_printf("MAC strings array: %p\n", g_string_arrays.mac_strings);
        ns_lp_printf("Output shapes array: %p\n", g_string_arrays.output_shapes);
        ns_lp_printf("Filter shapes array: %p\n", g_string_arrays.filter_shapes);
        
        // Print first few strings
        ns_lp_printf("First 3 MAC strings: ");
        for (uint32_t i = 0; i < 3 && i < g_string_arrays.num_operators; i++) {
            ns_lp_printf("'%s' ", g_string_arrays.mac_strings[i] ? g_string_arrays.mac_strings[i] : "NULL");
        }
        ns_lp_printf("\n");
        
        ns_lp_printf("=== END STRING ARRAYS SUMMARY ===\n");
    }
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
            // Free any existing derived arrays when starting a new model upload
            free_derived_arrays();
            model_state.total_chunks = header->total_chunks;
            model_state.received_chunks = 0;
            model_state.upload_in_progress = true;
            model_state.upload_complete = false;
            ns_lp_printf("Starting model upload: %d chunks\n", model_state.total_chunks);
            uint32_t estimated_size = header->total_chunks * payload_length;
            ns_lp_printf("Estimated model size: %d bytes\n", estimated_size);
            
            if (selected_model_location == MODEL_LOC_SRAM) {
                // Dynamic allocation for SRAM
                ns_lp_printf("Allocating model and arena in SRAM dynamically\n");
                if (!allocate_sram_model_and_arena(estimated_size)) {
                    ns_lp_printf("Error: Failed to allocate memory for model in SRAM\n");
                    model_state.upload_in_progress = false;
                    return;
                }
                model_state.model_buffer = sram_allocation.model_buffer;
            } else {
                // Static allocation for TCM
                ns_lp_printf("Using static TCM allocation\n");
                if (estimated_size > TCM_MODEL_SIZE) {
                    ns_lp_printf("Error: Model too large for TCM (%d > %d)\n", estimated_size, TCM_MODEL_SIZE);
                    model_state.upload_in_progress = false;
                    return;
                }
                model_state.model_buffer = tcm_model_array;
            }
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
// #include "arrhythmia_model_power_model_data.h"

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
    
    // Select model buffer and arena based on allocation type
    uint8_t* model_data;
    uint8_t* arena;
    uint32_t arena_size;
    
    if (selected_model_location == MODEL_LOC_SRAM) {
        // Use dynamically allocated SRAM
        if (!sram_allocation.allocated) {
            ns_lp_printf("Error: No SRAM allocation found\n");
            uint8_t error_packet[8] = {0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF};
            webusb_send_data(error_packet, 8);
            return;
        }
        model_data = sram_allocation.model_buffer;
        
        if (selected_arena_location == ARENA_LOC_SRAM) {
            arena = sram_allocation.arena_buffer;
            arena_size = sram_allocation.arena_size;
        } else {
            arena = tcm_arena;
            arena_size = TCM_ARENA_SIZE;
        }
    } else {
        // Use static TCM allocation
        model_data = tcm_model_array;
        
        if (selected_arena_location == ARENA_LOC_SRAM) {
            // Need to allocate arena in SRAM
            if (!sram_allocation.allocated || !sram_allocation.arena_buffer) {
                ns_lp_printf("Allocating arena in SRAM for TCM model\n");
                // Let the function try different sizes automatically
                if (!allocate_sram_arena_only(0)) { // Pass 0 to let function choose size
                    ns_lp_printf("Error: Failed to allocate arena in SRAM\n");
                    uint8_t error_packet[8] = {0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF};
                    webusb_send_data(error_packet, 8);
                    return;
                }
            }
            arena = sram_allocation.arena_buffer;
            arena_size = sram_allocation.arena_size;
        } else {
            arena = tcm_arena;
            arena_size = TCM_ARENA_SIZE;
        }
    }
    
    ns_lp_printf("Using model location: %s\n", selected_model_location == MODEL_LOC_SRAM ? "SRAM" : "TCM");
    ns_lp_printf("Using arena location: %s\n", selected_arena_location == ARENA_LOC_SRAM ? "SRAM" : "TCM");
    ns_lp_printf("Model data pointer: %p, size: %d\n", model_data, model_state.model_size);
    ns_lp_printf("Arena pointer: %p, size: %d\n", arena, arena_size);

    // Fill out model state struct
    memset(&model, 0, sizeof(model));
    model.model_array = model_data;
    model.arena = arena;
    model.arena_size = arena_size;
    
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

void send_next_pmu_csv_chunk(int chunk_id) {
    uint8_t packet[512];
    size_t packet_len = 0;

    uint32_t pmu_event_counters[NS_NUM_PMU_MAP_SIZE];

    ns_get_layer_counters(chunk_id, g_num_layers, g_rv_count, pmu_event_counters);

    int row_len = sizeof(pmu_event_counters);

    ns_lp_printf("row_len = %d\n", row_len);

    usb_message_header_t row_header = {
        .crc32 = CalcCrc32(0xFFFFFFFF, row_len, (uint8_t*)pmu_event_counters),
        .command = CHUNK_CMD_PMU_CSV,
        .chunk_id = chunk_id,
        .total_chunks = g_num_layers
    };
    packet[0] = 0x00; packet[1] = 0x02;
    memcpy(packet + 2, &row_header, 13);

    // Add tag and elapsed_us
    memcpy(packet + 15, ns_profiler_events_stats[chunk_id].tag, 20);
    memcpy(packet + 35, &ns_profiler_events_stats[chunk_id].elapsed_us, 4);

    memcpy(packet + 39, pmu_event_counters, row_len);
    packet_len = 2 + 13 + 20 + 4 + row_len;
    ns_lp_printf("Sending PMU CSV row chunk %u, len=%d\n", chunk_id, (int)packet_len);

    webusb_send_data(packet, packet_len);
}

volatile int last_acknowledged_chunk = -1; // at file scope

void msgReceived(const uint8_t *buffer, uint32_t length, void *args) {
    ns_lp_printf("=== msgReceived called ===\n");
    ns_lp_printf("Received %d bytes\n", length);
    
    // Check for frame header (0x00 0x02) and skip it if present
    const uint8_t* data = buffer;
    uint32_t data_length = length;
    
    if (length >= 15 && buffer[0] == 0x00 && buffer[1] == 0x02) {
        // Frame header found, skip it
        data = buffer + 2;
        data_length = length - 2;
        ns_lp_printf("Frame header found, processing %d bytes of data\n", data_length);
    }
    
    if (data_length >= 13) {
        const usb_message_header_t* header = (const usb_message_header_t*)data;
        const uint8_t* payload = data + 13;
        uint32_t payload_length = data_length - 13;
        ns_lp_printf("Header: CRC32=0x%08X, cmd=%d, chunk=%u, total=%u\n", header->crc32, header->command, header->chunk_id, header->total_chunks);
        if (header->command == CHUNK_CMD_MODEL_DATA) {
            ns_lp_printf("Processing model chunk\n");
            handle_model_chunk(data, data_length);
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
        } else if (header->command == CHUNK_CMD_DERIVED_ARRAYS) {
            ns_lp_printf("Processing derived arrays\n");
            handle_derived_arrays(data, data_length);
            ns_lp_printf("Derived arrays received\n");
            // return;
        } else if (header->command == CHUNK_CMD_STRING_ARRAYS) {
            ns_lp_printf("Processing string arrays\n");
            handle_string_arrays(data, data_length);
            ns_lp_printf("String arrays received\n");
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
                send_next_pmu_csv_chunk(last_acknowledged_chunk+1);
            }
            // ns_delay_us(1000000);
            // return;
        } else {
            ns_lp_printf("Unknown command: %d\n", header->command);
        }
    } else {
        ns_lp_printf("Packet too short for header (need 13, got %d)\n", data_length);
    }
    ns_lp_printf("=== end msgReceived ===\n");
}

int main(void) {
    usb_handle_t usb_handle = NULL;

    ns_core_config_t ns_core_cfg = {.api = &ns_core_V1_0_0};
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

    // Initialize dynamic allocation state
    memset(&sram_allocation, 0, sizeof(sram_allocation));

    ns_lp_printf("Model upload firmware ready with hybrid allocation\n");
    ns_lp_printf("TCM: Static allocation - %d KB model, %d KB arena\n", 
                 TCM_MODEL_SIZE / 1024, TCM_ARENA_SIZE / 1024);
    ns_lp_printf("SRAM: Dynamic allocation - up to %d KB available\n", AVAILABLE_SRAM_SIZE_KB);
    ns_lp_printf("Heap size: %d KB for ns_malloc\n", NS_MALLOC_HEAP_SIZE_IN_K);
    
    // Send a simple heartbeat message to verify USB is working
    uint8_t heartbeat[] = {0x48, 0x45, 0x4C, 0x4C, 0x4F}; // "HELLO"
    webusb_send_data(heartbeat, 5);
    ns_lp_printf("Sent heartbeat message\n");

    vTaskStartScheduler();
    while (1) {
        // example_status = NS_STATUS_SUCCESS;
    };
}
