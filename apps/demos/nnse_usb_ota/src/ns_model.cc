/**
 * @file generic_model.h
 * @author Carlos Morales (carlos.morales@ambiq.com)
 * @brief Generic TF Model wrapper
 * @version 0.1
 * @date 2023-3-08
 *
 * @copyright Copyright (c) 2023
 *
 */

// NS includes
#include "ns_model.h"
#include "ns_ambiqsuite_harness.h"
#include "ns_debug_log.h"
#include "supported_resolver.h"

// Tensorflow Lite for Microcontroller includes (somewhat boilerplate)
// #include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#ifdef NS_TFSTRUCTURE_RECENT
    #include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#else
    #include "tensorflow/lite/micro/micro_error_reporter.h"
#endif

uint32_t arrhythmia_model_power_mac_estimates[111] = {84000, 60000, 0, 0, 288, 288, 0, 0, 0, 192000, 40000, 0, 512, 512, 0, 0, 0, 256000, 0, 40000, 0, 512, 512, 0, 0, 0, 256000, 0, 24000, 0, 0, 256, 256, 0, 0, 0, 192000, 18000, 0, 576, 576, 0, 0, 0, 288000, 0, 18000, 0, 576, 576, 0, 0, 0, 288000, 0, 18000, 0, 0, 576, 576, 0, 0, 0, 193536, 12096, 0, 1024, 1024, 0, 0, 0, 258048, 0, 12096, 0, 1024, 1024, 0, 0, 0, 258048, 0, 12096, 0, 0, 1024, 1024, 0, 0, 0, 196608, 9216, 0, 2304, 2304, 0, 0, 0, 294912, 0, 9216, 0, 2304, 2304, 0, 0, 0, 294912, 0, 0, 192};

const char* arrhythmia_model_mac_strings[] = {"1*7*1*500*24*1", "1*5*1*500*24", "0", "0", "1*1*1*1*12*24", "1*1*1*1*24*12", "0", "0", "0", "1*1*1*250*32*24", "1*5*1*250*32", "0", "1*1*1*1*16*32", "1*1*1*1*32*16", "0", "0", "0", "1*1*1*250*32*32", "0", "1*5*1*250*32", "0", "1*1*1*1*16*32", "1*1*1*1*32*16", "0", "0", "0", "1*1*1*250*32*32", "0", "1*3*1*250*32", "0", "0", "1*1*1*1*8*32", "1*1*1*1*32*8", "0", "0", "0", "1*1*1*125*48*32", "1*3*1*125*48", "0", "1*1*1*1*12*48", "1*1*1*1*48*12", "0", "0", "0", "1*1*1*125*48*48", "0", "1*3*1*125*48", "0", "1*1*1*1*12*48", "1*1*1*1*48*12", "0", "0", "0", "1*1*1*125*48*48", "0", "1*3*1*125*48", "0", "0", "1*1*1*1*12*48", "1*1*1*1*48*12", "0", "0", "0", "1*1*1*63*64*48", "1*3*1*63*64", "0", "1*1*1*1*16*64", "1*1*1*1*64*16", "0", "0", "0", "1*1*1*63*64*64", "0", "1*3*1*63*64", "0", "1*1*1*1*16*64", "1*1*1*1*64*16", "0", "0", "0", "1*1*1*63*64*64", "0", "1*3*1*63*64", "0", "0", "1*1*1*1*16*64", "1*1*1*1*64*16", "0", "0", "0", "1*1*1*32*96*64", "1*3*1*32*96", "0", "1*1*1*1*24*96", "1*1*1*1*96*24", "0", "0", "0", "1*1*1*32*96*96", "0", "1*3*1*32*96", "0", "1*1*1*1*24*96", "1*1*1*1*96*24", "0", "0", "0", "1*1*1*32*96*96", "0", "0", "96*1*2", };
const char* arrhythmia_model_output_shapes[] = {"1*1*500*24", "1*1*500*24", "1*1*250*24", "1*1*1*24", "1*1*1*12", "1*1*1*24", "1*1*1*24", "1*1*1*24", "1*1*250*24", "1*1*250*32", "1*1*250*32", "1*1*1*32", "1*1*1*16", "1*1*1*32", "1*1*1*32", "1*1*1*32", "1*1*250*32", "1*1*250*32", "1*1*250*32", "1*1*250*32", "1*1*1*32", "1*1*1*16", "1*1*1*32", "1*1*1*32", "1*1*1*32", "1*1*250*32", "1*1*250*32", "1*1*250*32", "1*1*250*32", "1*1*125*32", "1*1*1*32", "1*1*1*8", "1*1*1*32", "1*1*1*32", "1*1*1*32", "1*1*125*32", "1*1*125*48", "1*1*125*48", "1*1*1*48", "1*1*1*12", "1*1*1*48", "1*1*1*48", "1*1*1*48", "1*1*125*48", "1*1*125*48", "1*1*125*48", "1*1*125*48", "1*1*1*48", "1*1*1*12", "1*1*1*48", "1*1*1*48", "1*1*1*48", "1*1*125*48", "1*1*125*48", "1*1*125*48", "1*1*125*48", "1*1*63*48", "1*1*1*48", "1*1*1*12", "1*1*1*48", "1*1*1*48", "1*1*1*48", "1*1*63*48", "1*1*63*64", "1*1*63*64", "1*1*1*64", "1*1*1*16", "1*1*1*64", "1*1*1*64", "1*1*1*64", "1*1*63*64", "1*1*63*64", "1*1*63*64", "1*1*63*64", "1*1*1*64", "1*1*1*16", "1*1*1*64", "1*1*1*64", "1*1*1*64", "1*1*63*64", "1*1*63*64", "1*1*63*64", "1*1*63*64", "1*1*32*64", "1*1*1*64", "1*1*1*16", "1*1*1*64", "1*1*1*64", "1*1*1*64", "1*1*32*64", "1*1*32*96", "1*1*32*96", "1*1*1*96", "1*1*1*24", "1*1*1*96", "1*1*1*96", "1*1*1*96", "1*1*32*96", "1*1*32*96", "1*1*32*96", "1*1*32*96", "1*1*1*96", "1*1*1*24", "1*1*1*96", "1*1*1*96", "1*1*1*96", "1*1*32*96", "1*1*32*96", "1*1*32*96", "1*96", "1*2", };
const uint32_t arrhythmia_model_output_magnitudes[] = {12000, 12000, 6000, 24, 12, 24, 24, 24, 6000, 8000, 8000, 32, 16, 32, 32, 32, 8000, 8000, 8000, 8000, 32, 16, 32, 32, 32, 8000, 8000, 8000, 8000, 4000, 32, 8, 32, 32, 32, 4000, 6000, 6000, 48, 12, 48, 48, 48, 6000, 6000, 6000, 6000, 48, 12, 48, 48, 48, 6000, 6000, 6000, 6000, 3024, 48, 12, 48, 48, 48, 3024, 4032, 4032, 64, 16, 64, 64, 64, 4032, 4032, 4032, 4032, 64, 16, 64, 64, 64, 4032, 4032, 4032, 4032, 2048, 64, 16, 64, 64, 64, 2048, 3072, 3072, 96, 24, 96, 96, 96, 3072, 3072, 3072, 3072, 96, 24, 96, 96, 96, 3072, 3072, 3072, 96, 2, };
const uint32_t arrhythmia_model_stride_h[] = {1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, };
const uint32_t arrhythmia_model_stride_w[] = {2, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, };
const uint32_t arrhythmia_model_dilation_h[] = {1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, };
const uint32_t arrhythmia_model_dilation_w[] = {1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, };
const char* arrhythmia_model_mac_filter_shapes[] = {"24*1*7*1", "1*1*5*24", "0*0*0*0", "0*0*0*0", "12*1*1*24", "24*1*1*12", "0*0*0*0", "0*0*0*0", "0*0*0*0", "32*1*1*24", "1*1*5*32", "0*0*0*0", "16*1*1*32", "32*1*1*16", "0*0*0*0", "0*0*0*0", "0*0*0*0", "32*1*1*32", "0*0*0*0", "1*1*5*32", "0*0*0*0", "16*1*1*32", "32*1*1*16", "0*0*0*0", "0*0*0*0", "0*0*0*0", "32*1*1*32", "0*0*0*0", "1*1*3*32", "0*0*0*0", "0*0*0*0", "8*1*1*32", "32*1*1*8", "0*0*0*0", "0*0*0*0", "0*0*0*0", "48*1*1*32", "1*1*3*48", "0*0*0*0", "12*1*1*48", "48*1*1*12", "0*0*0*0", "0*0*0*0", "0*0*0*0", "48*1*1*48", "0*0*0*0", "1*1*3*48", "0*0*0*0", "12*1*1*48", "48*1*1*12", "0*0*0*0", "0*0*0*0", "0*0*0*0", "48*1*1*48", "0*0*0*0", "1*1*3*48", "0*0*0*0", "0*0*0*0", "12*1*1*48", "48*1*1*12", "0*0*0*0", "0*0*0*0", "0*0*0*0", "64*1*1*48", "1*1*3*64", "0*0*0*0", "16*1*1*64", "64*1*1*16", "0*0*0*0", "0*0*0*0", "0*0*0*0", "64*1*1*64", "0*0*0*0", "1*1*3*64", "0*0*0*0", "16*1*1*64", "64*1*1*16", "0*0*0*0", "0*0*0*0", "0*0*0*0", "64*1*1*64", "0*0*0*0", "1*1*3*64", "0*0*0*0", "0*0*0*0", "16*1*1*64", "64*1*1*16", "0*0*0*0", "0*0*0*0", "0*0*0*0", "96*1*1*64", "1*1*3*96", "0*0*0*0", "24*1*1*96", "96*1*1*24", "0*0*0*0", "0*0*0*0", "0*0*0*0", "96*1*1*96", "0*0*0*0", "1*1*3*96", "0*0*0*0", "24*1*1*96", "96*1*1*24", "0*0*0*0", "0*0*0*0", "0*0*0*0", "96*1*1*96", "0*0*0*0", "0*0*0*0", "2*96", };
const uint32_t arrhythmia_model_write_estimate[] = {12000, 12000, 6000, 24, 12, 24, 24, 24, 6000, 8000, 8000, 32, 16, 32, 32, 32, 8000, 8000, 8000, 8000, 32, 16, 32, 32, 32, 8000, 8000, 8000, 8000, 4000, 32, 8, 32, 32, 32, 4000, 6000, 6000, 48, 12, 48, 48, 48, 6000, 6000, 6000, 6000, 48, 12, 48, 48, 48, 6000, 6000, 6000, 6000, 3024, 48, 12, 48, 48, 48, 3024, 4032, 4032, 64, 16, 64, 64, 64, 4032, 4032, 4032, 4032, 64, 16, 64, 64, 64, 4032, 4032, 4032, 4032, 2048, 64, 16, 64, 64, 64, 2048, 3072, 3072, 96, 24, 96, 96, 96, 3072, 3072, 3072, 3072, 96, 24, 96, 96, 96, 3072, 3072, 3072, 96, 2, };
const uint32_t arrhythmia_model_read_estimate[] = {3500, 60000, 12000, 6000, 24, 12, 24, 24, 6000, 6000, 40000, 8000, 32, 16, 32, 32, 8000, 8000, 8000, 40000, 8000, 32, 16, 32, 32, 8000, 8000, 8000, 24000, 8000, 4000, 32, 8, 32, 32, 4000, 4000, 18000, 6000, 48, 12, 48, 48, 6000, 6000, 6000, 18000, 6000, 48, 12, 48, 48, 6000, 6000, 6000, 18000, 6000, 3024, 48, 12, 48, 48, 3024, 3024, 12096, 4032, 64, 16, 64, 64, 4032, 4032, 4032, 12096, 4032, 64, 16, 64, 64, 4032, 4032, 4032, 12096, 4032, 2048, 64, 16, 64, 64, 2048, 2048, 9216, 3072, 96, 24, 96, 96, 3072, 3072, 3072, 9216, 3072, 96, 24, 96, 96, 3072, 3072, 3072, 3072, 192, };

ns_perf_mac_count_t basic_mac = {
        .number_of_layers = 111, 
        .mac_count_map = arrhythmia_model_power_mac_estimates,
        .output_magnitudes = (uint32_t *)arrhythmia_model_output_magnitudes,
        .stride_h = (uint32_t *)arrhythmia_model_stride_h,
        .stride_w = (uint32_t *)arrhythmia_model_stride_w,
        .dilation_h = (uint32_t *)arrhythmia_model_dilation_h,
        .dilation_w = (uint32_t *)arrhythmia_model_dilation_w,
        .mac_compute_string = (const char **)arrhythmia_model_mac_strings,
        .output_shapes = (const char **)arrhythmia_model_output_shapes,
        .filter_shapes = (const char **)arrhythmia_model_mac_filter_shapes
    };

/**
 * @brief Initialize TF with model using minimal configuration
 *
 * This function provides a simplified initialization interface that sets up
 * the model state with minimal configuration and then calls the main init function.
 * It's useful for applications that want a simpler initialization process.
 *
 * @param ms Model state and configuration struct
 * @return int status
 */
int
ns_model_minimal_init(ns_model_state_t *ms) {
    // Set up minimal configuration
    ms->runtime = TFLM;
    ms->numInputTensors = 1;  // Default to 1 input tensor
    ms->numOutputTensors = 1; // Default to 1 output tensor
    ms->rv_count = 0;         // No resource variables by default
    ms->numInputTensors = 1;
    ms->numOutputTensors = 1;
    
    // Initialize profiling components if enabled
#ifdef NS_MLPROFILE
    // Timer configuration for profiling
    static ns_timer_config_t basic_tickTimer = {
        .api = &ns_timer_V1_0_0,
        .timer = NS_TIMER_COUNTER,
        .enableInterrupt = false,
    };
    ms->tickTimer = &basic_tickTimer;

    ms->mac_estimates = &basic_mac;
    
    #ifdef AM_PART_APOLLO5B
    // PMU configuration for Apollo5B
    static ns_pmu_config_t basic_pmu_cfg;
    basic_pmu_cfg.api = &ns_pmu_V1_0_0;
    ns_pmu_reset_config(&basic_pmu_cfg);
    
    // Add default PMU events for profiling
    ns_pmu_event_create(&basic_pmu_cfg.events[0], NS_PROFILER_PMU_EVENT_0, NS_PMU_EVENT_COUNTER_SIZE_32);
    ns_pmu_event_create(&basic_pmu_cfg.events[1], NS_PROFILER_PMU_EVENT_1, NS_PMU_EVENT_COUNTER_SIZE_32);
    ns_pmu_event_create(&basic_pmu_cfg.events[2], NS_PROFILER_PMU_EVENT_2, NS_PMU_EVENT_COUNTER_SIZE_32);
    ns_pmu_event_create(&basic_pmu_cfg.events[3], NS_PROFILER_PMU_EVENT_3, NS_PMU_EVENT_COUNTER_SIZE_32);
    ns_pmu_init(&basic_pmu_cfg);
    ms->pmu = &basic_pmu_cfg;
    #else
    ms->pmu = NULL;
    #endif
#else
    ms->tickTimer = NULL;
    ms->mac_estimate = NULL;
    ms->pmu = NULL;
#endif

    // Call the main initialization function
    return ns_model_init(ms);
}

/**
 * @brief Initialize TF with model
 *
 * This code is fairly common across most TF-based models.
 * The major differences relate to input and output tensor
 * handling.
 *
 */
int
ns_model_init(ns_model_state_t *ms) {
    ms->state = NOT_READY;

    tflite::MicroErrorReporter micro_error_reporter;
    ms->error_reporter = &micro_error_reporter;

#ifdef NS_MLPROFILE
    // Need a timer for the profiler to collect latencies
    NS_TRY(ns_timer_init(ms->tickTimer), "Timer init failed.\n");
    static tflite::MicroProfiler micro_profiler;
    ms->profiler = &micro_profiler;

    // Create the config struct for the debug log
    ns_debug_log_init_t cfg = {
        .t = ms->tickTimer,
        .m = ms->mac_estimates,
        #ifdef AM_PART_APOLLO5B
        .pmu = ms->pmu,
        #endif
    };
    ns_TFDebugLogInit(&cfg);
#else
    #ifdef NS_MLDEBUG
    ns_TFDebugLogInit(NULL);
    #endif
#endif

    tflite::InitializeTarget();

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    ms->model = tflite::GetModel(ms->model_array);
    if (ms->model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(ms->error_reporter,
                             "Model provided is schema version %d not equal "
                             "to supported version %d.",
                             ms->model->version(), TFLITE_SCHEMA_VERSION);
        return NS_STATUS_FAILURE;
    }


    // Use a comprehensive op resolver that includes all common ops
    // This ensures compatibility with any TensorFlow Lite model
    static tflite::MicroMutableOpResolver<112> resolver; // Large enough for most models
    
    // Set up the resolver with all supported operations
    if (!setup_supported_resolver(resolver)) {
        TF_LITE_REPORT_ERROR(ms->error_reporter, "Failed to set up ops resolver");
        return NS_STATUS_FAILURE;
    }

    // Allocate ResourceVariable stuff if needed
    tflite::MicroResourceVariables *resource_variables;
    tflite::MicroAllocator *var_allocator;

    if (ms->rv_count != 0) {
        var_allocator = tflite::MicroAllocator::Create(ms->rv_arena, ms->rv_arena_size, nullptr);
        resource_variables = tflite::MicroResourceVariables::Create(var_allocator, ms->rv_count);
    } else {
        resource_variables = nullptr;
    }

    // Build an interpreter to run the model with.
#ifdef NS_TFSTRUCTURE_RECENT
    static tflite::MicroInterpreter static_interpreter(
        ms->model, resolver, ms->arena, ms->arena_size, resource_variables, ms->profiler);
#else
    static tflite::MicroInterpreter static_interpreter(
        ms->model, resolver, ms->arena, ms->arena_size, ms->error_reporter, nullptr, ms->profiler);
#endif
    ms->interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = ms->interpreter->AllocateTensors();

    if (allocate_status != kTfLiteOk) {
        ns_lp_printf("MODEL FAILURE\n");
        TF_LITE_REPORT_ERROR(ms->error_reporter, "AllocateTensors() failed");
        ms->computed_arena_size = 0xDEADBEEF;
        return NS_STATUS_FAILURE;
    }

    ms->computed_arena_size = ms->interpreter->arena_used_bytes(); // prep to send back to PC

    // Update actual tensor counts from interpreter
    ms->numInputTensors = ms->interpreter->inputs_size();
    ms->numOutputTensors = ms->interpreter->outputs_size();

    // Obtain pointers to the model's input and output tensors.
    for (uint32_t t = 0; t < ms->numInputTensors; t++) {
        ms->model_input[t] = ms->interpreter->input(t);
    }

    for (uint32_t t = 0; t < ms->numOutputTensors; t++) {
        ms->model_output[t] = ms->interpreter->output(t);
    }

    ms->state = READY;
    return NS_STATUS_SUCCESS;
}

uint32_t
ns_tf_get_num_input_tensors(ns_model_state_t *ms) {
    return ms->interpreter->inputs_size();
}

uint32_t
ns_tf_get_num_output_tensors(ns_model_state_t *ms) {
    return ms->interpreter->outputs_size();
}

TfLiteTensor *
ns_tf_get_input_tensor(ns_model_state_t *ms, uint32_t index) {
    return ms->interpreter->input(index);
}

TfLiteTensor *
ns_tf_get_output_tensor(ns_model_state_t *ms, uint32_t index) {
    return ms->interpreter->output(index);
}
