/**
 * @file tflm_profiling.h
 * @author Carlos Morales
 * @brief TFLM profiling functions for neuralSPOT
 * @version 0.1
 * @date 2023-02-28
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef TFLM_PROFILING_H
#define TFLM_PROFILING_H

#include "ns_model.h"
// #include "tensorflow/lite/core/c/c_api_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Run comprehensive TFLM profiling on a model
 * 
 * This function performs comprehensive profiling including:
 * - Power monitoring initialization
 * - PMU counter setup and reading
 * - Layer detection
 * - Performance metrics collection
 * 
 * @param model_state Pointer to the model state structure
 * @return TfLiteStatus The status of the inference operation
 */
TfLiteStatus tflm_profiling_run(ns_model_state_t *model_state);

#ifdef __cplusplus
}
#endif

#endif // TFLM_PROFILING_H 