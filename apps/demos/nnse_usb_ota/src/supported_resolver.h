/**
 * @file tflm_ops_resolver.h
 * @brief Comprehensive TensorFlow Lite Micro operations resolver
 * @version 0.1
 * @date 2024-12-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef TFLM_OPS_RESOLVER_H
#define TFLM_OPS_RESOLVER_H

#include <cstddef> // for size_t
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

/**
 * @brief Sets up a comprehensive op resolver with all supported TFLM operations
 * 
 * This function adds all currently supported TensorFlow Lite Micro operations to the resolver,
 * ensuring compatibility with a wide variety of models without needing to analyze
 * the model structure.
 * 
 * @tparam N The maximum number of operations the resolver can hold
 * @param resolver Reference to the MicroMutableOpResolver to populate
 * @return true if all operations were added successfully, false otherwise
 */

 // Supported Operations:
 // https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_mutable_op_resolver.h
template<size_t N>
bool setup_supported_resolver(tflite::MicroMutableOpResolver<N>& resolver) {
    // Add all commonly used TensorFlow Lite operations
    // This ensures compatibility with any TensorFlow Lite model

    resolver.AddAbs();
    resolver.AddAdd();
    resolver.AddAddN();
    resolver.AddArgMax();
    resolver.AddArgMin();
    resolver.AddAssignVariable();
    resolver.AddAveragePool2D();
    resolver.AddBatchMatMul();
    resolver.AddBatchToSpaceNd();
    resolver.AddBroadcastArgs();
    resolver.AddBroadcastTo();
    resolver.AddCallOnce();
    resolver.AddCast();
    resolver.AddCeil();
    resolver.AddCircularBuffer();
    resolver.AddConcatenation();
    resolver.AddConv2D();
    resolver.AddCos();
    resolver.AddCumSum();
    resolver.AddDelay();
    resolver.AddDepthToSpace();
    resolver.AddDepthwiseConv2D();
    resolver.AddDequantize();
    resolver.AddDetectionPostprocess();
    resolver.AddDiv();
    resolver.AddEmbeddingLookup();
    resolver.AddEnergy();
    resolver.AddElu();
    resolver.AddEqual();
    resolver.AddEthosU();
    resolver.AddExp();
    resolver.AddExpandDims();
    resolver.AddFftAutoScale();
    resolver.AddFill();
    resolver.AddFilterBank();
    resolver.AddFilterBankLog();
    resolver.AddFilterBankSquareRoot();
    resolver.AddFilterBankSpectralSubtraction();
    resolver.AddFloor();
    resolver.AddFloorDiv();
    resolver.AddFloorMod();
    resolver.AddFramer();
    resolver.AddFullyConnected();
    resolver.AddGather();
    resolver.AddGatherNd();
    resolver.AddGreater();
    resolver.AddGreaterEqual();
    resolver.AddHardSwish();
    resolver.AddIf();
    resolver.AddIrfft();
    resolver.AddL2Normalization();
    resolver.AddL2Pool2D();
    resolver.AddLeakyRelu();
    resolver.AddLess();
    resolver.AddLessEqual();
    resolver.AddLog();
    resolver.AddLogicalAnd();
    resolver.AddLogicalNot();
    resolver.AddLogicalOr();
    resolver.AddLogistic();
    resolver.AddLogSoftmax();
    resolver.AddMaximum();
    resolver.AddMaxPool2D();
    resolver.AddMirrorPad();
    resolver.AddMean();
    resolver.AddMinimum();
    resolver.AddMul();
    resolver.AddNeg();
    resolver.AddNotEqual();
    resolver.AddOverlapAdd();
    resolver.AddPack();
    resolver.AddPad();
    resolver.AddPadV2();
    resolver.AddPCAN();
    resolver.AddPrelu();
    resolver.AddQuantize();
    resolver.AddReadVariable();
    resolver.AddReduceMax();
    resolver.AddRelu();
    resolver.AddRelu6();
    resolver.AddReshape();
    resolver.AddResizeBilinear();
    resolver.AddResizeNearestNeighbor();
    resolver.AddRfft();
    resolver.AddRound();
    resolver.AddRsqrt();
    resolver.AddSelectV2();
    resolver.AddShape();
    resolver.AddSin();
    resolver.AddSlice();
    resolver.AddSoftmax();
    resolver.AddSpaceToBatchNd();
    resolver.AddSpaceToDepth();
    resolver.AddSplit();
    resolver.AddSplitV();
    resolver.AddSqueeze();
    resolver.AddSqrt();
    resolver.AddSquare();
    resolver.AddSquaredDifference();
    resolver.AddStridedSlice();
    resolver.AddStacker();
    resolver.AddSub();
    resolver.AddSum();
    resolver.AddSvdf();
    resolver.AddTanh();
    resolver.AddTransposeConv();
    resolver.AddTranspose();
    resolver.AddUnpack();
    resolver.AddUnidirectionalSequenceLSTM();
    resolver.AddVarHandle();
    resolver.AddWhile();
    resolver.AddWindow();
    resolver.AddZerosLike();
    
    return true;
}

#endif // TFLM_OPS_RESOLVER_H 