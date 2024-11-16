/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "MatrixMultiplication.h"
#include "hlslib/xilinx/Stream.h"

// Original ProcessingElement declaration
void ProcessingElement(Stream<ComputePackN_t> &aIn,
                       Stream<ComputePackN_t> &aOut,
                       Stream<ComputePackM_t> &bIn,
                       Stream<ComputePackM_t> &bOut,
                       Stream<ComputePackM_t> &cOut,
                       Stream<ComputePackM_t> &cIn,
                       const unsigned locationN,
                       const unsigned size_n,
                       const unsigned size_k,
                       const unsigned size_m);

// Tucker configuration structure
struct TuckerConfig {
    // 基本维度参数
    int batch_size;       // 批大小
    int in_features;      // 输入特征维度
    int out_features;     // 输出特征维度
    
    // Tucker分解的秩参数
    int rank1;           // 第一个模式的秩
    int rank2;           // 第二个模式的秩
    int rank3;           // 第三个模式的秩

    // 构造函数，提供默认值
    TuckerConfig(
        int _batch_size = 1,
        int _in_features = 1,
        int _out_features = 1,
        int _rank1 = 1,
        int _rank2 = 1,
        int _rank3 = 1
    ) : batch_size(_batch_size),
        in_features(_in_features),
        out_features(_out_features),
        rank1(_rank1),
        rank2(_rank2),
        rank3(_rank3) {}
};

// Tucker ProcessingElement declaration
void TuckerProcessingElement(Stream<ComputePackN_t> &aIn,
                            Stream<ComputePackN_t> &aOut,
                            Stream<ComputePackM_t> &bIn,
                            Stream<ComputePackM_t> &bOut,
                            Stream<ComputePackM_t> &cOut,
                            Stream<ComputePackM_t> &cIn,
                            const unsigned locationN,
                            const TuckerConfig &config);
