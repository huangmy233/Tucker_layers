#pragma once

#include "MatrixMultiplication.h"
#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/DataPack.h"

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
};
