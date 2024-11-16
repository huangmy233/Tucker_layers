#pragma once
#include "MatrixMultiplication.h"

// 定义Tucker层的配置结构
struct TuckerConfig {
    unsigned in_features;
    unsigned out_features;
    unsigned batch_size;
    unsigned rank;
    // 继承MatrixMultiplication的配置
    bool dynamic_sizes;
    unsigned memory_width_n;
    unsigned memory_width_k;
    unsigned memory_width_m;
};

// 修改 Compute.h - 添加新的计算单元
void TuckerProcessingElement(
    Stream<ComputePackN_t> &aIn,
    Stream<ComputePackN_t> &aOut, 
    Stream<ComputePackM_t> &bIn,
    Stream<ComputePackM_t> &bOut,
    Stream<ComputePackM_t> &cOut,
    Stream<ComputePackM_t> &cIn,
    const unsigned locationN,
    const TuckerConfig &config);

// 修改 Memory.h - 添加Tucker专用的内存访问模式
void TuckerReadFactorMatrix(
    MemoryPackK_t const memory[],
    Stream<MemoryPackK_t> &pipe,
    const TuckerConfig &config);

void TuckerWriteOutput(
    Stream<MemoryPackM_t> &pipe,
    MemoryPackM_t memory[],
    const TuckerConfig &config);

// Top.cpp 中添加新的顶层函数
extern "C" {
void TuckerLayerKernel(
    MemoryPackK_t const *input,
    MemoryPackK_t const *core_tensor,
    MemoryPackK_t const *factor_matrices[3],
    MemoryPackM_t *output,
    const TuckerConfig &config);
}