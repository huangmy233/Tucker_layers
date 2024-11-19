/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Utility.h"
#include "MatrixMultiplication.h"
#include "Compute.h"
#include <cassert>

void MatrixMultiplicationKernel(
    MemoryPackK_t const input_a[],
    MemoryPackM_t const input_b[],
    MemoryPackM_t output[],
    const unsigned size_n,
    const unsigned size_k, 
    const unsigned size_m) {

    #pragma HLS INTERFACE m_axi port=input_a offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=input_b offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=output  offset=slave bundle=gmem2 
    #pragma HLS INTERFACE s_axilite port=size_n bundle=control
    #pragma HLS INTERFACE s_axilite port=size_k bundle=control
    #pragma HLS INTERFACE s_axilite port=size_m bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // 使用固定大小的局部缓冲区
    static Data_t a_local[MAX_SIZE_N * MAX_SIZE_K];
    static Data_t b_local[MAX_SIZE_K * MAX_SIZE_M];
    static Data_t c_local[MAX_SIZE_N * MAX_SIZE_M];

    #pragma HLS ARRAY_PARTITION variable=a_local cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=b_local cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=c_local cyclic factor=8

    // 加载输入矩阵 A 到局部内存
    LoadInputA: for (unsigned i = 0; i < (size_n * size_k + kMemoryWidthK - 1) / kMemoryWidthK; ++i) {
        #pragma HLS PIPELINE II=1
        if (i < size_n * size_k / kMemoryWidthK) {
            auto pack = input_a[i];
            for (unsigned j = 0; j < kMemoryWidthK; ++j) {
                unsigned idx = i * kMemoryWidthK + j;
                if (idx < size_n * size_k) {
                    a_local[idx] = pack[j];
                }
            }
        }
    }

    // 加载输入矩阵 B 到局部内存
    LoadInputB: for (unsigned i = 0; i < (size_k * size_m + kMemoryWidthM - 1) / kMemoryWidthM; ++i) {
        #pragma HLS PIPELINE II=1
        if (i < size_k * size_m / kMemoryWidthM) {
            auto pack = input_b[i];
            for (unsigned j = 0; j < kMemoryWidthM; ++j) {
                unsigned idx = i * kMemoryWidthM + j;
                if (idx < size_k * size_m) {
                    b_local[idx] = pack[j];
                }
            }
        }
    }

    // 矩阵乘法计算
    ComputeLoop_N: for (unsigned n = 0; n < size_n; ++n) {
        ComputeLoop_M: for (unsigned m = 0; m < size_m; ++m) {
            #pragma HLS PIPELINE II=1
            Data_t acc = 0;
            ComputeLoop_K: for (unsigned k = 0; k < size_k; ++k) {
                const auto a_val = a_local[n * size_k + k];
                const auto b_val = b_local[k * size_m + m];
                acc = OperatorReduce::Apply(acc, 
                      OperatorMap::Apply(a_val, b_val));
            }
            c_local[n * size_m + m] = acc;
        }
    }

    // 将结果写回到输出矩阵
    WriteOutput: for (unsigned i = 0; i < (size_n * size_m + kMemoryWidthM - 1) / kMemoryWidthM; ++i) {
        #pragma HLS PIPELINE II=1
        if (i < size_n * size_m / kMemoryWidthM) {
            MemoryPackM_t pack;
            for (unsigned j = 0; j < kMemoryWidthM; ++j) {
                unsigned idx = i * kMemoryWidthM + j;
                if (idx < size_n * size_m) {
                    pack[j] = c_local[idx];
                } else {
                    pack[j] = 0;
                }
            }
            output[i] = pack;
        }
    }
}