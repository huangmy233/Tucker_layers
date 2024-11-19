/// @author    Your Name
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include "TuckerLinear.h"
#include "Utility.h"

// Simple direct matrix multiplication for reference
void MatrixMultiplyReference(
    const Data_t* A,        // Input matrix A
    const Data_t* B,        // Input matrix B
    Data_t* C,             // Output matrix C
    const unsigned rows_a,  // Rows of A
    const unsigned cols_a,  // Cols of A = Rows of B
    const unsigned cols_b   // Cols of B
) {
  for (unsigned i = 0; i < rows_a; ++i) {
    for (unsigned j = 0; j < cols_b; ++j) {
      Data_t sum = 0;
      for (unsigned k = 0; k < cols_a; ++k) {
        sum = OperatorReduce::Apply(sum,
              OperatorMap::Apply(A[i * cols_a + k], B[k * cols_b + j]));
      }
      C[i * cols_b + j] = sum;
    }
  }
}

// Reference implementation of Tucker decomposition
void TuckerReferenceImplementation(
    const Data_t* input,    // [I x B]
    const Data_t* U1,       // [R1 x I]
    const Data_t* S,        // [R2 x R1]
    const Data_t* U2,       // [O x R2]
    Data_t* output,         // [O x B]
    const unsigned size_n,  // Input dimension I
    const unsigned size_k,  // First rank R1
    const unsigned size_m,  // Batch size B
    const unsigned size_r2  // Second rank R2
) {
    // 临时缓冲区
    std::vector<Data_t> tmp1(size_k * size_m, 0.0);   // [R1 x B]
    std::vector<Data_t> tmp2(size_r2 * size_m, 0.0);  // [R2 x B]

    // 打印输入样本
    std::cout << "\nReference Implementation:" << std::endl;
    std::cout << "Input[0,0]: " << input[0] << std::endl;

    // Step 1: tmp1 = U1 × input [R1 x I] × [I x B] = [R1 x B]
    for (unsigned r1 = 0; r1 < size_k; ++r1) {
        for (unsigned b = 0; b < size_m; ++b) {
            Data_t sum = 0;
            for (unsigned i = 0; i < size_n; ++i) {
                sum += U1[r1 * size_n + i] * input[i * size_m + b];
            }
            tmp1[r1 * size_m + b] = sum;
        }
    }
    std::cout << "After U1[0,0]: " << tmp1[0] << std::endl;

    // Step 2: tmp2 = S × tmp1 [R2 x R1] × [R1 x B] = [R2 x B]
    for (unsigned r2 = 0; r2 < size_r2; ++r2) {
        for (unsigned b = 0; b < size_m; ++b) {
            Data_t sum = 0;
            for (unsigned r1 = 0; r1 < size_k; ++r1) {
                sum += S[r2 * size_k + r1] * tmp1[r1 * size_m + b];
            }
            tmp2[r2 * size_m + b] = sum;
        }
    }
    std::cout << "After S[0,0]: " << tmp2[0] << std::endl;

    // Step 3: output = U2 × tmp2 [O x R2] × [R2 x B] = [O x B]
    for (unsigned o = 0; o < size_n; ++o) {
        for (unsigned b = 0; b < size_m; ++b) {
            Data_t sum = 0;
            for (unsigned r2 = 0; r2 < size_r2; ++r2) {
                sum += U2[o * size_r2 + r2] * tmp2[r2 * size_m + b];
            }
            output[o * size_m + b] = sum;
        }
    }
    std::cout << "Reference Output[0,0]: " << output[0] << std::endl;
}


int main(int argc, char **argv) {
#ifdef MM_DYNAMIC_SIZES
  if (argc < 4 || argc > 4) {
    std::cerr << "Usage: ./TestTuckerSimulation I K M" << std::endl;
    return 1;
  }
  const unsigned size_n = std::stoul(argv[1]);  // Input dimension
  const unsigned size_k = std::stoul(argv[2]);  // First rank R1
  const unsigned size_m = std::stoul(argv[3]);  // Batch size
  const unsigned size_r2 = size_k;  // Second rank R2 = R1
#else
  constexpr auto size_n = kSizeN;
  constexpr auto size_k = kSizeK;
  constexpr auto size_m = kSizeM;
  constexpr auto size_r2 = size_k;
#endif
   bool has_mismatch = false;
  // Validation checks
  if (size_k % kMemoryWidthK != 0) {
    std::cerr << "K must be divisible by memory width." << std::endl;
    return 1;
  }
  if (size_m % kMemoryWidthM != 0) {
    std::cerr << "M must be divisible by memory width." << std::endl;
    return 1;
  }

  // 1. 首先定义随机数生成器和分布
    std::default_random_engine rng(kSeed);  // 使用固定种子保证可重现性
    std::normal_distribution<float> dist(0.0f, 0.1f);  // 均值0，标准差0.1的正态分布

    // 2. 声明并分配内存
    std::vector<Data_t> input(size_n * size_m);
    std::vector<Data_t> U1(size_k * size_n);    // [R1 x I]
    std::vector<Data_t> U2(size_n * size_r2);   // [O x R2]
    std::vector<Data_t> S(size_r2 * size_k);    // [R2 x R1]
    std::vector<Data_t> output(size_n * size_m, 0);
    std::vector<Data_t> reference(size_n * size_m, 0);

    // 3. 初始化输入数据（使用均匀分布）
    std::uniform_real_distribution<double> input_dist(1.0, 10.0);  // 输入数据范围[1,10]
    for(auto& val : input) {
        val = static_cast<Data_t>(input_dist(rng));
    }

    // 4. 初始化权重矩阵（使用正态分布）
    // 初始化U1
    for(auto& val : U1) {
        val = static_cast<Data_t>(dist(rng));
    }
    std::cout << "U1 samples: " << U1[0] << ", " << U1[1] << "\n";  // 打印样本检查

    // 初始化U2
    for(auto& val : U2) {
        val = static_cast<Data_t>(dist(rng));
    }
    std::cout << "U2 samples: " << U2[0] << ", " << U2[1] << "\n";

    // 初始化S
    for(auto& val : S) {
        val = static_cast<Data_t>(dist(rng));
    }
    std::cout << "S samples: " << S[0] << ", " << S[1] << "\n";

    // 5. Pack数据用于kernel
    const auto inputKernel = Pack<kMemoryWidthM>(input);
    auto outputKernel = Pack<kMemoryWidthM>(output);
    const auto U1Kernel = Pack<kMemoryWidthK>(U1);
    const auto U2Kernel = Pack<kMemoryWidthM>(U2);
    const auto SKernel = Pack<kMemoryWidthM>(S);

    // 6. 创建参数对象
    tucker::TuckerParams<Data_t> params(
        size_n, size_n, size_m, size_k, size_r2
    );

    std::cout << "Running Tucker simulation...\n" << std::flush;

    // 7. 执行计算
    TuckerLinearKernel(
        inputKernel.data(),
        outputKernel.data(),
        size_n, size_n, size_m,
        size_k, size_r2,
        true  // 第一次运行初始化权重
    );

    // 8. 运行参考实现
    std::cout << "Running reference implementation...\n" << std::flush;
    TuckerReferenceImplementation(
        input.data(),
        U1.data(), 
        S.data(), 
        U2.data(),
        reference.data(),
        size_n, size_k, size_m, size_r2
    );

    // 9. 验证结果
    std::cout << "Verifying results...\n" << std::flush;
    const auto outputTest = Unpack<kMemoryWidthM>(outputKernel);

    // 添加详细的验证输出
    for (unsigned i = 0; i < std::min(size_n, 3u); ++i) {
        for (unsigned j = 0; j < std::min(size_m, 3u); ++j) {
            std::cout << "Position (" << i << "," << j << "): \n";
            std::cout << "  Test value: " << outputTest[i * size_m + j] << "\n";
            std::cout << "  Reference: " << reference[i * size_m + j] << "\n";
        }
    }

    // 10. 结果比较
    for (unsigned i = 0; i < size_n; ++i) {
    for (unsigned j = 0; j < size_m; ++j) {
        const auto testVal = make_signed<Data_t>(outputTest[i * size_m + j]);
        const auto refVal = make_signed<Data_t>(reference[i * size_m + j]);
        
        // 计算绝对误差和相对误差
        const Data_t abs_diff = std::abs(testVal - refVal);
        const Data_t rel_diff = abs_diff / (std::abs(refVal) + 1e-6);
        
        // 使用更合理的阈值
        const Data_t abs_threshold = 1e-3;
        const Data_t rel_threshold = 1e-1;
        
        if (abs_diff > abs_threshold && rel_diff > rel_threshold) {
            std::cerr << "Mismatch at (" << i << ", " << j << "):\n"
                     << "  Test value: " << testVal << "\n"
                     << "  Reference: " << refVal << "\n"
                     << "  Absolute diff = " << abs_diff 
                     << ", Relative diff = " << rel_diff << "\n";
             has_mismatch = true;  // 标记有错误但继续验证;
            }
        }
    }

    // 最后输出统计信息
std::cout << "\nVerification complete.\n";
if (has_mismatch) {
    std::cout << "Found mismatches. Check tucker_results.txt for details.\n";
    return 1;
} else {
    std::cout << "All results within acceptable tolerance.\n";
    return 0;
}
}