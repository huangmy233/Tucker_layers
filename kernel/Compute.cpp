/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Utility.h"
#include "MatrixMultiplication.h"
#include "Memory.h"
#include <cassert>

void ProcessingElement(Stream<ComputePackN_t> &aIn,
                       Stream<ComputePackN_t> &aOut,
                       Stream<ComputePackM_t> &bIn,
                       Stream<ComputePackM_t> &bOut,
                       Stream<ComputePackM_t> &cOut,
                       Stream<ComputePackM_t> &cIn, const unsigned locationN,
                       const unsigned size_n, const unsigned size_k,
                       const unsigned size_m) {
  // A is double-buffered, such that new values can be read while the 
  // previous outer product is being computed. This is required to achieve
  // a perfect pipeline across the K-dimension, which is necessary for
  // many processing elements (kInnerTileSizeN).
  ComputePackN_t aBuffer[2 * kInnerTilesN];

  // This is where we spend all our T^2 fast memory
  ComputePackM_t cBuffer[kInnerTilesN * kInnerTilesM][kComputeTileSizeN];
  #pragma HLS ARRAY_PARTITION variable=cBuffer complete dim=2

  // Populate the buffer for the first outer product 
InitializeABuffer_Inner:
  for (unsigned n2 = 0; n2 < kInnerTilesN; ++n2) {
    if (locationN < kComputeTilesN - 1) {
      // All but the last processing element 
    InitializeABuffer_Outer:
      for (unsigned n1 = 0; n1 < kComputeTilesN - locationN; ++n1) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_FLATTEN
        const auto read = aIn.Pop();
        if (n1 == 0) {
          aBuffer[n2] = read;
        } else {
          aOut.Push(read);
        }
      }
    } else {
      // Last processing element gets a special case, because Vivado HLS
      // refuses to flatten and pipeline loops with trip count 1
      #pragma HLS PIPELINE II=1
      aBuffer[n2] = aIn.Pop();
    }
  }

OuterTile_N:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0) {
  OuterTile_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0) {

      // We do not tile K further, but loop over the entire outer tile here
    Collapse_K:
      for (unsigned k = 0; k < size_k; ++k) {
        // Begin outer tile ---------------------------------------------------

      Pipeline_N:
        for (unsigned n1 = 0; n1 < kInnerTilesN; ++n1) {

        Pipeline_M:
          for (unsigned m1 = 0; m1 < kInnerTilesM; ++m1) {

            // Begin compute tile ---------------------------------------------
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN

            static_assert(kInnerTilesM >= kInnerTilesN,
                          "Double buffering does not work if there are more "
                          "N-tiles than M-tiles");

            // Double-buffering scheme. This hijacks the m1-index to perform
            // the buffering and forwarding of values for the following outer
            // product, required to flatten the K-loop.
            if ((n0 < OuterTilesN(size_n) - 1 || m0 < OuterTilesM(size_m) - 1 ||
                 k < size_k - 1) &&
                m1 >= locationN            // Start at own index.
                && m1 < kComputeTilesN) {  // Number of PEs in front.
              const auto read = aIn.Pop();
              if (m1 == locationN) {
                // Double buffering
                aBuffer[n1 + (k % 2 == 0 ? kInnerTilesN : 0)] = read;
                #pragma HLS DEPENDENCE variable=aBuffer false
              } else {
                // Without this check, Vivado HLS thinks aOut can be written
                // from the last processing element and fails dataflow
                // checking.
                if (locationN < kComputeTilesN - 1) {
                  aOut.Push(read);
                }
              }
            }

            // Double buffering, read from the opposite end of where the buffer
            // is being written
            const auto aVal = aBuffer[n1 + (k % 2 == 0 ? 0 : kInnerTilesN)];
            #pragma HLS DEPENDENCE variable=aBuffer false
            const auto bVal = bIn.Pop();
            if (locationN < kComputeTilesN - 1) {
              bOut.Push(bVal);
            }

          Unroll_N:
            for (unsigned n2 = 0; n2 < kComputeTileSizeN; ++n2) {
              #pragma HLS UNROLL

              const bool inBoundsN = ((n0 * kInnerTilesN * kComputeTileSizeN +
                                       n1 * kComputeTileSizeN + n2) < size_n);

              ComputePackM_t cStore;
              const auto cPrev = (k > 0)
                                     ? cBuffer[n1 * kInnerTilesM + m1][n2]
                                     : ComputePackM_t(static_cast<Data_t>(0));

            Unroll_M:
              for (unsigned m2 = 0; m2 < kComputeTileSizeM; ++m2) {
                #pragma HLS UNROLL

                const bool inBoundsM = ((m0 * kInnerTilesM * kComputeTileSizeM +
                                         m1 * kComputeTileSizeM + m2) < size_m);

                const bool inBounds = inBoundsN && inBoundsM;

                const auto mapped = OperatorMap::Apply(aVal[n2], bVal[m2]);
                MM_MULT_RESOURCE_PRAGMA(mapped);
                const auto prev = cPrev[m2];

                const auto reduced = OperatorReduce::Apply(prev, mapped);
                MM_ADD_RESOURCE_PRAGMA(reduced);
                // If out of bounds, propagate the existing value instead of
                // storing the newly computed value
                cStore[m2] = inBounds ? reduced : prev; 
                #pragma HLS DEPENDENCE variable=cBuffer false
              }

              cBuffer[n1 * kInnerTilesM + m1][n2] = cStore;
            }

            // End compute tile -----------------------------------------------
          }
        }

        // End outer tile -----------------------------------------------------
      }

      // Write back tile of C -------------------------------------------------
      // 
      // This uses a flattened implementation of the loops, as we otherwise
      // introduce a lot of pipeline drains, which can have a small performance
      // impact for large designs.
      //
      const unsigned writeFlattenedInner =
          (kComputeTileSizeN * kInnerTilesM +
           (kComputeTilesN - locationN - 1) * kComputeTileSizeN * kInnerTilesM);
      const unsigned writeFlattened = kInnerTilesN * writeFlattenedInner;
      ap_uint<hlslib::ConstLog2(kInnerTilesN)> n1 = 0;
      ap_uint<((kComputeTileSizeN > 1) ? hlslib::ConstLog2(kComputeTileSizeN)
                                       : 1)>
          n2 = 0;
      ap_uint<hlslib::ConstLog2(kInnerTilesM)> m1 = 0;
      unsigned inner = 0;
    WriteC_Flattened:
      for (unsigned i = 0; i < writeFlattened; ++i) {
        #pragma HLS PIPELINE II=1
        if (inner < kComputeTileSizeN * kInnerTilesM) {
          cOut.Push(cBuffer[n1 * kInnerTilesM + m1][n2]);
          if (m1 == kInnerTilesM - 1) {
            m1 = 0;
            if (n2 == kComputeTileSizeN - 1) {
              n2 = 0;
            } else {
              ++n2;
            }
          } else {
            ++m1;
          }
        } else {
          if (locationN < kComputeTilesN - 1) {
            cOut.Push(cIn.Pop());
          }
        }
        if (inner == writeFlattenedInner - 1) {
          inner = 0;
          ++n1;
        } else {
          ++inner;
        }
      }
    }
  }
}
void TuckerProcessingElement(Stream<ComputePackN_t> &aIn,
                            Stream<ComputePackN_t> &aOut,
                            Stream<ComputePackM_t> &bIn,
                            Stream<ComputePackM_t> &bOut,
                            Stream<ComputePackM_t> &cOut,
                            Stream<ComputePackM_t> &cIn,
                            const unsigned locationN,
                            const TuckerConfig &config) {
    #pragma HLS INLINE OFF
    
    // Double-buffered storage for intermediate results
    ComputePackM_t intermediate_results[2][kInnerTilesN][kInnerTilesM][kComputeTileSizeN];
    #pragma HLS ARRAY_PARTITION variable=intermediate_results complete dim=1
    #pragma HLS ARRAY_PARTITION variable=intermediate_results complete dim=4

    // Stage 1: Mode-1 Product (Input × U1)
    Mode1_Product: {
        ProcessingElement(aIn, aOut, bIn, bOut, cOut, cIn,
                         locationN, config.batch_size, 
                         config.in_features, config.rank1);
    }

    // Stage 2: Mode-2 Product (Result × U2)
    Mode2_Product: {
        // Buffer for intermediate results from Mode-1
        ComputePackN_t mode1_buffer[kInnerTilesN];
        #pragma HLS ARRAY_PARTITION variable=mode1_buffer complete

    Mode2_Outer: for(unsigned n = 0; n < config.batch_size; n++) {
        Mode2_Inner: for(unsigned k = 0; k < config.rank1; k++) {
            #pragma HLS PIPELINE II=1
            
            // Read intermediate results
            ComputePackM_t val = cIn.read();
            
            // Compute Mode-2 product
            ComputePackM_t result;
            Mode2_Compute: for(unsigned m = 0; m < kComputeTileSizeM; m++) {
                #pragma HLS UNROLL
                result[m] = OperatorMap::Apply(val[m], bIn.read());
            }
            
            // Store result
            cOut.write(result);
        }
    }}

    // Stage 3: Mode-3 Product (Result × U3)
    Mode3_Product: {
        // Double buffering for U3 factor matrix
        ComputePackM_t U3_buffer[2][kInnerTilesM];
        #pragma HLS ARRAY_PARTITION variable=U3_buffer complete dim=1
        
    Mode3_Outer: for(unsigned n = 0; n < config.batch_size; n++) {
        Mode3_Middle: for(unsigned k = 0; k < config.rank2; k++) {
            Mode3_Inner: for(unsigned m = 0; m < config.out_features; m++) {
                #pragma HLS PIPELINE II=1
                
                // Read intermediate results
                ComputePackM_t val = cIn.read();
                
                // Compute Mode-3 product
                ComputePackM_t result;
                Mode3_Compute: for(unsigned i = 0; i < kComputeTileSizeM; i++) {
                    #pragma HLS UNROLL
                    result[i] = OperatorMap::Apply(val[i], U3_buffer[k%2][i]);
                }
                
                // Write final result
                cOut.write(result);
            }
        }
    }}

    // Core Tensor Computation
    CoreTensor_Compute: {
        // Buffer for core tensor values
        ComputePackM_t core_buffer[kInnerTilesN][kInnerTilesM];
        #pragma HLS ARRAY_PARTITION variable=core_buffer complete dim=1
        
    Core_Outer: for(unsigned r1 = 0; r1 < config.rank1; r1++) {
        Core_Middle: for(unsigned r2 = 0; r2 < config.rank2; r2++) {
            Core_Inner: for(unsigned r3 = 0; r3 < config.rank3; r3++) {
                #pragma HLS PIPELINE II=1
                
                // Read core tensor value
                ComputePackM_t core_val = bIn.read();
                
                // Compute core tensor multiplication
                ComputePackM_t result;
                Core_Compute: for(unsigned m = 0; m < kComputeTileSizeM; m++) {
                    #pragma HLS UNROLL
                    const auto intermediate = intermediate_results[r1%2][r2][r3][m];
                    result[m] = OperatorMap::Apply(intermediate, core_val[m]);
                }
                
                // Accumulate result
                cOut.write(result);
            }
        }
    }}
}