# Source this before any run. The NGC base image exports TRITON_*_PATH vars
# pointing at the system CUDA 13.0 toolchain; triton 3.4 (bundled with torch
# 2.8) does not understand ptxas 13.0 ("Triton only support CUDA 10.0 or
# higher, but got CUDA version: 13.0"). Unset them so triton uses its own
# bundled CUDA 12.8 tools.
unset TRITON_PTXAS_PATH TRITON_CUDACRT_PATH TRITON_CUDART_PATH \
      TRITON_CUOBJDUMP_PATH TRITON_NVDISASM_PATH \
      TRITON_CUPTI_LIB_PATH TRITON_CUPTI_INCLUDE_PATH
