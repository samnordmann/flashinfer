/*
 * Copyright (c) 2026, FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */

#include "flashinfer/gemm/bf16_gemm_nvfp4.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer::gemm {

namespace {

constexpr int64_t kHiddenIn = 8192;
constexpr int64_t kHiddenOut = 2048;
constexpr int64_t kMaxTokens = 256;

void check_tensor_2d(TensorView tensor, int64_t rows, int64_t columns, DLDataType dtype,
                     char const* name) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 2) << name << " must be 2D";
  TVM_FFI_ICHECK_EQ(tensor.size(0), rows) << name << " row mismatch";
  TVM_FFI_ICHECK_EQ(tensor.size(1), columns) << name << " column mismatch";
  TVM_FFI_ICHECK_EQ(tensor.strides()[1], 1) << name << " must be row-major";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(tensor.dtype()), encode_dlpack_dtype(dtype))
      << name << " dtype mismatch";
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLCUDA) << name << " must be on CUDA";
}

}  // namespace

void bf16_gemm_nvfp4(TensorView input, TensorView weight, TensorView output,
                     TensorView output_scale, TensorView global_scale, bool launch_with_pdl) {
  int64_t const num_tokens = input.size(0);
  TVM_FFI_ICHECK(num_tokens >= 1 && num_tokens <= kMaxTokens)
      << "num_tokens must be in [1, " << kMaxTokens << "]";
  check_tensor_2d(input, num_tokens, kHiddenIn, dl_bfloat16, "input");
  check_tensor_2d(weight, kHiddenOut, kHiddenIn, dl_bfloat16, "weight");
  check_tensor_2d(output, num_tokens, kHiddenOut / 2, dl_uint8, "output");
  check_tensor_2d(output_scale, num_tokens, kHiddenOut / 16, dl_float8_e4m3fn, "output_scale");
  TVM_FFI_ICHECK_EQ(global_scale.ndim(), 1) << "global_scale must be 1D";
  TVM_FFI_ICHECK_EQ(global_scale.numel(), 1) << "global_scale must have one element";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(global_scale.dtype()), float32_code)
      << "global_scale must be float32";
  TVM_FFI_ICHECK_EQ(global_scale.device().device_type, kDLCUDA) << "global_scale must be on CUDA";

  cudaStream_t stream = get_stream(input.device());
  cudaError_t status;
  if (num_tokens <= 8) {
    if (launch_with_pdl) {
      status = invokeBf16GemmNvfp4<kHiddenIn, kHiddenOut, 8, true>(
          static_cast<uint8_t*>(output.data_ptr()), static_cast<uint8_t*>(output_scale.data_ptr()),
          static_cast<bf16_t const*>(input.data_ptr()),
          static_cast<bf16_t const*>(weight.data_ptr()), num_tokens,
          static_cast<float const*>(global_scale.data_ptr()), stream);
    } else {
      status = invokeBf16GemmNvfp4<kHiddenIn, kHiddenOut, 8, false>(
          static_cast<uint8_t*>(output.data_ptr()), static_cast<uint8_t*>(output_scale.data_ptr()),
          static_cast<bf16_t const*>(input.data_ptr()),
          static_cast<bf16_t const*>(weight.data_ptr()), num_tokens,
          static_cast<float const*>(global_scale.data_ptr()), stream);
    }
  } else if (launch_with_pdl) {
    status = invokeBf16GemmNvfp4<kHiddenIn, kHiddenOut, 16, true>(
        static_cast<uint8_t*>(output.data_ptr()), static_cast<uint8_t*>(output_scale.data_ptr()),
        static_cast<bf16_t const*>(input.data_ptr()), static_cast<bf16_t const*>(weight.data_ptr()),
        num_tokens, static_cast<float const*>(global_scale.data_ptr()), stream);
  } else {
    status = invokeBf16GemmNvfp4<kHiddenIn, kHiddenOut, 16, false>(
        static_cast<uint8_t*>(output.data_ptr()), static_cast<uint8_t*>(output_scale.data_ptr()),
        static_cast<bf16_t const*>(input.data_ptr()), static_cast<bf16_t const*>(weight.data_ptr()),
        num_tokens, static_cast<float const*>(global_scale.data_ptr()), stream);
  }
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "bf16_gemm_nvfp4 launch failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer::gemm

TVM_FFI_DLL_EXPORT_TYPED_FUNC(bf16_gemm_nvfp4, flashinfer::gemm::bf16_gemm_nvfp4);
