/*
 * Copyright (c) 2017-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
 
#ifndef NVCOMP_CRC32_H
#define NVCOMP_CRC32_H

#include "nvcomp.h"

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * Batched compute interface for CRC32
 *****************************************************************************/

/**
 * @brief Perform CRC32 checksum calculation asynchronously. All pointers must point to GPU
 * accessible locations.
 *
 * @param device_uncompressed_ptrs The pointers on the GPU, to uncompressed batched items.
 * This pointer must be GPU accessible.
 * @param device_uncompressed_bytes The size of each uncompressed batch item on the GPU.
 * @param batch_size The number of chunks to compress.
 * @param device_compressed_ptrs The pointers on the GPU, to the output location for
 * each CRC32 checksum (output). This pointer must be GPU accessible.
 * @param stream The CUDA stream to operate on.
 *
 * @return nvcompSuccess if successfully launched, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedCRC32Async(
    const void* const* device_uncompressed_ptrs,
    const size_t* device_uncompressed_bytes,
    size_t batch_size,
    uint32_t* device_CRC32_ptrs,
    cudaStream_t stream);


#ifdef __cplusplus
}
#endif
#endif // NVCOMP_CRC32_H