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

#ifndef NVCOMP_ANS_H
#define NVCOMP_ANS_H

#include "nvcomp.h"

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * Batched compression/decompression interface for ANS
 *****************************************************************************/

typedef enum nvcompANSType_t {
  nvcomp_rANS,
} nvcompANSType_t;

typedef struct
{
  nvcompANSType_t type;
} nvcompBatchedANSOpts_t;

static const nvcompBatchedANSOpts_t nvcompBatchedANSDefaultOpts = {nvcomp_rANS};

const size_t nvcompANSCompressionMaxAllowedChunkSize = 1 << 24;

/**
 * This is the minimum alignment required for void type CUDA memory buffers
 * passed to compression or decompression functions.  Typed memory buffers must
 * still be aligned to their type's size, e.g. 8 bytes for size_t.
 */
const size_t nvcompANSRequiredAlignment = 8;

/**
 * @brief Get temporary space required for compression.
 *
 * @param batch_size The number of items in the batch.
 * @param max_uncompressed_chunk_bytes The maximum size of a chunk in the batch.
 * @param format_opts Compression options.
 * @param temp_bytes The size of the required GPU workspace for compression
 * (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedANSCompressGetTempSize(
    size_t batch_size,
    size_t max_uncompressed_chunk_bytes,
    nvcompBatchedANSOpts_t format_opts,
    size_t* temp_bytes);

/**
 * @brief Get temporary space required for compression.
 *
 * @param batch_size The number of items in the batch.
 * @param max_uncompressed_chunk_bytes The maximum size of a chunk in the batch.
 * @param format_opts Compression options.
 * @param temp_bytes The size of the required GPU workspace for compression
 * (output).
 * @param max_total_uncompressed_bytes Upper bound on the total uncompressed size of all
 * chunks
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedANSCompressGetTempSizeEx(
    size_t batch_size,
    size_t max_uncompressed_chunk_bytes,
    nvcompBatchedANSOpts_t format_opts,
    size_t* temp_bytes,
    const size_t max_total_uncompressed_bytes);

/**
 * @brief Get the maximum size any chunk could compress to in the batch. That
 * is, the minimum amount of output memory required to be given
 * nvcompBatched[R|T|H]ANSCompressAsync() for each batch item.
 *
 * @param max_chunk_size The maximum size of a chunk in the batch.
 * @param format_opts Compression options.
 * @param max_compressed_size The maximum compressed size of the largest chunk
 * (output).
 *
 * @return The nvcompSuccess unless there is an error.
 */
nvcompStatus_t nvcompBatchedANSCompressGetMaxOutputChunkSize(
    size_t max_chunk_size,
    nvcompBatchedANSOpts_t format_opts,
    size_t* max_compressed_size);

/**
 * @brief Perform compression.
 *
 * The caller is responsible for passing device_compressed_bytes of size
 * sufficient to hold compressed data
 *
 * @param device_uncompressed_ptrs The pointers on the GPU, to uncompressed batched items.
 * Each pointer must be aligned to a 4-byte boundary.
 * @param device_uncompressed_bytes The size of each uncompressed batch item on the GPU.
 * @param max_uncompressed_chunk_bytes The size of the largest uncompressed chunk.
 * @param batch_size The number of batch items.
 * @param device_temp_ptr The temporary GPU workspace, could be NULL in case temprorary space is not needed.
 * @param temp_bytes The size of the temporary GPU workspace.
 * @param device_compressed_ptrs The pointers on the GPU, to the output location for each compressed batch item (output).
 * Each pointer must be aligned to an 8-byte boundary.
 * @param device_compressed_bytes The compressed size of each chunk on the GPU (output).
 * @param format_opts Compression options.
 * @param stream The stream to operate on.
 *
 * @return nvcompSuccess if successfully launched, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedANSCompressAsync(
    const void* const* device_uncompressed_ptrs,
    const size_t* device_uncompressed_bytes,
    size_t max_uncompressed_chunk_bytes,
    size_t batch_size,
    void* device_temp_ptr,
    size_t temp_bytes,
    void* const* device_compressed_ptrs,
    size_t* device_compressed_bytes,
    nvcompBatchedANSOpts_t format_opts,
    cudaStream_t stream);

/**
 * @brief Get the amount of temp space required on the GPU for decompression.
 *
 * @param num_chunks The number of items in the batch.
 * @param max_uncompressed_chunk_bytes The size of the largest chunk in bytes
 * when uncompressed.
 * @param temp_bytes The amount of temporary GPU space that will be required to
 * decompress.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedANSDecompressGetTempSize(
    size_t num_chunks, size_t max_uncompressed_chunk_bytes, size_t* temp_bytes);

/**
 * @brief Get the amount of temp space required on the GPU for decompression with extra total size argument.
 * @param max_uncompressed_total_size  The total decompressed size of all the chunks. Unused in ANS.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedANSDecompressGetTempSizeEx(
    size_t num_chunks, size_t max_uncompressed_chunk_bytes, size_t* temp_bytes, size_t max_uncompressed_total_size );    

/**
 * @brief Compute uncompressed sizes.
 *
 * @param device_compresed_ptrs The pointers on the GPU, to the compressed chunks.
 * @param device_compressed_bytes The size of each compressed chunk on the GPU.
 * @param device_uncompressed_bytes The actual size of each uncompressed chunk.
 * @param batch_size The number of batch items.
 * @param stream The stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedANSGetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    cudaStream_t stream);

/**
 * @brief Perform decompression.
 *
 * @param device_compresed_ptrs The pointers on the GPU, to the compressed chunks. 
 * Each pointer must be aligned to an 8-byte boundary.
 * @param device_compressed_bytes The size of each compressed chunk on the GPU.
 * @param device_uncompressed_bytes The size of each device_uncompressed_ptrs[i] buffer.
 * @param device_actual_uncompressed_bytes The actual size of each uncompressed chunk
 * @param batch_size The number of batch items.
 * @param device_temp_ptr The temporary GPU space, could be NULL in case temporary space is not needed.
 * @param temp_bytes The size of the temporary GPU space.
 * @param device_uncompressed_ptrs The pointers on the GPU, to where to uncompress each chunk (output).
 * @param device_statuses The pointers on the GPU, to where to uncompress each chunk (output).
 * @param stream The stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedANSDecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const device_temp_ptr,
    const size_t temp_bytes,
    void* const* device_uncompressed_ptrs,
    nvcompStatus_t* device_statuses,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
