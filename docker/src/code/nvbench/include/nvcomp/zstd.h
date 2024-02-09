/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVCOMP_Zstd_H
#define NVCOMP_Zstd_H

#include "nvcomp.h"

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
  int reserved;
} nvcompBatchedZstdOpts_t;

static const nvcompBatchedZstdOpts_t nvcompBatchedZstdDefaultOpts = {0};

// Set this to 16 MB for now. In theory this could go up to 2 GB, but 
// we only want to provide that if a user has a really good reason for it because
// decompression performance will be really awful.
const size_t nvcompZstdCompressionMaxAllowedChunkSize = 1 << 24;

/**
 * This is the minimum alignment required for void type CUDA memory buffers
 * passed to compression or decompression functions.  Typed memory buffers must
 * still be aligned to their type's size, e.g. 8 bytes for size_t.
 */
const size_t nvcompZstdRequiredAlignment = 8;

/**
 * @brief Get the amount of temp space required on the GPU for decompression.
 *
 * @param num_chunks The number of items in the batch.
 * @param max_uncompressed_chunk_size The size of the largest chunk when uncompressed.
 * @param temp_bytes The amount of temporary GPU space that will be required to
 * decompress.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedZstdDecompressGetTempSize(
    size_t num_chunks, size_t max_uncompressed_chunk_size, size_t* temp_bytes);


/**
 * @brief Get the amount of temp space required on the GPU for decompression.
 *
 * @param num_chunks The number of items in the batch.
 * @param max_uncompressed_chunk_size The size of the largest chunk when uncompressed.
 * @param temp_bytes The amount of temporary GPU space that will be required to
 * decompress.
 * @param max_uncompressed_total_size  The total decompressed size of all the chunks.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedZstdDecompressGetTempSizeEx(
    size_t num_chunks, size_t max_uncompressed_chunk_size, size_t* temp_bytes, size_t max_uncompressed_total_size );

/**
 * @brief Compute uncompressed sizes.
 *
 * @param device_compresed_ptrs The pointers on the GPU, to the compressed chunks.
 * @param device_compressed_bytes The size of each compressed chunk on the GPU.
 * @param device_uncompressed_bytes The actual size of each uncompressed chunk.
 * @param batch_size The number of chunks in the batch.
 * @param stream The CUDA stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedZstdGetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    cudaStream_t stream);

/**
 * @brief Perform decompression.
 *
 * @param device_compresed_ptrs The pointers on the GPU, to the compressed chunks.
 * @param device_compressed_bytes The size of each compressed chunk on the GPU.
 * @param device_uncompressed_bytes The size of each device_uncompressed_ptrs[i] buffer.
 * @param device_actual_uncompressed_bytes The actual size of each uncompressed chunk
 * @param batch_size The number of chunks in the batch.
 * @param device_temp_ptr The temporary GPU space, could be NULL in case temprorary space is not needed.
 * @param temp_bytes The size of the temporary GPU space.
 * @param device_uncompressed_ptrs The pointers on the GPU, to where to uncompress each chunk (output).
 * @param device_statuses The pointers on the GPU, to where to uncompress each chunk (output).
 * @param stream The CUDA stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedZstdDecompressAsync(
    const void* const* device_compresed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const device_temp_ptr,
    const size_t temp_bytes,
    void* const* device_uncompressed_ptrs,
    nvcompStatus_t* device_statuses,
    cudaStream_t stream);

/**
 * @brief Perform compression asynchronously. All pointers must point to GPU
 * accessible locations. The individual chunk size must not exceed
 * 64 KB. For best performance, a chunk size of 64 KB is
 * recommended.
 *
 * @param device_uncompressed_ptrs The pointers on the GPU, to uncompressed batched items.
 * This pointer must be GPU accessible.
 * @param device_uncompressed_bytes The size of each uncompressed batch item on the GPU.
 * Each chunk size MUST be a multiple of the size of the data type specified by
 * format_opts.data_type, else this may crash or produce invalid output.
 * @param max_uncompressed_chunk_bytes The maximum size in bytes of the largest
 * chunk in the batch. This parameter is currently unused, so if it is not set
 * with the maximum size, it should be set to zero. If a future version makes
 * use of it, it will return an error if it is set to zero.
 * @param batch_size The number of chunks to compress.
 * @param device_temp_ptr The temporary GPU workspace.
 * @param temp_bytes The size of the temporary GPU workspace. 
 * @param device_compressed_ptrs The pointers on the GPU, to the output location for
 * each compressed batch item (output). This pointer must be GPU accessible.
 * @param device_compressed_bytes The compressed size of each chunk on the GPU
 * (output). This pointer must be GPU accessible.
 * @param format_opts The Zstd compression options to use. Currently empty.
 * @param stream The CUDA stream to operate on.
 *
 * @return nvcompSuccess if successfully launched, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedZstdCompressAsync(
    const void* const* const device_uncompressed_ptrs,
    const size_t* const device_uncompressed_bytes,
    const size_t max_uncompressed_chunk_bytes,
    const size_t batch_size,
    void* const device_temp_ptr,
    const size_t temp_bytes,
    void* const* const device_compressed_ptrs,
    size_t* const device_compressed_bytes,
    const nvcompBatchedZstdOpts_t format_opts,
    cudaStream_t stream);

/**
 * @brief Get the maximum size any chunk could compress to in the batch. That
 * is, the minimum amount of output memory required to be given
 * nvcompBatchedZstdCompressAsync() for each batch item.
 *
 * Chunk size must not exceed
 * 64 KB bytes. For best performance, a chunk size of 64 KB is
 * recommended.
 *
 * @param max_uncompressed_chunk_bytes The maximum size of a chunk in the batch.
 * @param format_opts The Zstd compression options to use. Currently empty.
 * @param max_compressed_byes The maximum compressed size of the largest chunk
 * (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedZstdCompressGetMaxOutputChunkSize(
    const size_t max_uncompressed_chunk_bytes,
    const nvcompBatchedZstdOpts_t,
    size_t* const max_compressed_size);

/**
 * @brief Get temporary space required for compression.
 *
 * Chunk size must not exceed
 * 64 KB. For best performance, a chunk size of 64 KB is
 * recommended.
 *
 * @param batch_size The number of items in the batch.
 * @param max_uncompressed_chunk_bytes The maximum size of a chunk in the
 * batch.
 * @param format_opts The ZSTD compression options to use -- currently empty
 * @param temp_bytes The size of the required GPU workspace for compression
 * (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedZstdCompressGetTempSize(
    const size_t batch_size,
    const size_t max_uncompressed_chunk_bytes,
    const nvcompBatchedZstdOpts_t format_opts,
    size_t* temp_bytes);

/**
 * @brief Get temporary space required for compression.
 *
 * Chunk size must not exceed
 * 16 MB. For best performance, a chunk size of 64 KB is
 * recommended.
 * 
 * This extended API is useful for cases where chunk sizes aren't uniform in the batch
 * I.e. in the regular API, if all but 1 chunk is 64 KB, but 1 chunk is 16 MB, the temporary space
 * computed is based on 16 MB * num_chunks.
 *
 * @param batch_size The number of items in the batch.
 * @param max_uncompressed_chunk_bytes The maximum size of a chunk in the
 * batch.
 * @param format_opts The ZSTD compression options to use -- currently empty
 * @param temp_bytes The size of the required GPU workspace for compression
 * (output).
 * @param max_total_uncompressed_bytes Upper bound on the total uncompressed size of all
 * chunks
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedZstdCompressGetTempSizeEx(
    const size_t batch_size,
    const size_t max_uncompressed_chunk_bytes,
    const nvcompBatchedZstdOpts_t format_opts,
    size_t* temp_bytes,
    const size_t max_total_uncompressed_bytes);    

#ifdef __cplusplus
}
#endif

#endif
