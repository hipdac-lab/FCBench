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

#ifndef NVCOMP_GZIP_H
#define NVCOMP_GZIP_H

#include "nvcomp.h"

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * Batched decompression interface for gzip
 *****************************************************************************/

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
nvcompStatus_t nvcompBatchedGzipDecompressGetTempSize(
    size_t num_chunks,
    size_t max_uncompressed_chunk_bytes,
    size_t* temp_bytes);

/**
 * @brief Get the amount of temp space required on the GPU for decompression.
 *
 * @param num_chunks The number of items in the batch.
 * @param max_uncompressed_chunk_bytes The size of the largest chunk in bytes
 * when uncompressed.
 * @param temp_bytes The amount of temporary GPU space that will be required to
 * decompress.
 * @param max_uncompressed_total_size  The total decompressed size of all the chunks. 
 * Unused in gzip.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedGzipDecompressGetTempSizeEx(
    size_t num_chunks,
    size_t max_uncompressed_chunk_bytes,
    size_t* temp_bytes,
    size_t max_uncompressed_total_size );

/**
 * @brief Perform decompression asynchronously. All pointers must be GPU
 * accessible. In the case where a chunk of compressed data is not a valid gzip
 * stream, 0 will be written for the size of the invalid chunk and
 * nvcompStatusCannotDecompress will be flagged for that chunk.
 *
 * @param device_compressed_ptrs The pointers on the GPU, to the compressed
 * chunks.
 * @param device_compressed_bytes The size of each compressed chunk on the GPU.
 * This pointer must be GPU accessible.
 * @param device_uncompressed_bytes The decompressed buffer size. This is needed
 * to prevent OOB accesses. This pointer must be GPU accessible.
 * @param device_actual_uncompressed_bytes The actual calculated decompressed
 * size of each chunk. Can be nullptr if desired, 
 * in which case the actual_uncompressed_bytes is not reported.
 * @param batch_size The number of batch items.
 * @param device_temp_ptr The temporary GPU space.
 * @param temp_bytes The size of the temporary GPU space.
 * @param device_uncompressed_ptrs The pointers on the GPU, to where to
 * uncompress each chunk (output).
 * @param device_statuses The status for each chunk of whether it was
 * decompressed or not. Can be nullptr if desired, 
 * in which case error status is not reported.
 * @param stream The CUDA stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedGzipDecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const device_temp_ptr,
    size_t temp_bytes,
    void* const* device_uncompressed_ptrs,
    nvcompStatus_t* device_statuses,
    cudaStream_t stream);

/**
 * @brief Calculates the decompressed size of each chunk asynchronously. This is
 * needed when we do not know the expected output size. All pointers must be GPU
 * accessible. Note, if the stream is corrupt, the sizes will be garbage.
 *
 * @param device_compress_ptrs The compressed chunks of data. List of pointers
 * must be GPU accessible along with each chunk.
 * @param device_compressed_bytes The size of each compressed chunk. Must be GPU
 * accessible.
 * @param device_uncompressed_bytes The calculated decompressed size of each
 * chunk. Must be GPU accessible.
 * @param batch_size The number of chunks
 * @param stream The CUDA stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedGzipGetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    cudaStream_t stream);
#ifdef __cplusplus
}
#endif

#endif // NVCOMP_GZIP_H
