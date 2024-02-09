#pragma once

/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "nvcomp.h"

#include <memory>
#include <vector>
#include <functional>
#include <stdint.h>

namespace nvcomp {

/******************************************************************************
 * CLASSES ********************************************************************
 *****************************************************************************/


typedef std::function<void*(size_t)> AllocFn_t;
typedef std::function<void(void*, size_t)> DeAllocFn_t;

/**
 * Internal memory pool used for compression / decompression configs
 */
template<typename T>
struct PinnedPtrPool;

/**
 * @brief Config used to aggregate information about the compression of a particular buffer.
 * 
 * Contains a "PinnedPtrHandle" to an nvcompStatus. After the compression is complete,
 * the user can check the result status which resides in pinned host memory.
 */
struct CompressionConfig {

private: // pimpl
  struct CompressionConfigImpl;
  std::shared_ptr<CompressionConfigImpl> impl;

public: // API
  size_t uncompressed_buffer_size;
  size_t max_compressed_buffer_size;
  size_t num_chunks;
  bool compute_checksums;

  /**
   * @brief Construct the config given an nvcompStatus_t memory pool
   */
  CompressionConfig(PinnedPtrPool<nvcompStatus_t>& pool, size_t uncompressed_buffer_size);

  /**
   * @brief Get the raw nvcompStatus_t*
   */
  nvcompStatus_t* get_status() const;
  
  CompressionConfig(CompressionConfig&& other);
  CompressionConfig(const CompressionConfig& other);
  CompressionConfig& operator=(CompressionConfig&& other);
  CompressionConfig& operator=(const CompressionConfig& other);

  ~CompressionConfig();
};

/**
 * @brief Config used to aggregate information about a particular decompression.
 * 
 * Contains a "PinnedPtrHandle" to an nvcompStatus. After the decompression is complete,
 * the user can check the result status which resides in pinned host memory.
 */
struct DecompressionConfig {

private: // pimpl to hide pool implementation
  struct DecompressionConfigImpl;
  std::shared_ptr<DecompressionConfigImpl> impl;

public: // API
  size_t decomp_data_size;
  size_t num_chunks;
  bool checksums_present;

  /**
   * @brief Construct the config given an nvcompStatus_t memory pool
   */
  DecompressionConfig(PinnedPtrPool<nvcompStatus_t>& pool);

  /**
   * @brief Get the nvcompStatus_t*
   */
  nvcompStatus_t* get_status() const;

  DecompressionConfig(DecompressionConfig&& other);
  DecompressionConfig(const DecompressionConfig& other);
  DecompressionConfig& operator=(DecompressionConfig&& other);
  DecompressionConfig& operator=(const DecompressionConfig& other);

  ~DecompressionConfig();
};

/**
 * @brief Abstract base class that defines the nvCOMP high level interface
 */
struct nvcompManagerBase {
  /**
   * @brief Configure the compression. 
   *
   * This routine computes the size of the required result buffer. The result config also
   * contains the nvcompStatus* that allows error checking. Synchronizes the device (cudaMemcpy)
   * 
   * @param decomp_buffer_size The uncompressed input data size.
   * \return comp_config Result
   */
  virtual CompressionConfig configure_compression(
    const size_t decomp_buffer_size) = 0;

  /**
   * @brief Perform compression asynchronously.
   *
   * @param decomp_buffer The uncompressed input data (GPU accessible).
   * @param comp_buffer The location to output the compressed data to (GPU accessible).
   * @param comp_config Resulted from configure_compression for this decomp_buffer.
   */
  virtual void compress(
      const uint8_t* decomp_buffer, 
      uint8_t* comp_buffer,
      const CompressionConfig& comp_config) = 0;

  /**
   * @brief Configure the decompression using a compressed buffer. 
   *
   * Synchronizes the user stream. 
   * 
   * In the base case, this only computes the size of the decompressed buffer from the compressed buffer header. 
   * 
   * @param comp_buffer The compressed input data (GPU accessible).
   * \return decomp_config Result
   */
  virtual DecompressionConfig configure_decompression(
    const uint8_t* comp_buffer) = 0;

  /**
   * @brief Configure the decompression using a CompressionConfig object. 
   *
   * Does not synchronize the user stream. 
   * 
   * In the base case, this only computes the size of the decompressed buffer from the compressed buffer header. 
   * 
   * @param comp_config The config used to compress a buffer
   * \return decomp_config Result
   */
  virtual DecompressionConfig configure_decompression(
    const CompressionConfig& comp_config) = 0;

  /**
   * @brief Perform decompression asynchronously.
   *
   * @param decomp_buffer The location to output the decompressed data to (GPU accessible).
   * @param comp_buffer The compressed input data (GPU accessible).
   * @param decomp_config Resulted from configure_decompression given this decomp_buffer_size.
   * Contains nvcompStatus* in CPU/GPU-accessible memory to allow error checking.
   */
  virtual void decompress(
      uint8_t* decomp_buffer, 
      const uint8_t* comp_buffer,
      const DecompressionConfig& decomp_config) = 0;
  
  /**
   * @brief Allows the user to provide a function for allocating / deallocating memory
   * 
   * The HLIF manager requires scratch memory to perform its operations. 
   * By default, it will use internal allocators which make use of 
   * cudaMallocAsync / cudaFreeAsync
   * The user can override the allocation functions with this API. 
   * The required signatures are
   *   void alloc_fn(size_t alloc_size)
   * and
   *   void dealloc_fn(size_t alloc_size)
   * 
   * @param alloc_fn The host function to use to alloc a new scratch result buffer
   * @param dealloc_fn The host function to use to dealloc a scratch result buffer
   * 
   */
  virtual void set_scratch_allocators(const AllocFn_t& alloc_fn, const DeAllocFn_t& dealloc_fn) = 0;
  
  /** 
   * @brief Computes the compressed output size of a given buffer 
   * 
   * Synchronously copies the size of the compressed buffer to a stack variable for return.
   * 
   * @param comp_buffer The start pointer of the compressed buffer to assess.
   * \return Size of the compressed buffer
   */ 
  virtual size_t get_compressed_output_size(uint8_t* comp_buffer) = 0;

  virtual ~nvcompManagerBase() = default;
};

struct PimplManager : nvcompManagerBase {

protected:
  std::unique_ptr<nvcompManagerBase> impl;

public:
  virtual ~PimplManager() {}

  PimplManager() 
    : impl(nullptr)
  {}

  PimplManager(const PimplManager&) = delete;
  PimplManager& operator=(const PimplManager&) = delete;

  virtual CompressionConfig configure_compression(const size_t decomp_buffer_size) 
  {
    return impl->configure_compression(decomp_buffer_size);
  }

  virtual void compress(
      const uint8_t* decomp_buffer, 
      uint8_t* comp_buffer,
      const CompressionConfig& comp_config)
  {
    return impl->compress(
        decomp_buffer,
        comp_buffer,
        comp_config);
  }

  virtual DecompressionConfig configure_decompression(const uint8_t* comp_buffer) 
  {
    return impl->configure_decompression(comp_buffer);
  }

  virtual DecompressionConfig configure_decompression(const CompressionConfig& comp_config)
  {
    return impl->configure_decompression(comp_config);
  }

  virtual void decompress(
      uint8_t* decomp_buffer, 
      const uint8_t* comp_buffer,
      const DecompressionConfig& decomp_config)
  {
    return impl->decompress(decomp_buffer, comp_buffer, decomp_config);
  }
 
  virtual void set_scratch_allocators(const AllocFn_t& alloc_fn, const DeAllocFn_t& dealloc_fn)
  {
    return impl->set_scratch_allocators(alloc_fn, dealloc_fn);
  }

  virtual size_t get_compressed_output_size(uint8_t* comp_buffer)
  {
    return impl->get_compressed_output_size(comp_buffer);
  }
};

} // namespace nvcomp
