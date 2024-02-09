#pragma once

#include "operators.hpp"

#include "detail/ans/compress_device.cuh"
#include "detail/ans/decompress_device.cuh"

namespace nvcomp::device
{
namespace detail
{

template <typename CG, nvcomp_datatype DT, nvcomp_algo A>
class Compress
{
public:
  static_assert(
      dependent_false<CG>::value,
      "Unsupported compressor config detected in compress(). Please consult "
      "the selected compression algorithm's documentation for a description of "
      "the accepted configurations.");

  __device__ void execute(
      const void* uncomp_chunk,
      void* comp_chunk,
      const size_t uncomp_chunk_size,
      size_t* comp_chunk_size,
      uint8_t* shared_buffer,
      uint8_t* tmp_buffer,
      size_t max_uncomp_chunk_size,
      CG& group);
};

template <typename CG, nvcomp_datatype DT, nvcomp_algo A>
class Decompress
{
public:
  static_assert(
      dependent_false<CG>::value,
      "Unsupported compressor config detected in decompress(). Please consult "
      "the selected compression algorithm's documentation for a description of "
      "the accepted configurations.");

  __device__ void execute(
      const void* comp_chunk,
      void* uncomp_chunk,
      uint8_t* shared_buffer,
      uint8_t* tmp_buf,
      CG& group);

  __device__ void execute(
      const void* comp_chunk,
      void* uncomp_chunk,
      const size_t comp_chunk_size,
      const size_t decomp_chunk_size,
      uint8_t* shared_buffer,
      uint8_t* tmp_buf,
      CG& group);
};

template <nvcomp_algo A, nvcomp_direction D>
class ShmemSizeBlock
{
public:
  static_assert(
      dependent_false_algo<A>::value,
      "Unsupported compressor config detected in shmem_size_block(). Please "
      "consult the selected compression algorithm's documentation for a "
      "description of the accepted configurations.");

  static __host__ __device__ constexpr size_t
  execute(int nvcomp_warps_per_block)
  {
    // This class will be overridden, ret val doesn't matter
    return 1;
  }
};

template <nvcomp_algo A, nvcomp_direction D>
class ShmemSizeGroup
{
public:
  static_assert(
      dependent_false_algo<A>::value,
      "Unsupported compressor config detected in shmem_size_local(). Please "
      "consult the selected compression algorithm's documentation for a "
      "description of the accepted configurations.");

  static __host__ __device__ constexpr size_t execute()
  {
    // This class will be overridden, ret val doesn't matter
    return 1;
  }
};

template <nvcomp_grouptype G, nvcomp_algo A, nvcomp_direction D>
class TmpSizeTotal
{
public:
  static_assert(
      dependent_false_algo<A>::value,
      "Unsupported compressor config detected in tmp_size_total(). Please "
      "consult the selected compression algorithm's documentation for a "
      "description of the accepted configurations.");

  static size_t constexpr execute(
      size_t max_uncomp_chunk_size, nvcomp_datatype dt, size_t num_warps)
  {
    // This class will be overridden, ret val doesn't matter
    return 0;
  }
};

template <nvcomp_grouptype G, nvcomp_algo A, nvcomp_direction D>
class TmpSizeGroup
{
public:
  static_assert(
      dependent_false_algo<A>::value,
      "Unsupported compressor config detected in tmp_size_local(). Please "
      "consult the selected compression algorithm's documentation for a "
      "description of the accepted configurations.");

  static size_t constexpr __host__ __device__
  execute(size_t max_uncomp_chunk_size, nvcomp_datatype dt)
  {
    // This class will be overridden, ret val doesn't matter
    return 0;
  }
};

template <nvcomp_grouptype G, nvcomp_algo A, nvcomp_direction D>
class ShmemAlignment
{
public:
  static_assert(
      dependent_false_algo<A>::value,
      "Unsupported compressor config detected in tmp_size_local(). Please "
      "consult the selected compression algorithm's documentation for a "
      "description of the accepted configurations.");

  static size_t constexpr __host__ __device__
  execute()
  {
    // This class will be overridden, ret val doesn't matter
    return 0;
  }
};

template <nvcomp_algo A>
class MaxCompChunkSize
{
public:
  static_assert(
      dependent_false_algo<A>::value,
      "Unsupported compressor config detected in max_comp_chunk_size(). Please "
      "consult the selected compression algorithm's documentation for a "
      "description of the accepted configurations.");

  static __host__
      __device__ size_t constexpr execute(size_t max_uncomp_chunk_size)
   {
    // This class will be overridden, ret val doesn't matter
    return 0;
  }
};

} // namespace detail
} // namespace nvcomp::device