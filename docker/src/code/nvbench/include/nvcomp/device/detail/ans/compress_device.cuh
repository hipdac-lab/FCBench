#pragma once
#include <stdint.h>
#include "../../operators.hpp"
#include "../../backend_common.hpp"

namespace nvcomp::device {
namespace detail {

namespace cg = cooperative_groups;

template <>
class ShmemSizeBlock<nvcomp_algo::ans, nvcomp_direction::compress> 
{
public:
  static __host__ __device__ constexpr size_t execute(int warps_per_block)
  {
    constexpr size_t SHMEM_PER_WARP = 3584;
    return SHMEM_PER_WARP * warps_per_block;
  }
};

template <>
class ShmemSizeGroup<nvcomp_algo::ans, nvcomp_direction::compress> 
{
public:
  static __host__ __device__ constexpr size_t execute()
  {
    constexpr size_t SHMEM_PER_WARP = 3584;
    return SHMEM_PER_WARP;
  }
};

template <>
class ShmemAlignment<nvcomp_grouptype::warp, nvcomp_algo::ans, nvcomp_direction::compress>
{
public:
  static __host__ __device__ constexpr size_t execute()
  {
    constexpr int SHMEM_ALIGNMENT = 16;
    return SHMEM_ALIGNMENT;
  }
};

template <>
class TmpSizeTotal<nvcomp_grouptype::warp, nvcomp_algo::ans, nvcomp_direction::compress>
{
public:
  static constexpr size_t execute(
    size_t max_uncomp_chunk_size, 
    nvcomp_datatype dt,
    size_t num_warps)
  {
    // Here, we need half of the chunk to store all the exponents
    const size_t tmp_size_per_warp = max_uncomp_chunk_size / 2 + 1024;

    switch(dt) {
      case nvcomp_datatype::uint8:
        return 0;
      case nvcomp_datatype::float16:
        return num_warps * tmp_size_per_warp;
      default:
        return 0;
    }
  }
};

template <>
class TmpSizeGroup<nvcomp_grouptype::warp, nvcomp_algo::ans, nvcomp_direction::compress>
{
public:
  static __host__ __device__ constexpr size_t execute(
    size_t max_uncomp_chunk_size,
    nvcomp_datatype dt)
  {
    // Here, we need half of the chunk to store all the exponents
    const size_t tmp_size_per_warp = max_uncomp_chunk_size / 2 + 1024;

    switch(dt) {
      case nvcomp_datatype::uint8:
        return 0;
      case nvcomp_datatype::float16:
        return tmp_size_per_warp;
      default:
        return 0;
    }
  }
};

template <>
class MaxCompChunkSize<nvcomp_algo::ans>
{
public:
  static __host__ __device__ constexpr size_t execute(size_t max_uncomp_chunk_size)
  {
    /* Assuming a tablelog of 10, a maximum average of 10 bits can be written out for each
     * 8-bit symbol. So the maximum overhead is about 10/8 = 1.2. We add a small safety overhead
     * 0.1 and a constant offset of 768, which is the largest header we can have.
     */
    return 1.3 * max_uncomp_chunk_size + 768;
  }
};

#define gen_compress(data_type)       \
  template <>                         \
  class Compress<                     \
    WarpGroup,                        \
    data_type,                        \
    nvcomp_algo::ans> {               \
      public:                         \
      __device__ void execute(        \
      const void* uncomp_chunk,       \
      void* comp_chunk,               \
      const size_t uncomp_chunk_size, \
      size_t* comp_chunk_size,        \
      uint8_t* shared_buffer,         \
      uint8_t* tmp_buffer,            \
      size_t max_uncomp_chunk_size,   \
      WarpGroup& group);              \
  }

gen_compress(nvcomp_datatype::uint8);
gen_compress(nvcomp_datatype::float16);

}
}
