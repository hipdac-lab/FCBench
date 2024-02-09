#pragma once

#include "backend_api.hpp"
#include "user_api_detail.hpp"
#include <cstdint>

namespace nvcomp::device
{
namespace detail
{

/**
 * An object of type `nvcomp_device_execution` is to be constructed in order to use the 
 * device-side API. To do this, start by declaring a user-defined type using the following
 * operators:
 * 
 * `Direction()`
 * `Algo()`
 * `Warp()/Block()/MultiBlock()` (choose one)
 * `Datatype()`
 * `MaxUncompChunkSize()`
 *
 * See the description of these operators in `operators.hpp`
 * 
 * The custom type is created using the `+` between the available operators. The resulting type
 * is then used to initialize an object of type `nvcomp_device_execution` and device API calls
 * can be made from that object.
 * 
 * Example using ANS compression:
 * ```
 * using ans_base_type
 *   decltype(Warp() +
 *            Algo<nvcomp_algo::ans>() + 
 *            Datatype<nvcomp_datatype::float16>());
 * 
 * using ans_compressor_type = 
 *   decltype(ans_base_type() + 
 *            Direction<nvcomp_direction::compress>() + 
 *            MaxUncompChunkSize<max_chunk_size>());
 *
 * constexpr size_t tmp_size = ans_compressor_type().tmp_size_total(total_num_warps);
 * constexpr size_t shared_size = ans_compressor_type().shmem_size_block(num_warps_per_block);
 * constexpr size_t shmem_alignment = ans_compressor_type().shmem_alignment();
 * __shared__ __align__(shmem_alignment) uint8_t shared_buffer[shared_size];
 * ...
 * ```
 *
 */
template <class... Operators>
class nvcomp_device_execution
{
  using algo = typename get_operator<nvcomp_operator::algo, Operators...>::type;
  using direction =
      typename get_operator<nvcomp_operator::direction, Operators...>::type;
  using max_uncomp_chunk_size = typename get_operator<
      nvcomp_operator::max_uncomp_chunk_size,
      Operators...>::type;
  using data_type =
      typename get_operator<nvcomp_operator::datatype, Operators...>::type;
  using group_type =
      typename get_operator<nvcomp_operator::grouptype, Operators...>::type;

public:
  /** @brief Compresses a contiguous buffer of data
   *
   * @param[in] input_chunk The to-be-compressed chunk
   * @param[in] output_chunk The resulting compressed chunk
   * @param[in] input_size The size in bytes of the to-be-comrpessed chunk
   * @param[in] output_size The size in bytes of the resulting compressed chunk
   * @param[in] shared_mem_buf The shared memory buffer to be used internally by the API
   * @param[in] tmp_buf The global scratch buffer to be used internally by the API
   * @param[in] group The cooperative group which compresses the input
   */
  template <typename CG>
  void __device__ compress(
      const void* input_chunk,
      void* output_chunk,
      const size_t input_size,
      size_t* output_size,
      uint8_t* shared_mem_buf,
      uint8_t* tmp_buf,
      CG& group)
  {
    Compress<CG, data_type::value, algo::value>().execute(
        input_chunk,
        output_chunk,
        input_size,
        output_size,
        shared_mem_buf,
        tmp_buf,
        max_uncomp_chunk_size::value,
        group);
  }
  
  /** @brief Decompresses a contiguous buffer of data
   *
   * @param[in] input_chunk The to-be-decompressed chunk
   * @param[in] output_chunk The resulting decompressed chunk
   * @param[in] shared_mem_buf The shared memory buffer to be used internally by the API
   * @param[in] tmp_buf The global scratch buffer to be used internally by the API
   * @param[in] group The cooperative group which decompresses the input
   */
  template <typename CG>
  void __device__ decompress(
      const void* input_chunk,
      void* output_chunk,
      uint8_t* shared_mem_buf,
      uint8_t* tmp_buf,
      CG& group)
  {
    Decompress<CG, data_type::value, algo::value>().execute(
        input_chunk, output_chunk, shared_mem_buf, tmp_buf, group);
  }
  
  /** @brief Decompresses a contiguous buffer of data
   *
   * @param[in] input_chunk The to-be-decompressed chunk
   * @param[in] output_chunk The resulting decompressed chunk
   * @param[in] comp_chunk_size The size of the compressed chunk
   * @param[in] decomp_chunk_size The size of the resulting decompressed chunk
   * @param[in] shared_mem_buf The shared memory buffer to be used internally by the API
   * @param[in] tmp_buf The global scratch buffer to be used internally by the API
   * @param[in] group The cooperative group which decompresses the input
   */
  template <typename CG>
  void __device__ decompress(
      const void* input_chunk,
      void* output_chunk,
      const size_t comp_chunk_size,
      const size_t decomp_chunk_size,
      uint8_t* shared_mem_buf,
      uint8_t* tmp_buf,
      CG& group)
  {
    Decompress<CG, data_type::value, algo::value>().execute(
        input_chunk,
        output_chunk,
        comp_chunk_size,
        decomp_chunk_size,
        shared_mem_buf,
        tmp_buf,
        group);
  }

  /** @brief Returns the amount of shared mem necessary for a given CTA
    * @param[in] num_warps_per_block The number of warps per block
    * @return The amount of shared mem in bytes
    */
  static __device__ __host__ constexpr size_t
  shmem_size_block(size_t num_warps_per_block)
  {
    return ShmemSizeBlock<algo::value, direction::value>::execute(
        num_warps_per_block);
  }
  
  /** @brief Returns the amount of shared mem necessary for each cooperative group.
    * Not the same as `shmem_size_block` because there could be multiple API invocations, 
    * each with a different cooperative group that are all part of the same CTA.
    *
    * @return The amount of shared mem in bytes
    */
  static __device__ __host__ constexpr size_t shmem_size_group()
  {
    return ShmemSizeGroup<algo::value, direction::value>::execute();
  }
  
  /** @brief Returns the alignment necessary for the CG's shared memory allocation.
    *
    * @return The shared memory alignment size
    */
  static __device__ __host__ constexpr size_t shmem_alignment()
  {
    return ShmemAlignment<group_type::value, algo::value, direction::value>::execute();
  }

  /** @brief Returns the maximium compressed chunk size
    * @return The max compressed chunk size in bytes
    */
  static constexpr size_t max_comp_chunk_size()
  {
    return MaxCompChunkSize<algo::value>::execute(max_uncomp_chunk_size::value);
  }
  
  /** @brief Returns the scratch space size needed for the whole kernel
    * @return The memory scratch space size in bytes
    */
  static constexpr size_t tmp_size_total(size_t num_warps)
  {
    return TmpSizeTotal<group_type::value, algo::value, direction::value>::execute(
        max_uncomp_chunk_size::value, data_type::value, num_warps);
  }

  /** @brief Returns the global memory scratch space needed for each cooperative group.
   * Not the same as `tmp_size_total` because there could be multiple API invocations per kernel,
   * each requiring part of the total amount of global scratch memory.
   *
   * @return The global memory scratch space size in bytes
   */
  static __device__ __host__ constexpr size_t tmp_size_group()
  {
    return TmpSizeGroup<group_type::value, algo::value, direction::value>::execute(
        max_uncomp_chunk_size::value, data_type::value);
  }
};

} // namespace detail
} // namespace nvcomp::device