#pragma once

#include <cuda/std/type_traits>
#include <stdint.h>

namespace nvcomp::device
{
namespace detail
{

/** 
 * @enum nvcomp_operator
 * @var nvcomp_operator::direction
 * Selects between compression/decompression
 * @var nvcomp_operator::algo
 * Selects the compression algorithm.
 * @var nvcomp_operator::warp
 * If provided, the API expects to be called with a single warp
 * @var nvcomp_operator::block
 * If provided, the API expects to be called with a single block
 * @var nvcomp_operator::multi_block
 * If provided, the API expects to be called with a one or more blocks
 * @var nvcomp_operator::datatype
 * The format of the input data
 * @var nvcomp_operator::max_uncomp_chunk_size
 * The maximum uncompressed chunk size. (For compression only)
 */
enum class nvcomp_operator
{
  direction,
  algo,
  grouptype,
  datatype,
  max_uncomp_chunk_size
};

struct operator_expression
{
};

template <class ValueType, ValueType Value>
struct constant_operator_expression
    : operator_expression,
      public cuda::std::integral_constant<ValueType, Value>
{
};

} // namespace detail

/**
 * @enum nvcomp_direction
 * Selection of compression or decompression.
 */
enum class nvcomp_direction
{
  compress,
  decompress
};

/**
 * @enum nvcomp_algo
 * The compression algorithm to be selected.
 */
enum class nvcomp_algo
{
  ans,
  zstd,
  bitcomp,
  lz4,
  deflate,
  gdeflate
};

/**
 * @enum nvcomp_datatype
 *
 * @brief The way in which the compression algo will interpret the input data
 *
 * @var uint8 - Data to be interpreted as consecutive bytes. If the input datatype is not included
 * in the options below, uint8 should be selected.
 *
 * @var nvcomp_datatype::float16 - Data to be interpreted as consecutive IEEE half-precision floats.
 * Requires the total number of input bytes per chunk to be divisible by two. 
 *
 * @var nvcomp_datatype::bfloat16 - Data to be interpreted as consecutive bfloat16 values. Requires
 * the total number of input bytes per chunk to be divisible by two. 
 */
enum class nvcomp_datatype
{
  uint8,
  float16,
  bfloat16
};

/**
 * @enum nvcomp_grouptype
 * @var warp - Group provided to API expected to be single-warp-sized.
 */
enum class nvcomp_grouptype
{
  warp
};

template <nvcomp_algo Value>
struct Algo : public detail::constant_operator_expression<nvcomp_algo, Value>
{
};

template <nvcomp_direction Value>
struct Direction
    : public detail::constant_operator_expression<nvcomp_direction, Value>
{
};

template <nvcomp_datatype Value>
struct Datatype
    : public detail::constant_operator_expression<nvcomp_datatype, Value>
{
};

template <nvcomp_grouptype Value>
struct Grouptype
    : public detail::constant_operator_expression<nvcomp_grouptype, Value>
{
};

template <size_t Value>
struct MaxUncompChunkSize
    : public detail::constant_operator_expression<size_t, Value>
{
};

template <class T>
struct dependent_false : std::false_type
{
};

template <nvcomp_algo A>
struct dependent_false_algo : std::false_type
{
};

} // namespace nvcomp::device
