#pragma once

#include "operators.hpp"
#include <cooperative_groups.h>

namespace nvcomp::device {
namespace detail {

template <nvcomp_algo A, nvcomp_direction D>
class ShmemSizeBlock;

template <nvcomp_algo A, nvcomp_direction D>
class ShmemSizeGroup;

template <nvcomp_grouptype G, nvcomp_algo A, nvcomp_direction D>
class TmpSizeTotal;

template <nvcomp_grouptype G, nvcomp_algo A, nvcomp_direction D>
class TmpSizeGroup;

template <nvcomp_grouptype G, nvcomp_algo A, nvcomp_direction D>
class ShmemAlignment;

template <nvcomp_algo A>
class MaxCompChunkSize;

template <typename CG, nvcomp_datatype DT, nvcomp_algo A>
class Compress;

template <typename CG, nvcomp_datatype DT, nvcomp_algo A>
class Decompress;

typedef cooperative_groups::__v1::thread_block_tile<32U, cooperative_groups::__v1::thread_block> WarpGroup;

} // namespace detail
} // namespace nvcomp::device
