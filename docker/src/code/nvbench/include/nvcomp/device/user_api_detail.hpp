#pragma once

#include "operators.hpp"
#include <cuda/std/type_traits>
#include <stdint.h>

namespace nvcomp::device
{
namespace detail
{

template <nvcomp_operator OperatorType, class T>
struct is_operator : cuda::std::false_type
{
};

template <nvcomp_direction D>
struct is_operator<nvcomp_operator::direction, Direction<D>>
    : cuda::std::true_type
{
};

template <nvcomp_algo A>
struct is_operator<nvcomp_operator::algo, Algo<A>> : cuda::std::true_type
{
};

template <nvcomp_datatype D>
struct is_operator<nvcomp_operator::datatype, Datatype<D>>
    : cuda::std::true_type
{
};

template <nvcomp_grouptype G>
struct is_operator<nvcomp_operator::grouptype, Grouptype<G>>
    : cuda::std::true_type
{
};

template <size_t N>
struct is_operator<
    nvcomp_operator::max_uncomp_chunk_size,
    MaxUncompChunkSize<N>> : cuda::std::true_type
{
};

namespace has_n_of_impl
{
template <
    unsigned int Counter,
    nvcomp_operator OperatorType,
    class Head,
    class... Types>
struct counter_helper
{
  static constexpr unsigned int value
      = is_operator<OperatorType, Head>::value
            ? counter_helper<(Counter + 1), OperatorType, Types...>::value
            : counter_helper<Counter, OperatorType, Types...>::value;
};

template <unsigned int Counter, nvcomp_operator OperatorType, class Head>
struct counter_helper<Counter, OperatorType, Head>
{
  static constexpr unsigned int value
      = is_operator<OperatorType, Head>::value ? Counter + 1 : Counter;
};

template <nvcomp_operator OperatorType, class Operator>
struct counter : cuda::std::integral_constant<
                     unsigned int,
                     is_operator<OperatorType, Operator>::value>
{
};

template <
    nvcomp_operator OperatorType,
    template <class...>
    class Description,
    class... Types>
struct counter<OperatorType, Description<Types...>>
    : cuda::std::integral_constant<
          unsigned int,
          counter_helper<0, OperatorType, Types...>::value>
{
};

} // namespace has_n_of_impl

template <nvcomp_operator OperatorType, class Description>
struct has_operator
    : cuda::std::integral_constant<
          bool,
          (has_n_of_impl::counter<OperatorType, Description>::value > 0)>
{
};

template <nvcomp_operator OperatorType, class... Operators>
struct get_operator
{
};

template <nvcomp_operator OperatorType, class T>
struct get_operator<OperatorType, T>
{
  using type = typename cuda::std::
      conditional<is_operator<OperatorType, T>::value, T, void>::type;
};

template <nvcomp_operator OperatorType, class TypeHead, class... TailTypes>
struct get_operator<OperatorType, TypeHead, TailTypes...>
{
  using type = typename cuda::std::conditional<
      is_operator<OperatorType, TypeHead>::value,
      TypeHead,
      typename get_operator<OperatorType, TailTypes...>::type>::type;
};

template<class... Operators>
class nvcomp_device_execution;

template <class... Operators>
class make_description
{
  using execution_type = nvcomp_device_execution<Operators...>;
  using nvcomp_execution_type =
      typename cuda::std::type_identity<execution_type>::type;

public:
  using type = nvcomp_execution_type;
};

template <class... Operators>
using make_description_t = typename make_description<Operators...>::type;

template <class T>
struct is_operator_expression
    : public cuda::std::is_base_of<operator_expression, T>
{
};

template <class T, class U>
struct are_operator_expressions
    : public cuda::std::integral_constant<
          bool,
          is_operator_expression<T>::value && is_operator_expression<U>::value>
{
};

} // namespace detail

template <class Operator1, class Operator2>
__host__ __device__ auto
operator+(const Operator1&, const Operator2&) -> typename cuda::std::enable_if<
    detail::are_operator_expressions<Operator1, Operator2>::value,
    detail::make_description_t<Operator1, Operator2>>::type
{
  return detail::make_description_t<Operator1, Operator2>();
}

template <class... Operators1, class Operator2>
__host__ __device__ auto operator+(
    const detail::nvcomp_device_execution<Operators1...>&, const Operator2&) ->
    typename cuda::std::enable_if<
        detail::is_operator_expression<Operator2>::value,
        detail::make_description_t<Operators1..., Operator2>>::type
{
  return detail::make_description_t<Operators1..., Operator2>();
}

template <class Operator1, class... Operators2>
__host__ __device__ auto operator+(
    const Operator1&, const detail::nvcomp_device_execution<Operators2...>&) ->
    typename cuda::std::enable_if<
        detail::is_operator_expression<Operator1>::value,
        detail::make_description_t<Operator1, Operators2...>>::type
{
  return detail::make_description_t<Operator1, Operators2...>();
}

template <class... Operators1, class... Operators2>
__host__ __device__ auto operator+(
    const detail::nvcomp_device_execution<Operators1...>&,
    const detail::nvcomp_device_execution<Operators2...>&)
    -> detail::make_description_t<Operators1..., Operators2...>
{
  return detail::make_description_t<Operators1..., Operators2...>();
}

} // namespace nvcomp::device
