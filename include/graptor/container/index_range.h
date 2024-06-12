// -*- C++ -*-
#ifndef GRAPTOR_CONTAINER_INDEX_RANGE_H
#define GRAPTOR_CONTAINER_INDEX_RANGE_H

#include <type_traits>
#include <array>
#include <tuple>
#include <utility>

namespace graptor {

/***********************************************************************
 * Helper utilities
 * See https://stackoverflow.com/questions/20874388/error-spliting-an-stdindex-sequence
 ***********************************************************************/
template <typename T, typename Seq, T Begin>
struct make_integer_range_impl;

template <typename T, T... Ints, T Begin>
struct make_integer_range_impl<T, std::integer_sequence<T, Ints...>, Begin> {
  using type = std::integer_sequence<T, Begin + Ints...>;
};

/* Similar to std::make_integer_sequence<>, except it goes from [Begin, End)
   instead of [0, End). */
template <typename T, T Begin, T End>
using make_integer_range = typename make_integer_range_impl<
    T, std::make_integer_sequence<T, End - Begin>, Begin>::type;

/* Similar to std::make_index_sequence<>, except it goes from [Begin, End)
   instead of [0, End). */
template <std::size_t Begin, std::size_t End>
using make_index_range = make_integer_range<std::size_t, Begin, End>;

template <std::size_t... Indices, std::size_t... I>
constexpr decltype(auto) slice_impl(
      std::index_sequence<Indices...>,
      std::index_sequence<I...>) {
  using Array = std::array<std::size_t, sizeof...(Indices)>;
  return std::index_sequence<std::get<I>(Array{{Indices...}})...>();
}

template <std::size_t Begin, std::size_t End, std::size_t... Indices>
constexpr decltype(auto) slice( std::index_sequence<Indices...> idx_seq ) {
  return slice_impl( idx_seq, make_index_range<Begin, End>() );
}

/***********************************************************************
 * Helper
 ***********************************************************************/
template<typename T, typename SeqL, typename SeqR>
struct make_index_range_for_remove_impl;

template<typename T, T... IntsL, T... IntsR>
struct make_index_range_for_remove_impl<T,
					std::integer_sequence<T, IntsL...>,
					std::integer_sequence<T, IntsR...>> {
    using type = std::integer_sequence<T, IntsL..., IntsR...>;
};

template <typename T, T N, T Omit>
struct make_index_range_for_remove {
    using type =
	typename make_index_range_for_remove_impl<
	T,
	make_index_range<0,Omit>,
	make_index_range<Omit+1,N>>::type;
};

template<typename Tuple, typename Idx>
struct tuple_select_type;
	
template<typename... T, std::size_t... I>
struct tuple_select_type<
    std::tuple<T...>,
    std::integer_sequence<std::size_t, I...>> {
    using type = std::tuple<std::tuple_element_t<I,std::tuple<T...>> ...>;
};

template<typename... T, std::size_t... I>
constexpr decltype(auto) tuple_select(
    std::tuple<T...> && t,
    std::index_sequence<I...>
    ) {
    return std::make_tuple( std::get<I>(t)... );
};

template<std::size_t rm, typename... T>
constexpr decltype(auto) tuple_remove( std::tuple<T...> && t ) {
    return tuple_select(
	std::forward<std::tuple<T...>>( t ),
	typename make_index_range_for_remove<std::size_t, sizeof...(T), rm>::type()
	);
};

template<typename Seq, typename Idx>
struct integer_sequence_select;

template<typename T, std::size_t... I, std::size_t... S>
struct integer_sequence_select<
    std::integer_sequence<T, I...>,
    std::integer_sequence<std::size_t, S...>> {
    using type = std::integer_sequence<T, 
	std::get<S>( std::integer_sequence<T, I...>() ) ... >;
};

template<typename... T>
struct convert_to_integer_sequence;

template<typename T, T... I>
struct convert_to_integer_sequence<
    std::tuple<std::integral_constant<T,I> ...>> {
    using type = std::integer_sequence<T, I...>;
};

template<>
struct convert_to_integer_sequence<std::tuple<>> {
    using type = std::integer_sequence<std::size_t>;
};

template<std::size_t rm, typename Seq>
struct integer_sequence_remove;

template<std::size_t rm, typename T, T... I>
struct integer_sequence_remove<rm,std::integer_sequence<T,I...>>
    : convert_to_integer_sequence<
    typename tuple_select_type<
    std::tuple<std::integral_constant<T,I> ...>,
    typename make_index_range_for_remove<std::size_t, sizeof...(I), rm>::type>
    ::type> {
};

/***********************************************************************
 * Tuples - iteration
 * https://codereview.stackexchange.com/a/67394/278929
 ***********************************************************************/

template <typename Tuple, typename F, std::size_t ...Indices>
void for_each_impl(Tuple&& tuple, F&& f, std::index_sequence<Indices...>) {
    using swallow = int[];
    (void)swallow{1,
        (f(std::get<Indices>(std::forward<Tuple>(tuple))), void(), int{})...
    };
}

template <typename Tuple, typename F>
void for_each(Tuple&& tuple, F&& f) {
    constexpr std::size_t N = std::tuple_size_v<std::remove_reference_t<Tuple>>;
    for_each_impl(std::forward<Tuple>(tuple), std::forward<F>(f),
                  std::make_index_sequence<N>{});
}

/***********************************************************************
 * Tuples - iteration over pairs of tuples
 ***********************************************************************/

template <typename Tuple, typename Tuple2, typename F, std::size_t ...Indices>
void for_each_impl( Tuple && tuple, Tuple2 & tuple2,
		    F && f, std::index_sequence<Indices...> ) {
    using swallow = int[];
    (void)swallow{1,
	    ( f( std::get<Indices>( std::forward<Tuple>(tuple) ),
		 std::get<Indices>( tuple2 ) ),
	      void(), int{} )... };
}

template <typename Tuple, typename Tuple2, typename F>
void for_each( Tuple && tuple, Tuple2 & tuple2, F && f ) {
    constexpr std::size_t N = std::tuple_size_v<std::remove_reference_t<Tuple>>;
    for_each_impl( std::forward<Tuple>(tuple), tuple2, std::forward<F>(f),
		   std::make_index_sequence<N>{});
}
    
} // namespace graptor

#endif // GRAPTOR_CONTAINER_INDEX_RANGE_H

