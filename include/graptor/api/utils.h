// -*- C++ -*-
#ifndef GRAPTOR_API_UTILS_H
#define GRAPTOR_API_UTILS_H

#include "graptor/dsl/ast/decl.h"

class frontier;

namespace api {

namespace { // anonymous

/************************************************************************
 * Auxiliary for validating argument types.
 * Each condition should evaluate to true at most once.
 ************************************************************************/
// Zero template conditions - there may not be any arguments
template<typename... Args>
struct check_arguments_0 : public std::false_type { };

template<>
struct check_arguments_0<> : public std::true_type { };

// One template condition
template<template<typename> class C0, typename... Args>
struct check_arguments_1;

template<template<typename> class C0>
struct check_arguments_1<C0> : public std::true_type { };

template<template<typename> class C0, typename Arg0, typename... Args>
struct check_arguments_1<C0,Arg0,Args...> {
    static constexpr bool value =
	C0<std::decay_t<Arg0>>::value && check_arguments_0<Args...>::value;
};

// Two template conditions
template<template<typename> class C0, template<typename> class C1,
	 typename... Args>
struct check_arguments_2;

template<template<typename> class C0, template<typename> class C1>
struct check_arguments_2<C0,C1> : public std::true_type { };

template<template<typename> class C0, template<typename> class C1,
	 typename Arg0, typename... Args>
struct check_arguments_2<C0,C1,Arg0,Args...> {
    static constexpr bool value =
	( C0<std::decay_t<Arg0>>::value
	  && check_arguments_1<C1,Args...>::value )
	|| ( C1<std::decay_t<Arg0>>::value
	     && check_arguments_1<C0,Args...>::value );
};

// Three template conditions
template<template<typename> class C0, template<typename> class C1,
	 template<typename> class C2, typename... Args>
struct check_arguments_3;

template<template<typename> class C0, template<typename> class C1,
	 template<typename> class C2>
struct check_arguments_3<C0,C1,C2> : public std::true_type { };

template<template<typename> class C0, template<typename> class C1,
	 template<typename> class C2,
	 typename Arg0, typename... Args>
struct check_arguments_3<C0,C1,C2,Arg0,Args...> {
    static constexpr bool value =
	( C0<std::decay_t<Arg0>>::value
	  && check_arguments_2<C1,C2,Args...>::value )
	|| ( C1<std::decay_t<Arg0>>::value
	     && check_arguments_2<C0,C2,Args...>::value )
	|| ( C2<std::decay_t<Arg0>>::value
	     && check_arguments_2<C0,C1,Args...>::value );
};

/************************************************************************
 * Auxiliary for identifying if an argument type is present
 ************************************************************************/
template<template<typename> class C, typename... Args>
struct has_argument : public std::false_type { };

template<template<typename> class C, typename Arg, typename... Args>
struct has_argument<C,Arg,Args...> {
    static constexpr bool value =
	C<std::decay_t<Arg>>::value || has_argument<C,Args...>::value;
};

template<template<typename> class C, typename... Args>
constexpr bool has_argument_v = has_argument<C,Args...>::value;

/************************************************************************
 * Auxiliary for picking up the first type argument that meets a specific
 * constraint.
 ************************************************************************/
template<typename... Args>
struct _pack { };

template<template<typename> class C0, typename Default, typename Pack,
	 typename Enable = void>
struct get_argument_type_helper;

template<template<typename> class C0, typename Default>
struct get_argument_type_helper<C0,Default,_pack<>> {
    using type = Default;
};

template<template<typename> class C0, typename Default,
	 typename Arg0, typename... Args>
struct get_argument_type_helper<C0,Default,_pack<Arg0,Args...>,
				std::enable_if_t<C0<std::decay_t<Arg0>>::value>> {
    using type = std::decay_t<Arg0>;
};

template<template<typename> class C0, typename Default,
	 typename Arg0, typename... Args>
struct get_argument_type_helper<C0,Default,_pack<Arg0,Args...>,
				std::enable_if_t<!C0<std::decay_t<Arg0>>::value>> {
    using type =
	typename get_argument_type_helper<C0,Default,_pack<Args...>>::type;
};

template<template<typename> class C0, typename Default, typename... Args>
using get_argument_type = get_argument_type_helper<C0,Default,_pack<Args...>>;

template<template<typename> class C0, typename Default, typename... Args>
using get_argument_type_t =
    typename get_argument_type<C0,Default,Args...>::type;

/************************************************************************
 * Auxiliary for picking up the first argument value whose type meets a
 * specific constraint.
 ************************************************************************/
template<template<typename> class C0, typename MissingTy>
MissingTy & get_argument_value() {
    static MissingTy missing;
    return missing;
}

template<template<typename> class C0, typename MissingTy,
	 typename Arg0, typename... Args>
auto & get_argument_value( Arg0 & arg0, Args &... args ) {
    if constexpr ( C0<std::decay_t<Arg0>>::value )
	return arg0;
    else
	return get_argument_value<C0,MissingTy>( args... );
}

} // namespace anonymous

/************************************************************************
 * Auxiliary for testing an argument is a frontier or active condition
 ************************************************************************/
template<typename T>
using is_frontier = std::is_same<T,frontier>;

template<typename T>
constexpr bool is_frontier_v = is_frontier<T>::value;

template<typename Fn>
using is_active =
    std::is_invocable<Fn,expr::value<simd::ty<VID,1>,expr::vk_dst>>;

template<typename T>
constexpr bool is_active_v = is_active<T>::value;

} // namespace api

#endif // GRAPTOR_API_UTILS_H
