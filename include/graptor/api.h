// -*- C++ -*-
#ifndef GRAPTOR_API_H
#define GRAPTOR_API_H

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

/************************************************************************
 * Definition of parameters expressing strength of filters
 ************************************************************************/
enum class filter_strength {
    weak,
    strong,
    none
};

template<filter_strength S>
struct filter_strength_argument {
    static constexpr filter_strength value = S;
};

static constexpr auto weak =
    filter_strength_argument<filter_strength::weak>();
static constexpr auto strong =
    filter_strength_argument<filter_strength::strong>();

template<typename T>
struct is_filter_strength : public std::false_type { };

template<filter_strength S>
struct is_filter_strength<filter_strength_argument<S>>
    : public std::true_type { };

template<typename T>
constexpr bool is_filter_strength_v = is_filter_strength<T>::value;

/************************************************************************
 * Definition of parameters expressing where filters apply
 ************************************************************************/
enum class filter_entity {
    src,
    dst
};

template<filter_entity E>
struct filter_entity_argument {
    static constexpr filter_entity value = E;
};

static constexpr auto src =
    filter_entity_argument<filter_entity::src>();
static constexpr auto dst =
    filter_entity_argument<filter_entity::dst>();

template<typename T>
struct is_filter_entity : public std::false_type { };

template<filter_entity E>
struct is_filter_entity<filter_entity_argument<E>>
    : public std::true_type { };

template<typename T>
constexpr bool is_filter_entity_v = is_filter_entity<T>::value;

/************************************************************************
 * Definition of parameters expressing parallelism
 ************************************************************************/
enum class parallelism_spec {
    parallel,
    sequential
};

template<parallelism_spec P>
struct parallelism_argument {
    static constexpr parallelism_spec value = P;
};

static constexpr auto parallel =
    parallelism_argument<parallelism_spec::parallel>();
static constexpr auto sequential =
    parallelism_argument<parallelism_spec::sequential>();

template<typename T>
struct is_parallelism : public std::false_type { };

template<parallelism_spec P>
struct is_parallelism<parallelism_argument<P>>
    : public std::true_type { };

template<typename T>
constexpr bool is_parallelism_v = is_parallelism<T>::value;

/************************************************************************
 * Definition of parameters expressing vectorization
 ************************************************************************/
template<unsigned short vlen>
struct vectorization_spec {
    static constexpr unsigned short value =
	std::min( vlen, (unsigned short)MAX_VL );
};

template<unsigned short vlen>
using vl_max = vectorization_spec<vlen>;

template<unsigned short vlen>
struct vectorization_argument {
    static constexpr vectorization_spec<vlen> value
    = vectorization_spec<vlen>();
};

template<typename T>
struct is_vectorization : public std::false_type { };

template<unsigned short vlen>
struct is_vectorization<vectorization_argument<vlen>>
    : public std::true_type { };

template<typename T>
constexpr bool is_vectorization_v = is_vectorization<T>::value;

/************************************************************************
 * Definition of sparse/dense threshold
 ************************************************************************/
class frac_threshold {
    friend std::ostream & operator << ( std::ostream & os, frac_threshold t );

public:
    frac_threshold( int threshold ) : m_threshold( threshold ) { }

    bool is_sparse( VID nactv, EID nacte, EID m ) const {
	return ( EID(nactv) + nacte ) <= m / EID( 100.0 / (float)m_threshold );
    }

    bool is_sparse( frontier F, EID m ) const {
	VID nactv = F.nActiveVertices();
	EID nacte = F.nActiveEdges();
	return is_sparse( nactv, nacte, m );
    }
private:
    int m_threshold;
};

struct default_threshold {
    bool is_sparse( VID nactv, EID nacte, EID m ) const {
	return ( EID(nactv) + nacte ) <= ( m / 20 );
    }

    bool is_sparse( frontier F, EID m ) const {
	VID nactv = F.nActiveVertices();
	EID nacte = F.nActiveEdges();
	return is_sparse( nactv, nacte, m );
    }
};


struct always_sparse_t {
    bool is_sparse( VID nactv, EID nacte, EID m ) const {
	return true;
    }

    bool is_sparse( frontier F, EID m ) const {
	VID nactv = F.nActiveVertices();
	EID nacte = F.nActiveEdges();
	return is_sparse( nactv, nacte, m );
    }
};

struct always_dense_t {
    bool is_sparse( VID nactv, EID nacte, EID m ) const {
	return false;
    }

    bool is_sparse( frontier F, EID m ) const {
	VID nactv = F.nActiveVertices();
	EID nacte = F.nActiveEdges();
	return is_sparse( nactv, nacte, m );
    }
};

std::ostream & operator << ( std::ostream & os, frac_threshold t ) {
    return os << "fractional(" << t.m_threshold << ")";
}

std::ostream & operator << ( std::ostream & os, default_threshold t ) {
    return os << "fractional(default)";
}

std::ostream & operator << ( std::ostream & os, always_sparse_t t ) {
    return os << "always-sparse";
}

std::ostream & operator << ( std::ostream & os, always_dense_t t ) {
    return os << "always-dense";
}

static constexpr auto always_sparse = always_sparse_t();
static constexpr auto always_dense = always_dense_t();

template<typename T>
struct is_threshold : public std::false_type { };

template<>
struct is_threshold<default_threshold>
    : public std::true_type { };

template<>
struct is_threshold<frac_threshold>
    : public std::true_type { };

template<>
struct is_threshold<always_sparse_t>
    : public std::true_type { };

template<>
struct is_threshold<always_dense_t>
    : public std::true_type { };

template<typename T>
constexpr bool is_threshold_v = is_threshold<T>::value;

/**=====================================================================*
 * Defintion of filters
 *
 * Filtering of inactive lanes can be done in following ways:
 *   1. Apply a mask on all computations on that lane
 *   2. Use an 'active'/'convergence' method that is checked for every
 *      in-neighbour. This is available only in pull traversal.
 *   3. Use an 'enabled' method that is checked for every
 *      out-neighbour. This is available only in push traversal.
 * Filters are defined according to the following rules:
 * - push traversal, filter dst: strong => mask, weak => none
 * - push traversal, filter src: strong => mask+enabled, weak => enabled
 * - pull traversal, filter dst: strong => mask+active, => active
 * - pull traversal, filter src: strong => mask, weak => none
 * - ireg traversal, filter either: strong => mask, weak => none
 *======================================================================*/
template<typename EMapConfig,
	 filter_entity E,
	 bool as_method,
	 bool as_mask,
	 typename Operator>
struct arg_filter_op {
    static constexpr frontier_type ftype = EMapConfig::rd_ftype;
    using StoreTy = typename frontier_params<ftype,EMapConfig::VL>::type;
    using array_ty =
	typename expr::array_select<ftype,StoreTy,VID,expr::aid_frontier_old_f,
				    array_encoding<StoreTy>>::type;
    
    static constexpr bool is_scan = Operator::is_scan;

    arg_filter_op( Operator op, frontier & fr )
	: m_op( op ), m_array(), m_frontier( fr ) { }

    template<typename VIDSrc, typename VIDDst, typename EIDEdge>
    auto relax( VIDSrc s, VIDDst d, EIDEdge e ) const {
	if constexpr ( as_mask ) {
	    if constexpr ( E == filter_entity::src ) {
		using Tr = typename VIDSrc::data_type::prefmask_traits;
		auto ss = expr::remove_mask( s );
		return expr::set_mask(
		    expr::get_mask_cond( s ),
		    expr::set_mask(
			expr::make_unop_cvt_to_mask<Tr>( m_array[ss] ),
			m_op.relax( ss, d, e ) ) );
	    } else {
		using Tr = typename VIDDst::data_type::prefmask_traits;
		auto dd = expr::remove_mask( d );
		return expr::set_mask(
		    expr::get_mask_cond( d ),
		    expr::set_mask(
			expr::make_unop_cvt_to_mask<Tr>( m_array[dd] ),
			m_op.relax( s, d, e ) ) );
	    }
	} else
	    return m_op.relax( s, d, e );
    }

    template<typename VIDType>
    auto enabled( VIDType s ) {
	if constexpr ( E == filter_entity::src && as_method ) {
	    using Tr = simd::detail::mask_preferred_traits_type<
		typename VIDType::type, VIDType::VL>;
	    return m_op.enabled( s ) &&
		expr::make_unop_cvt_to_mask<Tr>( m_array[s] );
	} else
	    return m_op.enabled( s );
    }
    
    template<typename VIDType>
    auto active( VIDType d ) {
	if constexpr ( E == filter_entity::dst && as_method ) {
	    using Tr = simd::detail::mask_preferred_traits_type<
		typename VIDType::type, VIDType::VL>;
	    return m_op.active( d ) &&
		expr::make_unop_cvt_to_mask<Tr>( m_array[d] );
	} else
	    return m_op.active( d );
    }
    
    template<typename VIDType>
    auto update( VIDType d ) {
	return m_op.update( d );
    }
    
    template<typename VIDType>
    auto vertexop( VIDType vid ) {
	// Semantics of this are unclear: what if we specify a filter src/dst
	// and a vertexmap is merged in with the edgemap. Is the vertexmap
	// then also filtered?
	if constexpr ( E == filter_entity::dst && as_mask ) {
	    using Tr = typename VIDType::data_type::prefmask_traits;
	    return expr::set_mask(
		expr::make_unop_cvt_to_mask<Tr>( m_array[vid] ),
		m_op.vertexop( vid ) );
	} else
	    return m_op.vertexop( vid );
    }

    template<typename PSet>
    auto get_ptrset( const PSet & pset ) const {
	return expr::map_set_if_absent<
	    (unsigned)aid_key(expr::array_aid(expr::aid_frontier_old_f))>(
		m_op.get_ptrset( pset ), m_frontier.getDense<ftype>() );
    }

    auto get_ptrset() const { // TODO - redundant
	return get_ptrset( expr::create_map() );
    }

    const auto get_config() const { return m_op.get_config(); }

private:
    Operator m_op;
    array_ty m_array;
    frontier & m_frontier;
};

template<filter_entity E, filter_strength S>
struct arg_filter {
    static constexpr filter_entity entity = E;
    static constexpr filter_strength strength = S;
    static constexpr bool is_frontier = true;

    arg_filter( frontier & f ) : m_frontier( f ) { };

    template<typename EMapConfig,
	     graph_traversal_kind gtk, typename Operator>
    auto check_strength( Operator op ) {
	constexpr bool as_mask = strength == filter_strength::strong;
	constexpr bool as_method
		      = ( gtk == graph_traversal_kind::gt_push
			  && entity == filter_entity::src )
		      || ( gtk == graph_traversal_kind::gt_pull
			   && entity == filter_entity::dst );
	if constexpr ( as_method || as_mask )
	    return check<EMapConfig,as_method,as_mask>( op );
	else
	    return op;
    }

    template<typename EMapConfig, bool as_method, bool as_mask,
	     typename Operator>
    auto check( Operator op ) {
	return arg_filter_op<EMapConfig,entity,as_method,as_mask,Operator>(
	    op, m_frontier );
    }

    template<frontier_type ftype>
    void setup( const partitioner & part ) {
	m_frontier.template toDense<ftype>( part );
    }

    template<typename GraphType>
    frontier & get_frontier( const GraphType & GA ) { return m_frontier; }

    bool is_true_frontier() const {
	return m_frontier.getType() == frontier_type::ft_true;
    }

private:
    frontier & m_frontier;
};

// This operator class is inserted only if a strong frontier is supplied.
template<typename EMapConfig, filter_entity E, bool as_method, bool as_mask,
	 typename Operator,
	 typename Fn, typename PtrSet>
struct arg_filter_a_op {
    static constexpr frontier_type ftype = EMapConfig::rd_ftype;
    using StoreTy = typename frontier_params<ftype,EMapConfig::VL>::type;
    using array_ty =
	typename expr::array_select<ftype,StoreTy,VID,expr::aid_frontier_old_f,
				    array_encoding<StoreTy>>::type;
    using ptrset_ty = PtrSet;
    
    static constexpr bool is_scan = Operator::is_scan;

    arg_filter_a_op( Operator op, Fn method, ptrset_ty & ptrset )
	: m_op( op ), m_method( method ), m_ptrset( ptrset ) { }

    template<typename VIDSrc, typename VIDDst, typename EIDEdge>
    auto relax( VIDSrc s, VIDDst d, EIDEdge e ) const {
	using Tr = simd::detail::mask_preferred_traits_type<
	    typename VIDSrc::type, VIDSrc::VL>;
	if constexpr ( as_mask ) {
	    if constexpr ( E == filter_entity::src )
		return expr::set_mask(
		    expr::rewrite_internal( m_method(s) ),
		    m_op.relax( s, d, e ) );
	    else
		return expr::set_mask(
		    expr::rewrite_internal( m_method(d) ),
		    m_op.relax( s, d, e ) );
	} else
	    return m_op.relax( s, d, e );
    }

    template<typename VIDType>
    auto enabled( VIDType s ) {
	if constexpr ( E == filter_entity::src && as_method ) {
	    using Tr = simd::detail::mask_preferred_traits_type<
		typename VIDType::type, VIDType::VL>;
	    return m_op.enabled( s ) &&
		expr::make_unop_cvt_to_mask<Tr>( m_method( s ) );
	} else
	    return m_op.enabled( s );
    }
    
    template<typename VIDType>
    auto active( VIDType d ) {
	if constexpr ( E == filter_entity::dst && as_method ) {
	    using Tr = simd::detail::mask_preferred_traits_type<
		typename VIDType::type, VIDType::VL>;
	    return m_op.active( d ) &&
		expr::make_unop_cvt_to_mask<Tr>( m_method( d ) );
	} else
	    return m_op.active( d );
    }
    
    template<typename VIDType>
    auto update( VIDType vid ) {
	return m_op.update( vid );
    }
    
    template<typename VIDType>
    auto vertexop( VIDType vid ) {
	if constexpr ( E == filter_entity::dst && as_mask ) {
	    return expr::set_mask(
		expr::rewrite_internal( m_method(vid) ),
		m_op.vertexop( vid ) );
	} else
	    return m_op.vertexop( vid );
    }

    template<typename PSet>
    auto get_ptrset( const PSet & pset ) const {
	return map_merge( m_op.get_ptrset( pset ), m_ptrset );
    }

    auto get_ptrset() const { // TODO - redundant
	return get_ptrset( expr::create_map() );
    }

    template<frontier_type ftype>
    void setup( const partitioner & part ) { }

    const auto get_config() const { return m_op.get_config(); }

private:
    Operator m_op;
    Fn m_method;
    ptrset_ty m_ptrset;
};

template<filter_entity E, filter_strength S, typename Fn>
struct arg_filter_a {
    static constexpr filter_entity entity = E;
    static constexpr filter_strength strength = S;
    static constexpr bool is_frontier = false;

    static_assert( entity == filter_entity::dst,
		   "currently only supported on destinations" );

    arg_filter_a( Fn & f ) : m_active( f ) { };

    template<typename EMapConfig,
	     graph_traversal_kind gtk, typename Operator>
    auto check_strength( Operator op ) {
	constexpr bool as_mask = strength == filter_strength::strong;
	constexpr bool as_method
		      = ( gtk == graph_traversal_kind::gt_push
			  && entity == filter_entity::src )
		      || ( gtk == graph_traversal_kind::gt_pull
			   && entity == filter_entity::dst );
	if constexpr ( as_method || as_mask )
	    return check<EMapConfig,as_method,as_mask>( op );
	else
	    return op;
    }

    template<typename EMapConfig, bool as_method, bool as_mask,
	     typename Operator>
    auto check( Operator op ) {
	if constexpr ( entity == filter_entity::src ) {
	    auto s = expr::value<simd::ty<VID,1>,expr::vk_src>();
	    auto ptrset = expr::extract_pointer_set( m_active( s ) );
	    return arg_filter_a_op<EMapConfig,entity,as_method,as_mask,Operator,
				   Fn,decltype(ptrset)>(
				       op, m_active, ptrset );
	} else {
	    auto d = expr::value<simd::ty<VID,1>,expr::vk_dst>();
	    auto ptrset = expr::extract_pointer_set( m_active( d ) );
	    return arg_filter_a_op<EMapConfig,entity,as_method,as_mask,Operator,
				   Fn,decltype(ptrset)>(
				       op, m_active, ptrset );
	}
    }

    bool is_true_frontier() const { return false; }

    Fn m_active;
};

struct missing_filter_argument {
    static constexpr filter_entity entity = filter_entity::dst;
    static constexpr filter_strength strength = filter_strength::none;
    static constexpr bool is_frontier = true;

    template<typename EMapConfig,
	     graph_traversal_kind gtk, typename Operator>
    auto check_strength( Operator op ) {
	return op;
    }
    template<typename EMapConfig, bool as_mask, typename Operator>
    auto check( Operator op ) {
	return op;
    }
    template<frontier_type ftype>
    void setup( const partitioner & part ) { }

    bool is_true_frontier() const { return true; }
};

template<typename... Args>
auto filter( Args &&... args ) {
    if constexpr( check_arguments_3<is_filter_entity,is_filter_strength,
		  is_frontier,Args...>::value ) {
	return arg_filter<
	    get_argument_type_t<is_filter_entity,decltype(src),Args...>::value,
	    get_argument_type_t<is_filter_strength,decltype(strong),Args...>::value>(
		get_argument_value<is_frontier,missing_filter_argument>(
		    args... ) );
    } else if constexpr ( check_arguments_3<is_filter_entity,
			  is_filter_strength,is_active,Args...>::value ) {
	auto & a = get_argument_value<is_active,missing_filter_argument>(
	    args... );
	return arg_filter_a<
	    get_argument_type_t<is_filter_entity,decltype(dst),Args...>::value,
	    get_argument_type_t<is_filter_strength,decltype(weak),Args...>::value,
	    decltype(a)>( a );
    } else
	return 0; // static_assert( false, "unexpected argument(s) supplied" );
}

template<typename T>
struct is_filter : public std::false_type { };

template<filter_entity E, filter_strength S>
struct is_filter<arg_filter<E,S>> : public std::true_type { };

template<typename T>
struct is_filter_method : public std::false_type { };

template<filter_entity E, filter_strength S, typename Fn>
struct is_filter_method<arg_filter_a<E,S,Fn>> : public std::true_type { };

template<filter_entity Entity, typename T>
struct is_filter_for : public std::false_type { };

template<filter_entity E, filter_strength S>
struct is_filter_for<E,arg_filter<E,S>> : public std::true_type { };

// template<filter_entity E, filter_strength S, typename Fn>
// struct is_filter_for<E,arg_filter_a<E,S,Fn>> : public std::true_type { };

template<typename T>
using is_filter_for_src = is_filter_for<filter_entity::src,T>;

template<typename T>
using is_filter_for_dst = is_filter_for<filter_entity::dst,T>;

/************************************************************************
 * Definition of parameters expressing how to record new frontier
 ************************************************************************/
enum class frontier_record {
    frontier_true,
    frontier_reduction,
    frontier_method
};

template<frontier_record R>
struct frontier_record_argument {
    static constexpr frontier_record value = R;

    static_assert( value != frontier_record::frontier_method,
		   "this case covered differently" );
};

static constexpr auto all_true =
    frontier_record_argument<frontier_record::frontier_true>();
static constexpr auto reduction =
    frontier_record_argument<frontier_record::frontier_reduction>();

template<typename T>
struct is_frontier_record : public std::false_type { };

template<frontier_record R>
struct is_frontier_record<frontier_record_argument<R>>
    : public std::true_type { };

template<typename T>
constexpr bool is_frontier_record_v = is_frontier_record<T>::value;

/************************************************************************
 * Defintion of filter recording method
 ************************************************************************/
template<typename EMapConfig, bool IsPriv, typename Operator,
	 typename Enable = void>
struct arg_record_reduction_op {
    static constexpr frontier_type ftype = EMapConfig::wr_ftype;
    static constexpr bool is_priv = IsPriv;
    using StoreTy = typename frontier_params<ftype,EMapConfig::VL>::type;
    using array_ty =
	typename expr::array_select<ftype,StoreTy,VID,expr::aid_frontier_new_f,
				    array_encoding<StoreTy>>::type;
    
    static constexpr bool is_scan = true;

    arg_record_reduction_op( Operator op, frontier & fr,
			     const VID * degree )
	: m_degree( const_cast<VID *>( degree ) ), m_op( op ), m_array(),
	  m_frontier( fr ) { }

    template<typename VIDSrc, typename VIDDst, typename EIDEdge>
    auto relax( VIDSrc s, VIDDst d, EIDEdge e ) const {
	// |= will be evaluated as atomic if !is_priv
	auto dd = expr::remove_mask( d );
	return expr::set_mask(
	    expr::get_mask_cond( d ),
	    m_array[dd] |= expr::cast<StoreTy>( m_op.relax( s, dd, e ) )
	    );
    }

    template<typename VIDType>
    auto enabled( VIDType vid ) {
	return m_op.enabled( vid );
    }
    
    template<typename VIDType>
    auto active( VIDType vid ) {
	return m_op.active( vid );
    }
    
    template<typename VIDType>
    auto update( VIDType vid ) {
	return m_op.update( vid );
    }
    
    template<typename VIDType>
    auto vertexop( VIDType vid ) {
	// Count active vertices and edges
	static_assert( ftype != frontier_type::ft_true
		       && ftype != frontier_type::ft_sparse
		       && ftype != frontier_type::ft_unbacked );

	expr::array_intl<VID,VID,expr::aid_graph_degree,
			 array_encoding<VID>,false> degree;
	expr::array_intl<VID,VID,expr::aid_frontier_nactv,array_encoding<VID>,
		       false> nactv;
	expr::array_intl<EID,VID,expr::aid_frontier_nacte,array_encoding<EID>,
		       false> nacte;

	using Tr = simd::detail::mask_preferred_traits_type<
	    typename VIDType::type, VIDType::VL>;
	auto mask = expr::make_unop_cvt_to_mask<Tr>( m_array[vid] );
	return expr::make_seq(
	    m_op.vertexop( vid ),
	    nacte[expr::zero_val(vid)] +=
	    expr::add_predicate( expr::make_unop_cvt_type<EID>( degree[vid] ),
				 mask ),
	    nactv[expr::zero_val(vid)] +=
	    expr::add_predicate( expr::constant_val_one(vid), mask ) );
    }

    template<typename PSet>
    auto get_ptrset( const PSet & pset ) const {
	return expr::map_set_if_absent<
	    (unsigned)aid_key(expr::array_aid(expr::aid_frontier_new))>(
		expr::map_set_if_absent<
		(unsigned)aid_key(expr::array_aid(expr::aid_frontier_nacte))>(
		    expr::map_set_if_absent<
		    (unsigned)aid_key(expr::array_aid(expr::aid_frontier_nactv))>(
			expr::map_set_if_absent<
			(unsigned)aid_key(expr::array_aid(expr::aid_graph_degree))>(
			    m_op.get_ptrset( pset ),
			    m_degree ),
			m_frontier.nActiveVerticesPtr() ),
		    m_frontier.nActiveEdgesPtr() ),
		m_frontier.getDense<ftype>() );
    }

    auto get_ptrset() const { // TODO - redundant
	return get_ptrset( expr::create_map() );
    }

    const auto get_config() const { return m_op.get_config(); }

private:
    VID * m_degree;
    Operator m_op;
    array_ty m_array;
    frontier & m_frontier;
};

template<typename EMapConfig, bool IsPriv, typename Operator>
struct arg_record_reduction_op<
    EMapConfig,IsPriv,Operator,
    std::enable_if_t<EMapConfig::wr_ftype == frontier_type::ft_unbacked>> {
    // TODO:
    // This use of unbacked frontiers may be only correct in pull mode. Need a
    // separate vertex scan phase with update method for unbacked in push
    // mode. Depends on the detail - not recently tested
    static constexpr frontier_type ftype = EMapConfig::wr_ftype;
    static constexpr bool is_priv = IsPriv;
    
    static constexpr bool is_scan = true;

    arg_record_reduction_op( Operator op, frontier & fr,
			     const VID * degree )
	: m_degree( const_cast<VID *>( degree ) ), m_op( op ),
	  m_frontier( fr ) { }

    template<typename VIDSrc, typename VIDDst, typename EIDEdge>
    auto relax( VIDSrc s, VIDDst d, EIDEdge e ) const {
	return m_op.relax( s, d, e );
    }

    template<typename VIDType>
    auto enabled( VIDType vid ) {
	return m_op.enabled( vid );
    }
    
    template<typename VIDType>
    auto active( VIDType vid ) {
	return m_op.active( vid );
    }
    
    template<typename VIDType>
    auto update( VIDType vid ) {
	return m_op.update( vid );
    }
    
    template<typename VIDType>
    auto vertexop( VIDType vid ) {
	// Count active vertices and edges
	static_assert( ftype != frontier_type::ft_true
		       && ftype != frontier_type::ft_sparse );

	expr::array_intl<VID,VID,expr::aid_graph_degree,
			 array_encoding<VID>,false> degree;
	expr::array_intl<VID,VID,expr::aid_frontier_nactv,array_encoding<VID>,
		       false> nactv;
	expr::array_intl<EID,VID,expr::aid_frontier_nacte,array_encoding<EID>,
		       false> nacte;

	using Tr = simd::detail::mask_preferred_traits_type<
	    typename VIDType::type, VIDType::VL>;
	// TODO: use let for mask?
	auto mask = update( vid ); // return formula, evaluated after vertexop
	return expr::make_seq(
	    m_op.vertexop( vid ),
	    nacte[expr::zero_val(vid)] +=
	    expr::add_predicate( expr::make_unop_cvt_type<EID>( degree[vid] ),
				 mask ),
	    nactv[expr::zero_val(vid)] +=
	    expr::add_predicate( expr::constant_val_one(vid), mask ) );
    }

    template<typename PSet>
    auto get_ptrset( const PSet & pset ) const {
	return
	    expr::map_set_if_absent<
		(unsigned)aid_key(expr::array_aid(expr::aid_frontier_nacte))>(
		    expr::map_set_if_absent<
		    (unsigned)aid_key(expr::array_aid(expr::aid_frontier_nactv))>(
			expr::map_set_if_absent<
			(unsigned)aid_key(expr::array_aid(expr::aid_graph_degree))>(
			    m_op.get_ptrset( pset ),
			    m_degree ),
			m_frontier.nActiveVerticesPtr() ),
		    m_frontier.nActiveEdgesPtr() );
    }

    auto get_ptrset() const { // TODO - redundant
	return get_ptrset( expr::create_map() );
    }

    const auto get_config() const { return m_op.get_config(); }

private:
    VID * m_degree;
    Operator m_op;
    frontier & m_frontier;
};


template<frontier_type ftype, unsigned short VL>
struct record_store_type {
    using type = typename frontier_params<ftype,VL>::type;
};

template<unsigned short VL>
struct record_store_type<frontier_type::ft_unbacked,VL> {
    using type = void;
};

template<typename EMapConfig, bool IsPriv, typename Operator,
	 typename Fn, typename PtrSet>
struct arg_record_method_op {
    static constexpr frontier_type ftype = EMapConfig::wr_ftype;
    static constexpr bool is_priv = IsPriv;
    using StoreTy = typename record_store_type<ftype,EMapConfig::VL>::type;
    using array_ty =
	typename expr::array_select<ftype,StoreTy,VID,expr::aid_frontier_new_f,
				    array_encoding<StoreTy>>::type;
    using ptrset_ty = PtrSet;
    
    static constexpr bool is_scan = true;

    arg_record_method_op( Operator op, frontier & fr,
			  const VID * degree,
			  Fn & method, ptrset_ty & ptrset )
	: m_degree( const_cast<VID *>( degree ) ), m_op( op ), m_array(),
	  m_frontier( fr ), m_method( method ), m_ptrset( ptrset ) { }

    template<typename VIDSrc, typename VIDDst, typename EIDEdge>
    auto relax( VIDSrc s, VIDDst d, EIDEdge e ) const {
	return m_op.relax( s, d, e );
    }

    template<typename VIDType>
    auto enabled( VIDType vid ) {
	return m_op.enabled( vid );
    }
    
    template<typename VIDType>
    auto active( VIDType vid ) {
	return m_op.active( vid );
    }
    
    template<typename VIDType>
    auto update( VIDType vid ) {
	return expr::rewrite_internal( m_method( vid ) );
    }
    
    template<typename VIDType>
    auto vertexop( VIDType vid ) {
	// Count active vertices and edges
	static_assert( ftype != frontier_type::ft_true && ftype != frontier_type::ft_sparse );

	expr::array_intl<VID,VID,expr::aid_graph_degree,
			 array_encoding<VID>,false> degree;
	expr::array_intl<VID,VID,expr::aid_frontier_nactv,
			 array_encoding<VID>,false> nactv;
	expr::array_intl<EID,VID,expr::aid_frontier_nacte,
			 array_encoding<EID>,false> nacte;

	if constexpr ( ftype == frontier_type::ft_unbacked ) {
	    auto mask = expr::rewrite_internal( m_method( vid ) );
	    return expr::make_seq(
		m_op.vertexop( vid ),
		nacte[expr::zero_val(vid)] +=
		expr::add_predicate( expr::make_unop_cvt_type<EID>( degree[vid] ),
				     mask ),
		nactv[expr::zero_val(vid)] +=
		expr::add_predicate( expr::constant_val_one(vid), mask ) );
	} else if constexpr ( is_priv ) {
	    // TODO: consider use of let to hold mask
	    auto mask = expr::rewrite_internal( m_method( vid ) );
	    return expr::make_seq(
		m_op.vertexop( vid ),
		m_array[vid] = expr::make_unop_switch_to_vector( mask ),
		nacte[expr::zero_val(vid)] +=
		expr::add_predicate( expr::make_unop_cvt_type<EID>( degree[vid] ),
				     mask ),
		nactv[expr::zero_val(vid)] +=
		expr::add_predicate( expr::constant_val_one(vid), mask ) );
	} else {
	    // TODO: consider use of let to hold mask
	    auto mask = expr::rewrite_internal( m_method( vid ) );
	    return expr::make_seq(
		m_op.vertexop( vid ),
		m_array[vid] |= expr::make_unop_switch_to_vector( mask ),
		nacte[expr::zero_val(vid)] +=
		expr::add_predicate( expr::make_unop_cvt_type<EID>( degree[vid] ),
				     mask ),
		nactv[expr::zero_val(vid)] +=
		expr::add_predicate( expr::constant_val_one(vid), mask ) );
	}
    }

    template<typename PSet>
    auto get_ptrset( const PSet & pset ) const {
	auto common = expr::map_set_if_absent<
	    (unsigned)aid_key(expr::array_aid(expr::aid_frontier_nacte))>(
		expr::map_set_if_absent<
		(unsigned)aid_key(expr::array_aid(expr::aid_frontier_nactv))>(
		    expr::map_set_if_absent<
		    (unsigned)aid_key(expr::array_aid(expr::aid_graph_degree))>(
			map_merge( m_op.get_ptrset( pset ), m_ptrset ),
			m_degree ),
		    m_frontier.nActiveVerticesPtr() ),
		m_frontier.nActiveEdgesPtr() );
	if constexpr ( ftype == frontier_type::ft_unbacked )
	    return common;
	else
	    return expr::map_set_if_absent<
		(unsigned)aid_key(expr::array_aid(expr::aid_frontier_new))>(
		    common, m_frontier.getDense<ftype>() );
    }

    auto get_ptrset() const { // TODO - redundant
	return get_ptrset( expr::create_map() );
    }

    const auto get_config() const { return m_op.get_config(); }

private:
    VID * m_degree;
    Operator m_op;
    array_ty m_array;
    frontier & m_frontier;
    Fn m_method;
    ptrset_ty m_ptrset;
};

template<filter_strength S, frontier_record R>
struct arg_record {
    static constexpr filter_strength strength = S; // weak implies unbacked ok
    static constexpr frontier_record record = R;
    static constexpr bool may_be_unbacked = strength == filter_strength::weak;

    static_assert( record != frontier_record::frontier_method, "required" );

    arg_record( frontier & f ) : m_frontier( f ) { };
    // arg_record( frontier && f ) : m_frontier( f ) { };

    template<typename EMapConfig, bool IsPriv, typename Operator>
    auto record_and_count( const VID * degree, Operator op ) {
	if constexpr ( record == frontier_record::frontier_reduction ) {
	    return arg_record_reduction_op<EMapConfig,IsPriv,Operator>(
		op, m_frontier, degree );
	} else // frontier_record::frontier_true
	    return op;
    }

    template<frontier_type ftype>
    void setup( const partitioner & part ) {
	m_frontier = frontier::create<ftype>( part );
    }

    template<typename GraphType>
    frontier & get_frontier( const GraphType & G ) { return m_frontier; }

private:
    frontier & m_frontier;
};

// Strong/weak here implies unbacked or not? Better to replace with other terms
// to make it more clear
template<filter_strength S, typename Fn>
struct arg_record_m {
    static constexpr filter_strength strength = S;
    static constexpr frontier_record record = frontier_record::frontier_method;
    static constexpr bool may_be_unbacked = strength == filter_strength::weak;

    arg_record_m( frontier & f, Fn method )
	: m_frontier( f ), m_method( method ) { }

    template<typename EMapConfig, bool IsPriv, typename Operator>
    auto record_and_count( const VID * degree, Operator op ) {
	static_assert( record == frontier_record::frontier_method,
		       "specialised class" );
	auto d = expr::value<simd::ty<VID,1>,expr::vk_dst>();
	auto ptrset = expr::extract_pointer_set( m_method( d ) );
	return arg_record_method_op<
	    EMapConfig,IsPriv,Operator,Fn,decltype(ptrset)>(
		op, m_frontier, degree, m_method, ptrset );
    }

    template<frontier_type ftype>
    void setup( const partitioner & part ) {
	m_frontier = frontier::create<ftype>( part );
    }

    template<typename GraphType>
    frontier & get_frontier( const GraphType & G ) { return m_frontier; }

private:
    frontier & m_frontier;
    Fn m_method;
};

struct missing_record_argument {
    static constexpr filter_strength strength = filter_strength::weak;
    static constexpr frontier_record record = frontier_record::frontier_true;
    static constexpr bool may_be_unbacked = true;

    template<typename EMapConfig, bool IsPriv, typename Operator>
    auto record_and_count( const VID * degree, Operator op ) {
	return op;
    }
    template<frontier_type>
    void setup( const partitioner & part ) { }
};

template<typename Record = frontier_record_argument<frontier_record::frontier_reduction>,
	 typename Strength = filter_strength_argument<filter_strength::strong>>
std::enable_if_t<
    is_frontier_record<Record>::value && is_filter_strength<Strength>::value,
    arg_record<Strength::value,Record::value>>
record( frontier & f,
	Record r = reduction,
	Strength s = strong ) {
    return arg_record<Strength::value,Record::value>( f );
}
	     
template<typename Fn,
	 typename Strength = filter_strength_argument<filter_strength::strong>>
std::enable_if_t<
    is_active<Fn>::value && is_filter_strength<Strength>::value,
    arg_record_m<Strength::value,Fn>>
record( frontier & f,
	Fn && fn,
	Strength s = strong ) {
    return arg_record_m<Strength::value,Fn>( f, std::forward<Fn>( fn ) );
}
	     
#if 0
template<typename... Args>
auto record( Args &&... args ) {
    if constexpr( check_arguments_3<is_filter_strength,is_frontier_record,
		  is_frontier,Args...>::value ) {
	return arg_record<
	    get_argument_type_t<is_filter_strength,decltype(strong),Args...>::value,
	    get_argument_type_t<is_frontier_record,decltype(reduction),Args...>::value>(
		get_argument_value<is_frontier,missing_record_argument>(
		    args... ) );
    } else if constexpr ( check_arguments_3<is_filter_strength,
			  is_frontier,is_active,Args...>::value ) {
	auto & f = get_argument_value<is_frontier,missing_filter_argument>(
	    args... );
	auto & m = get_argument_value<is_active,missing_record_argument>(
	    args... );
	return arg_record_m<
	    get_argument_type_t<is_filter_strength,decltype(strong),
				Args...>::value,decltype(m)>( f, m );
    } else
	; // static_assert( false, "unexpected argument(s) supplied" );
}
#endif

template<typename T>
struct is_record : public std::false_type { };

template<filter_strength S, frontier_record R>
struct is_record<arg_record<S,R>> : public std::true_type { };

template<filter_strength S, typename Fn>
struct is_record<arg_record_m<S,Fn>> : public std::true_type { };

/************************************************************************
 * Defintion of configuration option
 ************************************************************************/
template<parallelism_spec P, unsigned short VL, typename threshold_type>
struct arg_config {
    static constexpr parallelism_spec parallelism = P;
    static constexpr vectorization_spec<VL> vectorization =
	vectorization_spec<VL>();

    arg_config( const threshold_type & t ) : m_threshold( t ) { }

    static constexpr bool is_parallel() {
	return parallelism == parallelism_spec::parallel;
    }
    static constexpr bool is_vectorized() {
	return vectorization.value > 1;
    }
    static constexpr unsigned short max_vector_length() {
	return vectorization.value;
    }
    threshold_type get_threshold() const { return m_threshold; }

private:
    threshold_type m_threshold;
};

struct missing_config_argument {
    static constexpr parallelism_spec parallelism = parallelism_spec::parallel;
    static constexpr vectorization_spec vectorization = vl_max<MAX_VL>();

    static constexpr bool is_parallel() {
	return parallelism == parallelism_spec::parallel;
    }
    static constexpr bool is_vectorized() {
	return vectorization.value > 1;
    }
    static constexpr unsigned short max_vector_length() {
	return vectorization.value;
    }
    default_threshold get_threshold() const { return default_threshold(); }
};

template<typename... Args>
auto config( Args &&... args ) {
    if constexpr( check_arguments_3<is_parallelism,
		  is_vectorization,is_threshold,Args...>::value ) {
	auto & t = get_argument_value<is_threshold,default_threshold>(
	    args... );
	return arg_config<
	    get_argument_type_t<is_parallelism,decltype(parallel),Args...>::value,
	    get_argument_type_t<is_vectorization,decltype(vl_max<MAX_VL>()),Args...>::value,
	    get_argument_type_t<is_threshold,default_threshold,Args...>>( t );
    } else
	return 0; // static_assert( false, "unexpected argument(s) supplied" );
}

template<typename T>
struct is_config : public std::false_type { };

template<parallelism_spec P, unsigned short VL, typename threshold_type>
struct is_config<arg_config<P,VL,threshold_type>> : public std::true_type { };

/************************************************************************
 * Defintion of relax method
 ************************************************************************/
template<typename Fn>
using is_relax_method =
    std::is_invocable<Fn,
		      expr::value<simd::ty<VID,1>,expr::vk_src>,
		      expr::value<simd::ty<VID,1>,expr::vk_dst>,
		      expr::value<simd::ty<EID,1>,expr::vk_edge>>;

template<typename Fn>
using is_vertexop_method =
    std::is_invocable<Fn,expr::value<simd::ty<VID,1>,expr::vk_dst>>;

template<typename RFn>
struct arg_relax {
    arg_relax( RFn method ) : m_method( method ) { }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) const {
	return expr::make_noop();
    }
	
    const RFn m_method;
};

template<typename RFn, typename VFn>
struct arg_relax_vop {
    arg_relax_vop( RFn method, VFn vertexop )
	: m_method( method ), m_vertexop( vertexop ) { }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) const {
	return m_vertexop( d );
    }
    const RFn m_method;
    const VFn m_vertexop;
};

struct missing_relax_argument { };

struct missing_vertexop_argument {
    template<typename VIDDst>
    auto vertexop( VIDDst d ) const {
	return expr::make_noop();
    }
};

template<typename... Args>
auto relax( Args && ... args ) {
    auto & fn = get_argument_value<is_relax_method,missing_relax_argument>(
	args... );

    static_assert( !std::is_same_v<std::decay_t<decltype(fn)>,
		   missing_relax_argument>,
		   "must specify relax method to relax operation" );

    auto & vo = get_argument_value<is_vertexop_method,missing_vertexop_argument>(
	args... );
    if constexpr ( !std::is_same_v<std::decay_t<decltype(vo)>,missing_vertexop_argument> ) {
	return arg_relax_vop<std::decay_t<decltype(fn)>,
			     std::decay_t<decltype(vo)>>( fn, vo );
    } else {
	return arg_relax<std::decay_t<decltype(fn)>>( fn );
    }
}

template<typename T>
struct is_relax : public std::false_type { };

template<typename Fn>
struct is_relax<arg_relax<Fn>> : public std::true_type { };

template<typename Fn, typename Vo>
struct is_relax<arg_relax_vop<Fn,Vo>> : public std::true_type { };

/************************************************************************
 * Construct conventional operator class
 ************************************************************************/
template<typename Rlx,
	 typename Fsrc,
	 typename Fdst,
	 typename Adst,
	 typename Rec,
	 typename Cfg>
struct op_def {
    static constexpr frontier_mode new_frontier =
	std::is_same_v<Rec,missing_record_argument> ? fm_all_true
	: std::decay_t<Rec>::record == frontier_record::frontier_reduction ? fm_reduction
	: fm_calculate;

    static constexpr bool is_scan =
	!std::is_same_v<Rec,missing_record_argument>;

    // Weak frontiers on source of edge may be omitted
    static constexpr bool may_omit_frontier_rd =
	std::is_same_v<Fsrc,missing_filter_argument>
	|| Fsrc::strength == filter_strength::weak;

    // Can only omit writing a frontier if user did not ask for it.
    static constexpr bool may_omit_frontier_wr =
	std::is_same_v<Rec,missing_record_argument>;

    static constexpr bool new_frontier_dense = false; // to be removed

    op_def( Rlx && relax, Fsrc && filter_src, Fdst && filter_dst,
	    Adst && active_dst, Rec && record, Cfg && config )
	: m_relax( std::forward<Rlx>( relax ) ),
	  m_filter_src( std::forward<Fsrc>( filter_src ) ),
	  m_filter_dst( std::forward<Fdst>( filter_dst ) ),
	  m_active_dst( std::forward<Adst>( active_dst ) ),
	  m_record( std::forward<Rec>( record ) ),
	  m_config( std::forward<Cfg>( config ) ) { }

    template<typename VIDSrc, typename VIDDst, typename EIDEdge>
    auto relax( VIDSrc s, VIDDst d, EIDEdge e ) const {
	// This is primarily used for determing VL and frontier byte widths.
	// Frontiers will be added in later.
	auto ss = expr::remove_mask( s );
	auto dd = expr::remove_mask( d );
	auto ee = expr::remove_mask( e );
	auto m = expr::get_mask_cond( s )
	    && expr::get_mask_cond( d )
	    && expr::get_mask_cond( e );
	// set_mask sets mask on lhs before evaluating rhs
	return expr::set_mask( m, m_relax.m_method( ss, dd, ee ) );
    }

    // Should source vertex be traversed?
    template<typename VIDType>
    auto enabled( VIDType vid ) const {
	auto vv = expr::remove_mask( vid );
	return expr::set_mask( expr::get_mask_cond( vid ),
			       expr::true_val( vv ) );
    }
    
    // Should destination vertex be traversed?
    template<typename VIDType>
    auto active( VIDType vid ) const {
	auto vv = expr::remove_mask( vid );
	return expr::set_mask( expr::get_mask_cond( vid ),
			       expr::true_val( vv ) );
    }
    
    template<typename VIDType>
    auto update( VIDType vid ) const {
	auto vv = expr::remove_mask( vid );
	return expr::set_mask( expr::get_mask_cond( vid ),
			       expr::true_val( vv ) );
    }
    
    template<typename VIDDst>
    auto vertexop( VIDDst d ) const {
	auto dd = expr::remove_mask( d );
	auto m = expr::get_mask_cond( d );
	auto e = m_relax.vertexop( dd );
	return expr::set_mask( m, e );
    }

    template<typename PSet>
    auto get_ptrset( const PSet & pset ) const {
	auto s = expr::value<simd::ty<VID,1>,expr::vk_src>();
	auto d = expr::value<simd::ty<VID,1>,expr::vk_dst>();
	auto e = expr::value<simd::ty<EID,1>,expr::vk_edge>();
	return map_merge(
	    expr::extract_pointer_set_with( pset, relax( s, d, e ) ),
	    expr::extract_pointer_set( vertexop( d ) ) );
    }

    auto get_ptrset() const { // TODO - redundant
	return get_ptrset( expr::create_map() );
    }

    bool is_true_src_frontier() const {
	return m_filter_src.is_true_frontier();
    }
    bool is_true_dst_frontier() const {
	return m_filter_dst.is_true_frontier();
    }

    template<graph_traversal_kind gtk, bool SFtrue, bool DFtrue,
	     bool IsPriv, typename cfg>
    auto variant( const VID * degree ) {
	// In order to support both source and destination frontiers, they
	// would need different AID
	assert( ( SFtrue || DFtrue ) && "not supporting two frontiers" );

	if constexpr ( gtk == graph_traversal_kind::gt_push )
	    return push_variant<typename cfg::push,SFtrue,DFtrue,IsPriv>( degree );
	else if constexpr ( gtk == graph_traversal_kind::gt_pull )
	    return pull_variant<typename cfg::pull,SFtrue,DFtrue,IsPriv>( degree );
	else if constexpr ( gtk == graph_traversal_kind::gt_ireg )
	    return ireg_variant<typename cfg::ireg,SFtrue,DFtrue,IsPriv>( degree );
	UNREACHABLE_CASE_STATEMENT;
    }

private:
    template<typename cfg_push, bool SFtrue, bool DFtrue, bool IsPriv>
    auto push_variant( const VID * degree ) {
	// Setting up push-style operator.
	// filter_src: A push style code template will perform frontier checks
	//             and perform control flow on the basis of this. So we do
	//             not need to include checks here.
	// filter_dst: A push style code template needs to have these checks
	//             inserted in the operator as it won't perform them itself.
	//             The checks are omitted if they are weak.
	// active_dst: A push style code template needs to have these checks
	//             inserted in the operator as it won't perform them itself.
	//             The checks are omitted if they are weak.
// assert( SFtrue && "oops where did this frontier go?" );
	return m_record.template record_and_count<cfg_push,IsPriv>(
	    degree,
	    filter_src<cfg_push,SFtrue,graph_traversal_kind::gt_push>(
		filter_dst<cfg_push,DFtrue,graph_traversal_kind::gt_push>(
		    *this ) ) );
    }

    template<typename cfg_pull, bool SFtrue, bool DFtrue, bool IsPriv>
    auto pull_variant( const VID * degree ) {
	// Setting up pull-style operator.
	// filter_src: A pull style code template needs to have these checks
	//             inserted in the operator as it won't perform them itself.
	//             The checks are omitted if they are weak.
	// filter_dst: A pull style code template will perform active checks
	//             and perform control flow on the basis of this. So we
	//             include the checks in active if weak, and we also
	//             include the checks in relax if strong
	// active_dst: A pull style code template will perform active checks
	//             and perform control flow on the basis of this. So we
	//             include the checks in active, not in relax.
	return m_record.template record_and_count<cfg_pull,IsPriv>(
	    degree,
	    filter_dst<cfg_pull,DFtrue,graph_traversal_kind::gt_pull>(
		filter_src<cfg_pull,SFtrue,graph_traversal_kind::gt_pull>(
		    *this ) ) );
    }

    template<typename cfg, bool SFtrue,
	     graph_traversal_kind gtk,
	     typename Operator>
    auto filter_src( Operator op ) {
	if constexpr ( SFtrue )
	    return op;
	else
	    return m_filter_src.template check_strength<cfg,gtk>( op );
    }

    template<typename cfg, bool DFtrue,
	     graph_traversal_kind gtk,
	     typename Operator>
    auto filter_dst( Operator op ) {
	if constexpr ( DFtrue )
	    return m_active_dst.template check_strength<cfg,gtk>( op );
	else
	    return m_active_dst.template check_strength<cfg,gtk>(
		m_filter_dst.template check_strength<cfg,gtk>( op ) );
    }

    template<typename cfg_ireg, bool SFtrue, bool DFtrue, bool IsPriv>
    auto ireg_variant( const VID * degree ) {
	// Setting up irregular operator (COO).
	// filter_src: An irregular code template needs to have these checks
	//             inserted in the operator as it won't perform them itself.
	//             The checks are omitted if they are weak.
	// filter_dst: An irregular code template needs to have these checks
	//             inserted in the operator as it won't perform them itself.
	//             The checks are omitted if they are weak.
	// active_dst: An irregular code template needs to have these checks
	//             inserted in the operator as it won't perform them itself.
	//             The checks are omitted if they are weak.
	return m_record.template record_and_count<cfg_ireg,IsPriv>(
	    degree,
	    filter_src<cfg_ireg,SFtrue,graph_traversal_kind::gt_ireg>(
		filter_dst<cfg_ireg,DFtrue,graph_traversal_kind::gt_ireg>(
			*this ) ) );
    }

public:
    const Cfg & get_config() const { return m_config; }

private:
    Rlx m_relax;
    Fsrc m_filter_src;
    Fdst m_filter_dst;
    Adst m_active_dst;
    Rec m_record;
    Cfg m_config;
};

template<typename Rlx,
	 typename Fsrc,
	 typename Fdst,
	 typename Adst,
	 typename Rec,
	 typename Cfg>
auto op_create( Rlx && rlx, Fsrc && fsrc, Fdst && fdst, Adst && adst,
		Rec && rec, Cfg && cfg ) {
    return op_def<std::decay_t<Rlx>,
		  std::decay_t<Fsrc>,
		  std::decay_t<Fdst>,
		  std::decay_t<Adst>,
		  std::decay_t<Rec>,
		  std::decay_t<Cfg>>(
		      std::forward<std::decay_t<Rlx>>( rlx ),
		      std::forward<std::decay_t<Fsrc>>( fsrc ),
		      std::forward<std::decay_t<Fdst>>( fdst ),
		      std::forward<std::decay_t<Adst>>( adst ),
		      std::forward<std::decay_t<Rec>>( rec ),
		      std::forward<std::decay_t<Cfg>>( cfg ) );
}

/*
template<filter_entity E, filter_strength S>
struct arg_filter {
    ) {
    return op_relax<Operator,Fn>( arg );
}
*/

/************************************************************************
 * sparse frontiers
 ************************************************************************/
template<typename GraphType>
void sparsify( const GraphType & GA, frontier & F ) {
    const partitioner &part = GA.get_partitioner();

    EID m = GA.numEdges();
    EID threshold = m/20;

    VID nactv = F.nActiveVertices();
    EID nacte = F.nActiveEdges();

    if( nactv + nacte <= threshold )
	F.toSparse( part );
}


/************************************************************************
 * \page page_emap Higher-order methods on edge sets
 *
 * The #edgemap method maps a "relax" method to all of the edges of graph
 * #GA, subject to filters based on frontiers (sets) or methods. It can
 * also optionally record a new frontier. The #edgemap method dynamically
 * chooses between sparse (work-list) based and dense traversal; it selects
 * between push (#graph_traversal_kind::gt_push), pull
 * (#graph_traversal_kind::gt_pull), and (#graph_traversal_kind::gt_irreg) irregular
 * variants. The latter correspond, e.g., to a coordinate list graph format.
 *
 * The arguments to #edgemap consist of the graph, followed by a number
 * of options to set filters and record new frontiers. Each argument consists
 * of instantiating specific classes. Following options are available:
 * \li #filter( #filter_entity, #filter_strength, #frontier ) \n
 *     Edges are filtered out by looking up an endpoint of each edge in the
 *     frontier argument. Filtering occurs on the sources of edges if the
 *     #src is specified, on the destinations if #dst is specified.
 *     Filtering is mandatory (e.g., for correctness reasons) if #filter_strength
 *     is #strong. It is optional if #filter_strength is #weak. In the latter
 *     case, the system is free to choose whether the filter is applied, or
 *     if all edges may be processed. The latter is faster when the frontier
 *     is highly dense.
 * \li #filter( #filter_entity, #filter_strength, lambda ) \n
 *     Similar to the filter with frontiers, but here filtering occurs by
 *     evaluating the lambda method on the vertex ID. The method must satisfy
 *     #is_active and take one argument of type 'auto' (it is a syntax tree).
 *     The lambda method returns an expression syntax tree. If the expression
 *     evaluates to true, then edges are skipped
 *    (filtered out).
 * \li #record( #frontier, #frontier_record, #filter_strength ) \n
 * \li #record( #frontier, #frontier_record, lambda ) \n
 *     Records a new frontier.
 *     If a #frontier_record of #frontier_record::frontier_true is specified,
 *     then all vertices become active and a frontier of type #frontier_type::ft_true
 *     is returned.
 *     If a #frontier_record of #frontier_record::frontier_reduction is specified,
 *     then vertices are added to the frontier if the relax method evaluates to
 *     true for any of their incoming edges.
 *     If a #frontier_record of #frontier_record::frontier_method is specified,
 *     then for each vertex, the specified lambda is evaluated after all the incoming
 *     edges to this vertex have been processed. The vertex is added to the frontier
 *     if the lambda method evaluates to true. The lambda must satisfy #is_active,
 *     have one argument of type 'auto' (a syntax tree) and must return a syntax tree
 *     that can be evaluated to determine whether the vertex is active.
 *     The method is applied only to vertices for which incoming edges have been
 *     processed. If no incoming edge has been processed, then the vertex will be
 *     absent from the recorded frontier.
 * \li relax( lambda ) \n
 *     The lambda method must satisfy #is_relax_method: taking 3 arguments of
 *     type 'auto', in this order: source vertex ID, destination vertex ID,
 *     edge vertex ID. All arguments are syntax trees and the method must
 *     return a syntax tree. Upon evaluation of the syntax
 *     tree, the return value is a boolean that indicates whether properties for the
 *     vertex have been updated. The return value is used only to populate the next
 *     frontier when a #record argument is given of type
 *     #frontier_record::frontier_reduction.
 * \li api_config( ... ) \n
 *     Configuration of parallelism/sequential execution, vector length, or
 *     sparse/dense choice.
 *
 * Currently, it is not possible to apply filters using frontiers simultaneously
 * on sources and destinations of edges. One filter should be a frontier, while the
 * other uses a method.
 * Record methods are not currently supported in sparse traversals.
 *
 * In a pull traversal with #weak strength, an unbacked frontier may be
 * generated, i.e., a frontier containing active counts for vertices
 * and edges but no detailed information on which vertices are active.
 * 
 * The #edgemap method will dynamically select between a push, pull or irregular
 * traversal depending on the filters: if filters are applied to destinations,
 * then a pull traversal will see more performance benefit in the filtering.
 * If filters are applied to sources, then a push traversal will result in a higher
 * reduction of work.
 *
 * The choice between traversal types is specified by the graph data type, method
 * select_traversal, such that the most appropriate choice can be made based on
 * traversal types supported by the graph data type.
 *
 * Use case example:
 * \include{lineno} ex_emap.hh
 *
 * \note
 * The edgemap method returns an instance of #lazy_executor, i.e., it may
 * not perform the map operation yet, but instead defer and register in
 * a #lazy_executor. Therefore, the method materialize()
 * must be called on the returned object.
 *
 * \note
 * Methods specified in the domain-specific language are not called for every
 * vertex or edge. They are called once (perhaps a few times) in order to
 * construct a syntax tree that represents the desired computations. As
 * such, the methods must always return a value, otherwise they are perceived
 * as a no-op.
 ************************************************************************/

/************************************************************************
 * \brief Apply method to all edges, subject to filters
 *
 * For details on edgemap, see \ref page_emap.
 *
 * \param GA graph
 * \param ... filter, record and relax specification
 ************************************************************************/
template<typename GraphType, typename... Args>
static auto DBG_NOINLINE edgemap( const GraphType & GA, Args &&... args ) {
    // Sanity check: we must have a relax operator to apply
    static_assert( has_argument_v<is_relax,Args...>,
		   "must specify relax operation" );
    // Sanity check: limited set of arguments allowed
/*
    static_assert( check_arguments_4<is_relax_method,is_filter_for<src>,
		   is_filter_for<dst>,is_filter_method,Args...>::value,
		   "argument list constraint violated" );
		   */

    // Some context
    const partitioner & part = GA.get_partitioner();
    EID m = GA.numEdges();

    // What are trying to do?
    auto relax = get_argument_value<is_relax,missing_relax_argument>( args... );
    auto filter_src =
	get_argument_value<is_filter_for_src,missing_filter_argument>(
	    args... );
    auto filter_dst =
	get_argument_value<is_filter_for_dst,missing_filter_argument>(
	    args... );
    auto active_dst =
	get_argument_value<is_filter_method,missing_filter_argument>( args... );
    auto record =
	get_argument_value<is_record,missing_record_argument>( args... );

    auto config
	= get_argument_value<is_config,missing_config_argument>( args... );

    static_assert( std::is_same_v<std::decay_t<decltype(active_dst)>,
		   missing_filter_argument>
		   || decltype(active_dst)::entity == filter_entity::dst,
		   "filter method only applicable to destination" );

    // Build operator
    auto op = op_create( relax, filter_src, filter_dst, active_dst, record,
			 config );
    using Operator = decltype(op);

    // Analyse operator
    constexpr bool rd_frontier =
		  !std::is_same_v<decltype(filter_src),missing_filter_argument>;
    constexpr bool wr_frontier =
		  !std::is_same_v<decltype(filter_dst),missing_filter_argument>
		  && !std::is_same_v<decltype(active_dst),missing_filter_argument>;

    // TODO: take into account how filtering is done. E.g., in destination
    //       filtering in pull, the read frontier is sequential access like
    //       the written frontier
    using cfg = expr::determine_emap_config<
	VID,Operator,GraphType,decltype(record)::may_be_unbacked>;
    using cfg_pull = typename cfg::pull;
    using cfg_push = typename cfg::push;
    using cfg_ireg = typename cfg::ireg;
    using cfg_scalar = typename cfg::scalar;

    // Track if through all of this, the edgemap operation has been applied
    // in some manner
    bool applied_emap = false;
    bool need_record = false;

    // If a filter has been specified on edge sources: check if it is
    // sparse; if so execute and complete.
    if constexpr ( has_argument_v<is_filter_for_src,Args...> ) {
	frontier & F = filter_src.get_frontier( GA );
	VID nactv = F.nActiveVertices();
	EID nacte = F.nActiveEdges();

	bool do_sparse = config.get_threshold().is_sparse( nactv, nacte, m )
	    && !applied_emap;
	if( do_sparse && F.getType() == frontier_type::ft_unbacked ) {
	    // We cannot convert to a sparse frontier as we are missing detail.
	    // Perform another dense iteration with recording forced
	    need_record = true;
	} else if( do_sparse ) {
	    F.toSparse( part ); 	 	 // ensure using sparse format
	    assert( F.getType() == frontier_type::ft_sparse
		    && "Conversion to sparse frontier failed" );

	    // Scalar push op
	    auto sparse_op =
		filter_dst.template
		check_strength<cfg_scalar,graph_traversal_kind::gt_push>(
		    active_dst.template
		    check_strength<cfg_scalar,graph_traversal_kind::gt_push>(
			op ) );

	    // Do we need to create a new frontier?
	    if constexpr ( !std::is_same_v<decltype(record),
			   missing_record_argument> ) {
		frontier & G = record.get_frontier( GA );
		
		// Do sparse edgemap and construct new frontier. Note that
		// if a new frontier is constructed using a method, we will
		// not currently use the method.
		if constexpr ( record.record == frontier_record::frontier_true )
		    G = csr_sparse_no_f(
			config,
			GA.getCSR(), GA.get_eid_retriever(), F, sparse_op );
		else if constexpr( record.record == frontier_record::frontier_reduction )
		    G = csr_sparse_with_f(
			config,
			GA.getCSR(), GA.get_eid_retriever(),
			GA.get_partitioner(), F, sparse_op );
		else if constexpr( record.record == frontier_record::frontier_method ) {
		    // Ignore the method (strong assumption on prog model) ...
		    // assert( 0 && "NYI" );
		    G = csr_sparse_with_f(
			config,
			GA.getCSR(), GA.get_eid_retriever(),
			GA.get_partitioner(), F, sparse_op );
		}
	    } else {
		// No argument supplied that tells us where to store frontier.
		frontier G = csr_sparse_no_f(
		    config,
		    GA.getCSR(), GA.get_eid_retriever(), F, sparse_op );
		G.del();
	    }

	    // The edgemap has been performed
	    applied_emap = true;
	}
    }

    // If a filter has been specified on edge destination: check if it is
    // sparse; if so execute and complete.
    if constexpr ( has_argument_v<is_filter_for_dst,Args...> ) {
	frontier & F = filter_dst.get_frontier( GA );
	VID nactv = F.nActiveVertices();
	EID nacte = F.nActiveEdges();

	bool do_sparse = config.get_threshold().is_sparse( nactv, nacte, m )
	    && !applied_emap;
	if( do_sparse && F.getType() == frontier_type::ft_unbacked ) {
	    // We cannot convert to a sparse frontier as we are missing detail.
	    // Perform another dense iteration with recording forced
	    need_record = true;
	} else if( do_sparse ) {
	    F.toSparse( part ); 	 	 // ensure using sparse format
	    assert( F.getType() == frontier_type::ft_sparse
		    && "Conversion to sparse frontier failed" );

	    // Scalar pull op. Sparse code template will apply filter_dst.
	    auto scalar_op =
		filter_src.template
		check_strength<cfg_scalar,graph_traversal_kind::gt_pull>( op );

	    // Do we need to create a new frontier?
	    if constexpr ( !std::is_same_v<decltype(record),
			   missing_record_argument> ) {
		frontier & G = record.get_frontier( GA );

		// Do sparse edgemap and construct new frontier. Note that
		// if a new frontier is constructed using a method, we will
		// not currently use the method.
		// TODO: this is only correct for symmetric graphs!
		assert( GA.getCSR().isSymmetric()
			&& "symmetry required in absence of getCSC()" );
		if constexpr ( record.record == frontier_record::frontier_true )
		    G = csc_sparse_aset_no_f(
			config, GA.getCSR(), part, scalar_op, F );
		else if constexpr ( record.record == frontier_record::frontier_reduction )
		    G = csc_sparse_aset_with_f(
			config, GA.getCSR(), part, scalar_op, F );
		else if constexpr ( record.record == frontier_record::frontier_method ) {
		    const VID * degree = GA.getOutDegree();
		    auto record_op = record.template
			record_and_count<cfg_scalar,true>( degree, scalar_op );
		    G = csc_sparse_aset_with_f_record(
			config, GA.getCSR(), part, record_op, F );
		}
	    } else {
		// TODO: this is only correct for symmetric graphs!
		assert( GA.getCSR().isSymmetric()
			&& "symmetry required in absence of getCSC()" );
		// No argument supplied that tells us where to store frontier.
		frontier G = csc_sparse_aset_no_f(
		    config, GA.getCSR(), part, scalar_op, F );
		G.del();
	    }

	    // The edgemap has been performed
	    applied_emap = true;
	}
    }

    // Determine if we prefer push / pull / irregular (COO) based on frontier
    // density, filters supplied and GraphType.
    // Override recording strength if unbacked frontier is sparse.
    frontier ftrue = frontier::all_true( GA.numVertices(), GA.numEdges() );
    frontier * gtk_frontier = &ftrue;
    if constexpr
	( !std::is_same_v<decltype(filter_src),missing_filter_argument> )
	gtk_frontier = &filter_src.get_frontier( GA );
    graph_traversal_kind gtk =
	applied_emap ? graph_traversal_kind::gt_sparse
	: GA.select_traversal(
	    filter_src.strength == filter_strength::strong,
	    filter_dst.strength == filter_strength::strong,
	    active_dst.strength == filter_strength::strong,
	    record.strength == filter_strength::strong || need_record,
	    *gtk_frontier,
	    config.get_threshold().is_sparse( *gtk_frontier, m ) && !need_record
	    );
    ftrue.del();

    // Error in control flow in this function...
    assert( ( applied_emap || gtk != graph_traversal_kind::gt_sparse )
	    && "Too late to decide on sparse traversal now..." );

    // Set up frontiers (memory allocation / conversion)
    switch( gtk ) {
    case graph_traversal_kind::gt_sparse: break; // done
    case graph_traversal_kind::gt_push:
	filter_src.template setup<cfg_push::rd_ftype>( part );
	filter_dst.template setup<cfg_push::rd_ftype>( part );
	record.template setup<cfg_push::wr_ftype>( part );
	break;
    case graph_traversal_kind::gt_pull:
	filter_src.template setup<cfg_pull::rd_ftype>( part );
	filter_dst.template setup<cfg_pull::rd_ftype>( part );
	record.template setup<cfg_pull::wr_ftype>( part );
	break;
    case graph_traversal_kind::gt_ireg:
	filter_src.template setup<cfg_ireg::rd_ftype>( part );
	filter_dst.template setup<cfg_ireg::rd_ftype>( part );
	record.template setup<cfg_ireg::wr_ftype>( part );
	break;
    default:
	UNREACHABLE_CASE_STATEMENT;
    }
	
    // Build lazy executor
    // Note: nothing to do if sparse frontier - all complete
    auto lax = make_lazy_executor( part )
	.template edge_map<cfg>( GA, op, gtk );

    return lax;
}

/************************************************************************
 * Representation of a vertex property. Can be used both as a syntax tree
 * element and as a native C++ array.
 *
 * Memory is allocated at construction time and has to be freed explicitly
 * by calling #del.
 * Memory allocation proceeds using a vertex-balanced partitioned allocation.
 * The array size as well as the dimensions of partitioning are taken
 * from the partitioner object supplied to the constructor.
 *
 * @see mm
 * @param <T> the type of array elements
 * @param <U> the type of the array index
 * @param <AID_> an integral identifier that uniquely identifies the
 *               property array
 * @param <Encoding> optional in-memory array encoding specification
 ************************************************************************/
template<typename T, typename U, short AID_,
	 typename Encoding = array_encoding<T>, bool NT = false>
class vertexprop : private NonCopyable<vertexprop<T,U,AID_,Encoding,NT>> {
public:
    using type = T; 	 	 	//!< type of array elements
    using index_type = U; 	 	//!< index type of property
    using encoding = Encoding; 	 	//!< data encoding in memory
    static constexpr short AID = AID_;	//!< array ID of property
    //! type of array syntax tree for this property
    using array_ty = expr::array_ro<type,index_type,AID,encoding,NT>;

    /** Constructor: create an vertex property.
     *
     * @param[in] name explanation string for debugging
     */
    vertexprop( const partitioner & part, const char * name )
	: mem( numa_allocation_partitioned( part ), m_name ),
	  m_name( name ) { }

    /** Constructor: create vertex property from file.
     *
     * @param[in] part graph partitioner
     * @param[in] fname file name
     * @param[in] name explanation string for debugging
     */
    vertexprop( const partitioner & part, const char * fname,
		const char * name )
	: m_name( name ) {
	mem.map_file( numa_allocation_partitioned( part ), fname, name );
    }

    /*! Factory creation method for a vertex property.
     *
     * @param[in] part graph partitioner object dictating size and allocation
     * @param[in] name explanation string for debugging
     */
    static vertexprop
    create( const partitioner & part, const char * name = nullptr ) {
	return vertexprop( part, name );
    }

    /*! Factory creation method reading data from file.
     *
     * @param[in] part graph partitioner object dictating size and allocation
     * @param[in] fname filename of the file containing the vertex property
     * @param[in] name explanation string for debugging
     */
    static vertexprop
    from_file( const partitioner & part, const char * fname,
	       const char * name = nullptr ) {
	return vertexprop( part, fname, name );
    }

    //! Release memory
    void del() {
	mem.del( m_name );
    }

    /*! Subscript operator overload for syntax trees
     * This operator builds a syntax tree that represents an array index
     * operation.
     *
     * @param[in] v Syntax tree element for the index
     * @return The syntax tree representing the indexing of the array
     */
    template<typename Expr>
    std::enable_if_t<expr::is_expr_v<Expr>,
		     decltype(array_ty(nullptr)[*(Expr*)nullptr])>
    operator[] ( const Expr & e ) const {
	static_assert( std::is_same_v<index_type, typename Expr::type>,
		       "requires a match of index_type" );
	return array_ty( mem.get() )[e];
    }

    /*! Subscript operator overload for native C++ array operation.
     * Indexes the array and returns the value at index #v of the array.
     * This operator returns an r-value; it cannot be used to modify array contents.
     *
     * @param[in] v array index
     * @return value found at array index #e
     */
    T operator[] ( VID v ) const {
	return encoding::template load<simd::ty<T,1>>( mem.get(), v );
    }

    typename encoding::stored_type * get_ptr() const {
	return mem.get();
    }

    const char * get_name() const { return m_name; }

private:
    mm::buffer<typename encoding::stored_type> mem;	//!< memory buffer
    const char * m_name;	//!< explanatory name describing vertex property
};

template<typename T, typename U, short AID, typename Enc, bool NT>
void print( std::ostream & os,
	    const partitioner & part,
	    const vertexprop<T,U,AID,Enc,NT> & vp ) {
    os << vp.get_name() << ':';
    VID n = part.get_vertex_range();
    for( VID v=0; v < n; ++v )
	os << ' ' << vp[v];
    os << '\n';
}

template<typename lVID,
	 typename T, typename U, short AID, typename Enc, bool NT>
void print( std::ostream & os,
	    const RemapVertexIdempotent<lVID> &,
	    const partitioner & part,
	    const vertexprop<T,U,AID,Enc,NT> & vp ) {
    print( os, part, vp );
}

template<typename lVID,
	 typename T, typename U, short AID, typename Enc, bool NT>
void print( std::ostream & os,
	    const RemapVertex<lVID> & remap,
	    const partitioner & part,
	    const vertexprop<T,U,AID,Enc,NT> & vp ) {
    os << vp.get_name() << ':';
    VID n = part.get_vertex_range();
    for( VID v=0; v < n; ++v )
	os << ' ' << vp[remap.remapID(v)];
    os << '\n';
}

/************************************************************************
 * Representation of an edge property. Can be used both as a syntax tree
 * element and as a native C++ array.
 *
 * Memory is allocated at construction time and has to be freed explicitly
 * by calling #del.
 * Memory allocation proceeds using an edge-balanced partitioned allocation.
 * The array size as well as the dimensions of edge partitioning are taken
 * from the partitioner object supplied to the constructor.
 *
 * @see mm
 * @param <T> the type of array elements
 * @param <U> the type of the array index
 * @param <AID_> an integral identifier that uniquely identifies the
 *               property array
 * @param <Encoding> optional in-memory array encoding specification
 ************************************************************************/
template<typename T, typename U, short AID_,
	 typename Encoding = array_encoding<T>>
class edgeprop : private NonCopyable<edgeprop<T,U,AID_,Encoding>> {
public:
    using type = T; 	 	 	//!< type of array elements
    using index_type = U; 	 	//!< index type of property
    using encoding = Encoding; 	 	//!< data encoding in memory
    static constexpr short AID = AID_;	//!< array ID of property
    //! type of array syntax tree for this property
    using array_ty = expr::array_ro<type,index_type,AID,encoding>;

    /** Constructor: create an edge property.
     *
     * @param[in] part graph partitioner object dictating size and allocation
     * @param[in] name explanation string for debugging
     */
    edgeprop( const partitioner & part, const char * name = nullptr )
	: mem( numa_allocation_edge_partitioned( part ), m_name ),
	  m_name( name ) {
    }

    /*! Factory creation method for an edge property.
     *
     * @param[in] part graph partitioner object dictating size and allocation
     * @param[in] name explanation string for debugging
     */
    static edgeprop
    create( const partitioner & part, const char * name = nullptr ) {
	return edgeprop( part, name );
    }

    //! Release memory
    void del() {
	mem.del( m_name );
    }

    /*! Subscript operator overload for syntax trees
     * This operator builds a syntax tree that represents an array index
     * operation.
     *
     * @param[in] e Syntax tree element for the index
     * @return The syntax tree representing the indexing of the array
     */
    template<typename Expr>
    std::enable_if_t<expr::is_expr_v<Expr>,
		     decltype(array_ty(nullptr)[*(Expr*)nullptr])>
    operator[] ( const Expr & e ) const {
	static_assert( std::is_same_v<index_type,
		       typename Expr::data_type::element_type>,
		       "requires a match of index_type" );
	return array_ty( mem.get() )[e];
    }

    /*! Subscript operator overload for native C++ array operation.
     * Indexes the array and returns the value at index #e of the array.
     * This operator returns an r-value; it cannot be used to modify array
     * contents.
     *
     * @param[in] e array index
     * @return value found at array index #e
     */
    T operator[] ( EID e ) const {
	return encoding::template load<simd::ty<T,1>>( mem.get(), e );
    }

    typename encoding::stored_type * get_ptr() const {
	return mem.get();
    }

    const char * get_name() const { return m_name; }

private:
    mm::buffer<typename encoding::stored_type> mem; //!< memory buffer
    const char * m_name;	//!< explanatory name describing edge property
};

/************************************************************************
 * Representation of a property for edge weights. This class does not
 * contain the actual weights, as the weight array is considered immutable
 * and the order in which weights are stored is specialised to the graph
 * data structure and layout.
 * Indexing produces a syntax tree. For safety reasons, the address is a
 * null pointer.
 *
 * The class has an interface that is compatible to that of the general
 * edgeprop class.
 *
 * @param <T> the type of array elements
 * @param <U> the type of the array index
 * @param <Encoding> optional in-memory array encoding specification
 ************************************************************************/
template<typename T, typename U, typename Encoding>
class edgeprop<T,U,expr::vk_eweight,Encoding>
    : private NonCopyable<edgeprop<T,U,expr::vk_eweight,Encoding>> {
public:
    using type = T; 	 	 	//!< type of array elements
    using index_type = U; 	 	//!< index type of property
    using encoding = Encoding; 	 	//!< data encoding in memory
    static constexpr short AID = expr::vk_eweight; //!< array ID of property
    //! type of array syntax tree for this property
    using array_ty = expr::array_ro<type,index_type,AID,encoding>;

    /** Constructor: create an edge property.
     *
     * @param[in] part graph partitioner object dictating size and allocation
     * @param[in] name explanation string for debugging
     */
    edgeprop( const partitioner & part, const char * name = nullptr )
	: m_name( name ) { }

    /*! Factory creation method for an edge property.
     *
     * @param[in] part graph partitioner object dictating size and allocation
     * @param[in] name explanation string for debugging
     */
    static edgeprop
    create( const partitioner & part, const char * name = nullptr ) {
	return edgeprop( part, name );
    }

    //! Release memory - noop
    void del() { }

    /*! Subscript operator overload for syntax trees
     * This operator builds a syntax tree that represents an array index
     * operation.
     *
     * @param[in] e Syntax tree element for the index
     * @return The syntax tree representing the indexing of the array
     */
    template<typename Expr>
    std::enable_if_t<expr::is_expr_v<Expr>,
		     decltype(array_ty(nullptr)[*(Expr*)nullptr])>
    operator[] ( const Expr & e ) const {
	static_assert( std::is_same_v<index_type,
		       typename Expr::data_type::element_type>,
		       "requires a match of index_type" );
	return array_ty( nullptr )[e];
    }

    /*! Subscript operator overload for native C++ array operation.
     * Indexes the array and returns the value at index #e of the array.
     * This operator is deleted for vk_eweight; the class does not
     * contain the data.
     *
     * @param[in] e array index
     * @return value found at array index #e
     */
    T operator[] ( EID e ) const = delete;

    typename encoding::stored_type * get_ptr() const {
	return nullptr;
    }

    const char * get_name() const { return m_name; }

private:
    const char * m_name;	//!< explanatory name describing edge property
};

template<typename T, typename U, short AID, typename Enc>
void print( std::ostream & os,
	    const partitioner & part,
	    const edgeprop<T,U,AID,Enc> & ep ) {
    os << ep.get_name() << ':';
    EID m = part.get_edge_range();
    if constexpr ( is_logical_v<T> ) {
	for( EID e=0; e < m; ++e )
	    os << ( ep[e] ? 'T' : '.' );
    } else {
	for( EID e=0; e < m; ++e )
	    os << ' ' << ep[e];
    }
    os << '\n';
}

} // namespace api

#endif // GRAPTOR_API_H
