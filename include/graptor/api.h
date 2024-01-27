// -*- C++ -*-
#ifndef GRAPTOR_API_H
#define GRAPTOR_API_H

#include "graptor/dsl/ast/decl.h"
#include "graptor/api/utils.h"
#include "graptor/api/vertexprop.h"
#include "graptor/api/edgeprop.h"

class frontier;

namespace api {

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
	return float( EID(nactv) + nacte )
	    <= float( m ) / ( 100.0 / (float)m_threshold );
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
    constexpr bool is_sparse( VID nactv, EID nacte, EID m ) const {
	return true;
    }

    constexpr bool is_sparse( frontier F, EID m ) const {
	return true;
    }
};

struct always_dense_t {
    constexpr bool is_sparse( VID nactv, EID nacte, EID m ) const {
	return false;
    }

    constexpr bool is_sparse( frontier F, EID m ) const {
	return false;
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

/************************************************************************
 * Definition of fusion threshold
 ************************************************************************/
class fusion_select {
    friend std::ostream & operator << ( std::ostream & os, fusion_select t );

public:
    fusion_select( bool enabled ) : m_enabled( enabled ) { }

    constexpr bool do_fusion( VID nactv, EID nacte, EID m ) const {
	return m_enabled;
    }

    constexpr bool do_fusion( frontier F, EID m ) const {
	return do_fusion( F.nActiveVertices(), F.nActiveEdges(), m );
    }

private:
    bool m_enabled;
};

struct default_fusion_select {
    static constexpr bool do_fusion( VID nactv, EID nacte, EID m ) {
	return nacte >= 4096;
    }

    static constexpr bool do_fusion( frontier F, EID m ) {
	return do_fusion( F.nActiveVertices(), F.nActiveEdges(), m );
    }
};

struct always_fusion_t {
    constexpr bool do_fusion( VID nactv, EID nacte, EID m ) const {
	return true;
    }

    constexpr bool do_fusion( frontier F, EID m ) const {
	return true;
    }
};

std::ostream & operator << ( std::ostream & os, fusion_select t ) {
    return os << "fusion(" << t.m_enabled << ")";
}

std::ostream & operator << ( std::ostream & os, default_fusion_select t ) {
    return os << "fusion(default)";
}

std::ostream & operator << ( std::ostream & os, always_fusion_t t ) {
    return os << "fusion(always)";
}

static constexpr auto always_fusion = always_fusion_t();

template<typename T>
struct is_fusion_select : public std::false_type { };

template<>
struct is_fusion_select<default_fusion_select>
    : public std::true_type { };

template<>
struct is_fusion_select<fusion_select>
    : public std::true_type { };

template<>
struct is_fusion_select<always_fusion_t>
    : public std::true_type { };

template<typename T>
constexpr bool is_fusion_select_v = is_fusion_select<T>::value;


/**=====================================================================*
 * Definition of filters
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
    using encoding = typename frontier_params<ftype,EMapConfig::VL>::encoding;
    using array_ty =
	typename expr::array_select<ftype,StoreTy,VID,expr::aid_frontier_old_f,
				    encoding>::type;
    
    static constexpr bool is_scan = Operator::is_scan;
    static constexpr bool defines_frontier = false;

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

    template<typename map_type0>
    struct ptrset {
	using map_type1 = typename Operator::ptrset<map_type0>::map_type;
	using map_type =
	    typename expr::ast_ptrset::ptrset_pointer<
	    expr::aid_frontier_old_f,
	    typename frontier_params<ftype,EMapConfig::VL>::type,
	    map_type1
	    >::map_type;

	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_eweight)),map_type0>, "check 0" );
	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_eweight)),map_type1>, "check 1" );
	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_eweight)),map_type1>, "check 2" );

	template<typename MapTy>
	static void initialize( MapTy & map, const arg_filter_op & op ) {
	    Operator::template ptrset<map_type0>::initialize( map, op.m_op );
	    expr::ast_ptrset::ptrset_pointer<
		expr::aid_frontier_old_f,
		typename frontier_params<ftype,EMapConfig::VL>::type,
		map_type1
		>::initialize( map, op.m_frontier.getDense<ftype>() );
	}
    };

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

    frontier & get_frontier() const { return m_frontier; }

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
    using encoding = typename frontier_params<ftype,EMapConfig::VL>::encoding;
    using array_ty =
	typename expr::array_select<ftype,StoreTy,VID,expr::aid_frontier_old_f,
				    encoding>::type;
    using ptrset_ty = PtrSet;
    
    static constexpr bool is_scan = Operator::is_scan;
    static constexpr bool defines_frontier = Operator::defines_frontier;

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
		// m_method( d );
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

    template<typename map_type0>
    struct ptrset {
	using map_type1 = typename Operator::ptrset<map_type0>::map_type;
	// using map_type =
	// typename expr::ast_ptrset::merge_maps<map_type1, ptrset_ty>::type;
	using mexpr_type = decltype( ((std::decay_t<Fn>*)nullptr)->operator()( expr::value<simd::ty<VID,1>,expr::vk_vid>() ) );
	using map_type =
	    typename expr::ast_ptrset::ptrset_list<map_type1,mexpr_type>::map_type;

	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_eweight)),map_type0>, "check 0" );
	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_eweight)),map_type1>, "check 1" );
	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_eweight)),map_type>, "check 2" );

	template<typename MapTy>
	static void initialize( MapTy & map, const arg_filter_a_op & op ) {
	    auto v = expr::value<simd::ty<VID,1>,expr::vk_vid>();

	    Operator::template ptrset<map_type0>::initialize( map, op.m_op );
	    expr::ast_ptrset::ptrset_list<map_type1,mexpr_type>
		::initialize( map, op.m_method( v ) );
	    // expr::map_copy_entries( map, op.m_ptrset );
	}
    };

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
    frontier_true = 0,
    frontier_reduction = 1,
    frontier_method = 2,
    frontier_reduction_or_method = 3
};

template<frontier_record R>
struct frontier_record_argument {
    static constexpr frontier_record value = R;
};

static constexpr auto all_true =
    frontier_record_argument<frontier_record::frontier_true>();
static constexpr auto reduction =
    frontier_record_argument<frontier_record::frontier_reduction>();
static constexpr auto method =
    frontier_record_argument<frontier_record::frontier_method>();
static constexpr auto reduction_or_method =
    frontier_record_argument<frontier_record::frontier_reduction_or_method>();

constexpr auto operator | ( frontier_record l, frontier_record r ) {
    return frontier_record( (int)l | (int)r );
}

template<frontier_record L, frontier_record R>
constexpr auto operator | ( frontier_record_argument<L> &&,
			    frontier_record_argument<R> && ) {
    return frontier_record_argument<L | R>();
}

template<typename T>
struct is_frontier_record : public std::false_type { };

template<frontier_record R>
struct is_frontier_record<frontier_record_argument<R>>
    : public std::true_type { };

template<typename T>
constexpr bool is_frontier_record_v = is_frontier_record<T>::value;

/************************************************************************
 * Definition of filter recording method
 ************************************************************************/
template<typename EMapConfig, bool IsPriv,
	 typename Operator, typename Enable = void>
struct arg_record_reduction_op {
    static constexpr frontier_type ftype = EMapConfig::wr_ftype;
    static constexpr bool is_priv = IsPriv;
    using StoreTy = typename frontier_params<ftype,EMapConfig::VL>::type;
    using encoding = typename frontier_params<ftype,EMapConfig::VL>::encoding;
    using array_ty =
	typename expr::array_select<ftype,StoreTy,VID,expr::aid_frontier_new_f,
				    encoding>::type;
    
    static constexpr bool is_scan = true;
    static constexpr bool defines_frontier = true;

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
/* -- this version does not work correctly for reduction_or_method with
 * -- embedded frontier
	return expr::set_mask(
	    expr::get_mask_cond( d ),
	    expr::set_mask(
		m_op.relax( s, dd, e ),
		m_array[dd]
		= expr::value<typename VIDDst::data_type::prefmask_traits,expr::vk_true>() )
	    );
*/
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
	    expr::add_predicate( expr::cast<EID>( degree[vid] ), mask ),
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

    template<typename map_type0>
    struct ptrset {
	using map_type1 =
	    typename Operator::ptrset<map_type0>::map_type;
	using map_type2 =
	    typename expr::ast_ptrset::ptrset_pointer<
	    expr::aid_frontier_new,
	    typename frontier_params<ftype,EMapConfig::VL>::type,
	    map_type1
	    >::map_type;
	using map_type3 =
	    typename expr::ast_ptrset::ptrset_pointer<
	    expr::aid_frontier_nacte, EID, map_type2>::map_type;
	using map_type4 =
	    typename expr::ast_ptrset::ptrset_pointer<
	    expr::aid_frontier_nactv, VID, map_type3>::map_type;
	using map_type =
	    typename expr::ast_ptrset::ptrset_pointer<
	    expr::aid_graph_degree, VID, map_type4>::map_type;
    
	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_eweight)),map_type1>, "check 1" );
	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_frontier_new)),map_type2>, "check 2" );
	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_frontier_nacte)),map_type3>, "check 3" );
	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_frontier_nactv)),map_type4>, "check 4" );
	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_graph_degree)),map_type>, "check 5" );

	template<typename MapTy>
	static void initialize( MapTy & map,
				const arg_record_reduction_op & op ) {
	    Operator::template ptrset<map_type0>::initialize( map, op.m_op );
	    expr::ast_ptrset::ptrset_pointer<
		expr::aid_frontier_new,
		typename frontier_params<ftype,EMapConfig::VL>::type,
		map_type1
		>::initialize( map, op.m_frontier.getDense<ftype>() );
	    expr::ast_ptrset::ptrset_pointer<
		expr::aid_frontier_nacte, EID, map_type2
		>::initialize( map, op.m_frontier.nActiveEdgesPtr() );
	    expr::ast_ptrset::ptrset_pointer<
		expr::aid_frontier_nactv, VID, map_type3
		>::initialize( map, op.m_frontier.nActiveVerticesPtr() );
	    expr::ast_ptrset::ptrset_pointer<
		expr::aid_graph_degree, VID, map_type4
		>::initialize( map, op.m_degree );
	}
    };

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
    static constexpr bool defines_frontier = true;

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
	    expr::add_predicate( expr::cast<EID>( degree[vid] ), mask ),
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

    template<typename map_type0>
    struct ptrset {
	using map_type1 =
	    typename Operator::ptrset<map_type0>::map_type;
	using map_type2 =
	    typename expr::ast_ptrset::ptrset_pointer<
	    expr::aid_frontier_nacte, EID, map_type1>::map_type;
	using map_type3 =
	    typename expr::ast_ptrset::ptrset_pointer<
	    expr::aid_frontier_nactv, EID, map_type2>::map_type;
	using map_type =
	    typename expr::ast_ptrset::ptrset_pointer<
	    expr::aid_graph_degree, EID, map_type3>::map_type;

	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_eweight)),map_type0>, "check 0" );
	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_eweight)),map_type1>, "check 1" );
	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_eweight)),map_type2>, "check 2" );
	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_eweight)),map_type3>, "check 3" );
	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_eweight)),map_type>, "check 4" );
    
	template<typename MapTy>
	static void initialize( MapTy & map,
				const arg_record_reduction_op & op ) {
	    Operator::template ptrset<map_type0>::initialize( map, op.m_op );
	    expr::ast_ptrset::ptrset_pointer<
		expr::aid_frontier_nacte, EID, map_type1
		>::initialize( map, op.m_frontier.nActiveEdgesPtr() );
	    expr::ast_ptrset::ptrset_pointer<
		expr::aid_frontier_nactv, EID, map_type2
		>::initialize( map, op.m_frontier.nActiveVerticesPtr() );
	    expr::ast_ptrset::ptrset_pointer<
		expr::aid_graph_degree, EID, map_type3
		>::initialize( map, op.m_degree );
	}
    };


    const auto get_config() const { return m_op.get_config(); }

private:
    VID * m_degree;
    Operator m_op;
    frontier & m_frontier;
};


template<frontier_type ftype, unsigned short VL>
struct record_store_type {
    using type = typename frontier_params<ftype,VL>::type;
    using encoding = typename frontier_params<ftype,VL>::encoding;
};

template<unsigned short VL>
struct record_store_type<frontier_type::ft_unbacked,VL> {
    using type = void;
    using encoding = array_encoding<void>;
};

template<typename EMapConfig, bool IsPriv, typename Operator,
	 typename Fn, typename PtrSet>
struct arg_record_method_op {
    using self_type =
	arg_record_method_op<EMapConfig,IsPriv,Operator,Fn,PtrSet>;
    static constexpr frontier_type ftype = EMapConfig::wr_ftype;
    static constexpr bool is_priv = IsPriv;
    using StoreTy = typename record_store_type<ftype,EMapConfig::VL>::type;
    using encoding = typename record_store_type<ftype,EMapConfig::VL>::encoding;
    using array_ty =
	typename expr::array_select<ftype,StoreTy,VID,expr::aid_frontier_new_f,
				    encoding>::type;
    using ptrset_ty = PtrSet;
    
    static constexpr bool is_scan = true;
    static constexpr bool defines_frontier = true;

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
		expr::add_predicate( expr::cast<EID>( degree[vid] ), mask ),
		nactv[expr::zero_val(vid)] +=
		expr::add_predicate( expr::constant_val_one(vid), mask ) );
	} else if constexpr ( is_priv ) {
	    // TODO: consider use of let to hold mask
	    // ... experimental code below
#if 0
	    auto mask = expr::rewrite_internal( m_method( vid ) );
	    return expr::make_seq(
		m_op.vertexop( vid ),
		m_array[vid] = expr::make_unop_switch_to_vector( mask ),
		nacte[expr::zero_val(vid)] +=
		expr::add_predicate( expr::cast<EID>( degree[vid] ), mask ),
		nactv[expr::zero_val(vid)] +=
		expr::add_predicate( expr::constant_val_one(vid), mask ) );
#else
	    auto mm = m_method( vid );
	    using MTr = typename std::decay_t<decltype(mm)>::data_type;
	    return expr::let<expr::aid_let_record>(
		expr::rewrite_internal(
		    expr::make_unop_switch_to_vector( mm ) ),
		[&]( auto mask ) {
		    return expr::make_seq(
			m_op.vertexop( vid ),
			m_array[vid] = mask, // expr::make_unop_switch_to_vector( mask ),
			nacte[expr::zero_val(vid)] +=
			expr::add_predicate( expr::cast<EID>( degree[vid] ),
					     expr::make_unop_cvt_to_mask<MTr>( mask ) ),
			nactv[expr::zero_val(vid)] +=
			expr::add_predicate( expr::constant_val_one(vid),
					     expr::make_unop_cvt_to_mask<MTr>( mask ) ) );
		} );
		
#endif
	} else {
	    // TODO: consider use of let to hold mask
	    auto mask = expr::rewrite_internal( m_method( vid ) );
	    return expr::make_seq(
		m_op.vertexop( vid ),
		m_array[vid] |= expr::make_unop_switch_to_vector( mask ),
		nacte[expr::zero_val(vid)] +=
		expr::add_predicate( expr::cast<EID>( degree[vid] ), mask ),
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

    template<typename map_type0>
    struct ptrset {
	using method_expr = decltype(
	    static_cast<self_type*>( nullptr )->
	    m_method( expr::value<simd::ty<VID,1>,expr::vk_dst>() ) );
	using map_type1 =
	    typename Operator::ptrset<map_type0>::map_type;
	using map_type2 =
	    typename expr::ast_ptrset::ptrset_pointer<
	    expr::aid_frontier_nacte, EID, map_type1>::map_type;
	using map_type3 =
	    typename expr::ast_ptrset::ptrset_pointer<
	    expr::aid_frontier_nactv, VID, map_type2>::map_type;
	using map_type4 =
	    typename expr::ast_ptrset::ptrset_pointer<
	    expr::aid_graph_degree, VID, map_type3>::map_type;
	using map_type5 =
	    std::conditional_t<
	    ftype == frontier_type::ft_unbacked,
	    map_type4,
	    typename expr::ast_ptrset::ptrset_pointer<
		expr::aid_frontier_new,
		typename frontier_params<ftype,EMapConfig::VL>::type,
		map_type4
		>::map_type>;
	using map_type = typename expr::ast_ptrset::ptrset_list<
	    map_type5,method_expr>::map_type;
	    
	template<typename MapTy>
	static void initialize( MapTy & map,
				const arg_record_method_op & op ) {
	    Operator::template ptrset<map_type0>::initialize( map, op.m_op );
	    expr::ast_ptrset::ptrset_pointer<
		expr::aid_frontier_nacte, EID, map_type1
		>::initialize( map, op.m_frontier.nActiveEdgesPtr() );
	    expr::ast_ptrset::ptrset_pointer<
		expr::aid_frontier_nactv, VID, map_type2
		>::initialize( map, op.m_frontier.nActiveVerticesPtr() );
	    expr::ast_ptrset::ptrset_pointer<
		expr::aid_graph_degree, VID, map_type3
		>::initialize( map, op.m_degree );
	    if constexpr ( ftype != frontier_type::ft_unbacked )
		expr::ast_ptrset::ptrset_pointer<
		    expr::aid_frontier_new,
		    typename frontier_params<ftype,EMapConfig::VL>::type,
		    map_type4
		    >::initialize( map, op.m_frontier.getDense<ftype>() );

	    auto d = expr::value<simd::ty<VID,1>,expr::vk_dst>();
	    expr::ast_ptrset::ptrset_list<map_type5,method_expr>
		::initialize( map, op.m_method( d ) );
	}
    };

    const auto get_config() const { return m_op.get_config(); }

    frontier & get_frontier() const { return m_frontier; }

private:
    VID * m_degree;
    Operator m_op;
    array_ty m_array;
    frontier & m_frontier;
    Fn m_method;
    ptrset_ty m_ptrset;
};

template<filter_strength S, frontier_record R, typename MF = void>
struct arg_record {
    static constexpr filter_strength strength = S; // weak implies unbacked ok
    static constexpr frontier_record record = R;
    static constexpr bool may_be_unbacked = strength == filter_strength::weak;
    static constexpr bool may_merge_frontier = !std::is_same_v<MF,void>;

    using merge_frontier_type = MF;

    static_assert( record != frontier_record::frontier_method, "required" );

    arg_record( frontier & f ) : m_frontier( f ) { }

    arg_record( frontier & f, merge_frontier_type * merge_vp )
	: m_frontier( f ), m_merge_vp( merge_vp ) {
	static_assert( !std::is_same_v<merge_frontier_type,void>,
		       "if supplied, must be non-void pointer" );
    }

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
	if constexpr ( ftype == frontier_type::ft_msb4 )
	    m_frontier = frontier::msb( part, m_merge_vp );
	else
	    m_frontier = frontier::create<ftype>( part );
    }

    template<typename GraphType>
    frontier & get_frontier( const GraphType & G ) { return m_frontier; }

    frontier & get_frontier() const { return m_frontier; }

private:
    frontier & m_frontier;
    merge_frontier_type * m_merge_vp;
};

// Strong/weak here implies unbacked or not? Better to replace with other terms
// to make it more clear
template<filter_strength S, frontier_record M, typename Fn, typename MF = void>
struct arg_record_m {
    static constexpr filter_strength strength = S;
    static constexpr frontier_record record = M;
    static constexpr bool may_be_unbacked = strength == filter_strength::weak;
    static constexpr bool may_merge_frontier = !std::is_same_v<MF,void>;

    using merge_frontier_type = MF;

    static_assert( record == frontier_record::frontier_method
		   || record == frontier_record::frontier_reduction_or_method,
		   "this class is designed for frontier using method" );

    arg_record_m( frontier & f, Fn method )
	: m_frontier( f ), m_method( method ) { }

    arg_record_m( frontier & f, merge_frontier_type * merge_vp, Fn method )
	: m_frontier( f ), m_merge_vp( merge_vp ), m_method( method ) {
	static_assert( !std::is_same_v<merge_frontier_type,void>,
		       "if supplied, must be non-void pointer" );
    }

    template<typename EMapConfig, bool IsPriv, typename Operator>
    auto record_and_count( const VID * degree, Operator op ) {
	// method: use method
	// reduction_or_method: use method if push (not privatized destinations)
	if constexpr ( record == frontier_record::frontier_method || !IsPriv ) {
	    auto d = expr::value<simd::ty<VID,1>,expr::vk_dst>();
	    auto ptrset = expr::extract_pointer_set( m_method( d ) );
	    return arg_record_method_op<
		EMapConfig,IsPriv,Operator,Fn,decltype(ptrset)>(
		    op, m_frontier, degree, m_method, ptrset );
	} else {
	    // As per arg_record / reduction
	    return arg_record_reduction_op<EMapConfig,IsPriv,Operator>(
		op, m_frontier, degree );
	}
    }

    template<frontier_type ftype>
    void setup( const partitioner & part ) {
	if constexpr ( ftype == frontier_type::ft_msb4 )
	    m_frontier = frontier::msb( part, m_merge_vp );
	else
	    m_frontier = frontier::create<ftype>( part );
    }

    template<typename GraphType>
    frontier & get_frontier( GraphType & G ) { return m_frontier; }

    frontier & get_frontier() const { return m_frontier; }

private:
    frontier & m_frontier;
    merge_frontier_type * m_merge_vp;
    Fn m_method;
};

struct missing_record_argument {
    static constexpr filter_strength strength = filter_strength::weak;
    static constexpr frontier_record record = frontier_record::frontier_true;
    static constexpr bool may_be_unbacked = true;
    static constexpr bool may_merge_frontier = false;

    template<typename EMapConfig, bool IsPriv, typename Operator>
    auto record_and_count( const VID * degree, Operator op ) {
	return op;
    }
    template<frontier_type>
    void setup( const partitioner & part ) { }
};

template<frontier_record Record = frontier_record::frontier_reduction,
	 filter_strength Strength = filter_strength::strong>
auto
record( frontier & f,
	frontier_record_argument<Record> r = reduction,
	filter_strength_argument<Strength> s = strong ) {
    return arg_record<Strength,Record>( f );
}
    
template<typename Fn, filter_strength Strength = filter_strength::strong>
std::enable_if_t<is_active<Fn>::value,
		 arg_record_m<Strength,frontier_record::frontier_method,Fn>>
record( frontier & f,
	Fn && fn,
	filter_strength_argument<Strength> s = strong ) {
    return arg_record_m<Strength,frontier_record::frontier_method,Fn>(
	f, std::forward<Fn>( fn ) );
}
	     
template<frontier_record Record,
	 typename Fn,
	 filter_strength Strength = filter_strength::strong>
std::enable_if_t<
    is_active<Fn>::value,
    arg_record_m<Strength,Record,Fn>>
record( frontier & f,
	frontier_record_argument<Record> r,
	Fn && fn,
	filter_strength_argument<Strength> s = strong ) {
    static_assert( Record == frontier_record::frontier_method
		   || Record == frontier_record::frontier_reduction_or_method,
		   "Require a method option in this interface variant" );
    return arg_record_m<Strength,Record,Fn>(
	f, std::forward<Fn>( fn ) );
}

template<typename MF,
	 frontier_record Record = frontier_record::frontier_reduction,
	 filter_strength Strength = filter_strength::strong>
std::enable_if_t<!std::is_same_v<MF,void>,
		 arg_record<Strength,Record,MF>>
record( frontier & f,
	MF * merge_vp,
	frontier_record_argument<Record> r = reduction,
	filter_strength_argument<Strength> s = strong ) {
    return arg_record<Strength,Record,MF>( f, merge_vp );
}
   

template<frontier_record Record,
	 typename MF,
	 typename Fn,
	 filter_strength Strength = filter_strength::strong>
std::enable_if_t<
    is_active<Fn>::value && !std::is_same_v<MF,void>,
    arg_record_m<Strength,Record,Fn,MF>>
record( frontier & f,
	MF * merge_vp,
	frontier_record_argument<Record> r,
	Fn && fn,
	filter_strength_argument<Strength> s = strong ) {
    static_assert( Record == frontier_record::frontier_method
		   || Record == frontier_record::frontier_reduction_or_method,
		   "Require a method option in this interface variant" );
    return arg_record_m<Strength,Record,Fn,MF>(
	f, merge_vp, std::forward<Fn>( fn ) );
}
	     
template<typename T>
struct is_record : public std::false_type { };

template<filter_strength S, frontier_record R, typename MF>
struct is_record<arg_record<S,R,MF>> : public std::true_type { };

template<filter_strength S, frontier_record R, typename Fn, typename MF>
struct is_record<arg_record_m<S,R,Fn,MF>> : public std::true_type { };

/************************************************************************
 * Defintion of configuration option
 ************************************************************************/
template<parallelism_spec P, unsigned short VL, typename threshold_type,
	 typename fusion_type>
struct arg_config {
    static constexpr parallelism_spec parallelism = P;
    static constexpr vectorization_spec<VL> vectorization =
	vectorization_spec<VL>();
	
    arg_config( const threshold_type & t, const fusion_type & f )
	: m_threshold( t ), m_fusion( f ) { }

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
    constexpr fusion_type get_fusion() const { return m_fusion; }
    constexpr fusion_flags get_fusion_flags() const {
	return m_fusion.get_flags();
    }
    constexpr bool do_fusion( VID nactv, EID nacte, EID m ) const {
	return m_fusion.do_fusion( nactv, nacte, m );
    }
    constexpr bool do_fusion( frontier F, EID m ) const {
	return do_fusion( F.nActiveVertices(), F.nActiveEdges(), m );
    }
    
private:
    threshold_type m_threshold;
    fusion_type m_fusion;
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
    constexpr default_fusion_select get_fusion() const {
	return default_fusion_select();
    }
    constexpr bool do_fusion( VID nactv, EID nacte, EID m ) const {
	return default_fusion_select::do_fusion( nactv, nacte, m );
    }
    constexpr bool do_fusion( frontier F, EID m ) const {
	return do_fusion( F.nActiveVertices(), F.nActiveEdges(), m );
    }
};

template<typename... Args>
auto config( Args &&... args ) {
    if constexpr( check_arguments_4<is_parallelism,
		  is_vectorization,is_threshold,is_fusion_select,
		  Args...>::value ) {
	auto & t = get_argument_value<is_threshold,default_threshold>(
	    args... );
	auto & f = get_argument_value<is_fusion_select,default_fusion_select>(
	    args... );

	return arg_config<
	    get_argument_type_t<is_parallelism,decltype(parallel),Args...>::value,
	    get_argument_type_t<is_vectorization,decltype(vl_max<MAX_VL>()),Args...>::value,
	    get_argument_type_t<is_threshold,default_threshold,Args...>,
	    get_argument_type_t<is_fusion_select,default_fusion_select,Args...>>( t, f );
    } else
	return 0; // static_assert( false, "unexpected argument(s) supplied" );
}

template<typename T>
struct is_config : public std::false_type { };

template<parallelism_spec P, unsigned short VL, typename threshold_type,
	 typename fusion_type>
struct is_config<arg_config<P,VL,threshold_type,fusion_type>>
    : public std::true_type { };

/************************************************************************
 * Definition of relax method
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
 * Definition of fusion method
 ************************************************************************/
#include "graptor/api/fusion.h"

/************************************************************************
 * Construct conventional operator class
 ************************************************************************/
template<typename Rlx,
	 typename Fsrc,
	 typename Fdst,
	 typename Adst,
	 typename Rec,
	 typename Fus,
	 typename Cfg>
struct op_def {
    static constexpr frontier_mode new_frontier =
	std::is_same_v<Rec,missing_record_argument> ? fm_all_true
	: std::decay_t<Rec>::record == frontier_record::frontier_reduction ? fm_reduction
	: fm_calculate;

    static constexpr bool defines_frontier =
	!std::is_same_v<Rec,missing_record_argument>;

    static constexpr bool is_scan = defines_frontier;

    // Weak frontiers on source of edge may be omitted
    static constexpr bool may_omit_frontier_rd =
	std::is_same_v<Fsrc,missing_filter_argument>
	|| Fsrc::strength == filter_strength::weak;

    // Can only omit writing a frontier if user did not ask for it.
    static constexpr bool may_omit_frontier_wr =
	std::is_same_v<Rec,missing_record_argument>;

    static constexpr bool new_frontier_dense = false; // to be removed

    op_def( Rlx && relax, Fsrc && filter_src, Fdst && filter_dst,
	    Adst && active_dst, Rec && record, Fus && fusion, Cfg && config )
	: m_relax( std::forward<Rlx>( relax ) ),
	  m_filter_src( std::forward<Fsrc>( filter_src ) ),
	  m_filter_dst( std::forward<Fdst>( filter_dst ) ),
	  m_active_dst( std::forward<Adst>( active_dst ) ),
	  m_record( std::forward<Rec>( record ) ),
	  m_fusion( std::forward<Fus>( fusion ) ),
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
	// set_mask sets lhs mask on rhs before evaluating rhs
	return expr::set_mask( m, m_relax.m_method( ss, dd, ee ) );
    }

    template<typename VIDSrc, typename VIDDst, typename EIDEdge>
    auto fusionop( VIDSrc s, VIDDst d, EIDEdge e ) const {
	// This is primarily used for determing VL and frontier byte widths.
	// Frontiers will be added in later.
	auto ss = expr::remove_mask( s );
	auto dd = expr::remove_mask( d );
	auto ee = expr::remove_mask( e );
	auto m = expr::get_mask_cond( s )
	    && expr::get_mask_cond( d )
	    && expr::get_mask_cond( e );
	// set_mask sets lhs mask on rhs before evaluating rhs
	return expr::set_mask( m, m_fusion.fusionop( ss, dd, ee ) );
    }

    constexpr fusion_flags get_fusion_flags() const {
	return m_fusion.get_flags();
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
#if 0
	return map_merge(
	    map_merge(
		expr::extract_pointer_set_with( pset, relax( s, d, e ) ),
		expr::extract_pointer_set( vertexop( d ) ) ),
	    expr::extract_pointer_set( fusionop( s, d, e ) ) );
#else
	return expr::extract_pointer_set_with(
	    pset, relax( s, d, e ), vertexop( d ), fusionop( s, d, e ) );
#endif
    }

    auto get_ptrset() const { // TODO - redundant
	return get_ptrset( expr::create_map() );
    }

    template<typename PSet>
    struct ptrset {
	using relax_expr = decltype(
	    static_cast<op_def*>( nullptr )->
	    relax( expr::value<simd::ty<VID,1>,expr::vk_src>(),
		   expr::value<simd::ty<VID,1>,expr::vk_dst>(),
		   expr::value<simd::ty<EID,1>,expr::vk_edge>() ) );
	using vertexop_expr = decltype(
	    static_cast<op_def*>( nullptr )->
	    vertexop( expr::value<simd::ty<VID,1>,expr::vk_dst>() ) );
	using fusionop_expr = decltype(
	    static_cast<op_def*>( nullptr )->
	    fusionop( expr::value<simd::ty<VID,1>,expr::vk_src>(),
		      expr::value<simd::ty<VID,1>,expr::vk_dst>(),
		      expr::value<simd::ty<EID,1>,expr::vk_edge>() ) );
					
	using map_type = typename expr::ast_ptrset::ptrset_list<
	    PSet,relax_expr,vertexop_expr,fusionop_expr>::map_type;

	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_eweight)),PSet>, "check 0" );
	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_eweight)),map_type>, "check 1" );

	template<typename MapTy>
	static void initialize( MapTy & map, const op_def & op ) {
	    auto s = expr::value<simd::ty<VID,1>,expr::vk_src>();
	    auto d = expr::value<simd::ty<VID,1>,expr::vk_dst>();
	    auto e = expr::value<simd::ty<EID,1>,expr::vk_edge>();

	    expr::ast_ptrset::ptrset_list<
		PSet,relax_expr,vertexop_expr,fusionop_expr>
		::initialize( map, op.relax( s, d, e ), op.vertexop( d ),
			      op.fusionop( s, d, e ) );
	}
    };

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
    frontier & get_frontier() const { return m_record.get_frontier(); }

private:
    Rlx m_relax;
    Fsrc m_filter_src;
    Fdst m_filter_dst;
    Adst m_active_dst;
    Rec m_record;
    Fus m_fusion;
    Cfg m_config;
};

template<typename Rlx,
	 typename Fsrc,
	 typename Fdst,
	 typename Adst,
	 typename Rec,
	 typename Fus,
	 typename Cfg>
auto op_create( Rlx && rlx, Fsrc && fsrc, Fdst && fdst, Adst && adst,
		Rec && rec, Fus && fus, Cfg && cfg ) {
    return op_def<std::decay_t<Rlx>,
		  std::decay_t<Fsrc>,
		  std::decay_t<Fdst>,
		  std::decay_t<Adst>,
		  std::decay_t<Rec>,
		  std::decay_t<Fus>,
		  std::decay_t<Cfg>>(
		      std::forward<std::decay_t<Rlx>>( rlx ),
		      std::forward<std::decay_t<Fsrc>>( fsrc ),
		      std::forward<std::decay_t<Fdst>>( fdst ),
		      std::forward<std::decay_t<Adst>>( adst ),
		      std::forward<std::decay_t<Rec>>( rec ),
		      std::forward<std::decay_t<Fus>>( fus ),
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
    auto fusion =
	get_argument_value<is_fusion,missing_fusion_argument>( args... );

    auto config
	= get_argument_value<is_config,missing_config_argument>( args... );

    static_assert( std::is_same_v<std::decay_t<decltype(active_dst)>,
		   missing_filter_argument>
		   || decltype(active_dst)::entity == filter_entity::dst,
		   "filter method only applicable to destination" );

    // Build operator
    auto op = op_create( relax, filter_src, filter_dst, active_dst, record,
			 fusion, config );
    using Operator = decltype(op);

    // Analyse operator
    /*
    constexpr bool rd_frontier =
		  !std::is_same_v<decltype(filter_src),missing_filter_argument>;
    constexpr bool wr_frontier =
		  !std::is_same_v<decltype(filter_dst),missing_filter_argument>
		  && !std::is_same_v<decltype(active_dst),missing_filter_argument>;
    */

    // TODO: take into account how filtering is done. E.g., in destination
    //       filtering in pull, the read frontier is sequential access like
    //       the written frontier
    using cfg = expr::determine_emap_config<
	VID,Operator,GraphType,decltype(record)::may_be_unbacked,
	decltype(record)::may_merge_frontier
	>;
    using cfg_pull = typename cfg::pull;
    using cfg_push = typename cfg::push;
    using cfg_ireg = typename cfg::ireg;
    using cfg_scalar = typename cfg::scalar;

    // Track if through all of this, the edgemap operation has been applied
    // in some manner
    bool applied_emap = false;
    bool need_record = false;

    // If always dense required, then skip all sparse implementations
    if constexpr ( !std::is_same_v<decltype(config.get_threshold()),always_dense_t> ) {
	
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
		
		// Do sparse edgemap and construct new frontier.
		// The current edgemap implementations (at least CSR) use
		// the record method, or reduction.
		if constexpr ( record.record == frontier_record::frontier_true )
		    G = csr_sparse_no_f(
			config,
			GA.getCSR(), GA.get_eid_retriever(), F, sparse_op );
		else if constexpr (
		    record.record == frontier_record::frontier_reduction
		    || record.record ==
		    frontier_record::frontier_reduction_or_method )
		    G = csr_sparse_with_f(
			config,
			GA.getCSR(), GA.get_eid_retriever(),
			GA.get_partitioner(), F, sparse_op );
		else if constexpr( record.record == frontier_record::frontier_method ) {
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
		check_strength<cfg_scalar,graph_traversal_kind::gt_pull>(
		    active_dst.template
		    check_strength<cfg_scalar,graph_traversal_kind::gt_pull>(
			op ) );

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
		else if constexpr (
		    record.record == frontier_record::frontier_method
		    || record.record ==
		    frontier_record::frontier_reduction_or_method ) {
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

    }

    if constexpr ( std::is_same_v<decltype(config.get_threshold()),always_sparse_t> ) {
	assert( applied_emap && "if always-sparse, should have done so" );
	return make_lazy_executor( part ); // nothing more to do
    } else {
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
}

} // namespace api

#endif // GRAPTOR_API_H
