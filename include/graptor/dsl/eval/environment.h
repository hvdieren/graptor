// -*- c++ -*-
#ifndef GRAPTOR_DSL_EVAL_ENVIRONMENT_H
#define GRAPTOR_DSL_EVAL_ENVIRONMENT_H

#include "graptor/dsl/eval/evaluator.h"

namespace expr {
namespace eval {

template<typename AddressMap>
class execution_environment {
public:
    using address_map_t = AddressMap;

public:
    execution_environment( address_map_t && amap )
	: m_aid_to_address( std::forward<address_map_t>( amap ) ) { }
    execution_environment( const address_map_t & amap )
	: m_aid_to_address( std::move( amap ) ) { }
    execution_environment() { }

    template<typename cache_map_type, typename value_map_type, typename Expr>
    __attribute__((always_inline))
    inline auto
    evaluate( cache_map_type &c, const value_map_type & m, const Expr & e ) const {
	sb::mask_pack<> mpack;
	return expr::evaluator<value_map_type, cache_map_type, address_map_t,
			       false>( c, m, m_aid_to_address )
	    .evaluate( e, mpack );
    }

    template<typename cache_map_type, typename value_map_type,
	     typename mask_pack_type, typename Expr>
    __attribute__((always_inline))
    inline auto
    evaluate( cache_map_type &c, const value_map_type & m,
	      const mask_pack_type & mpack, const Expr & e ) const {
	return expr::evaluator<value_map_type, cache_map_type, address_map_t,
			       false>( c, m, m_aid_to_address )
	    .evaluate( e, mpack );
    }


    template<bool Atomic, typename cache_map_type, typename value_map_type, typename Expr>
    __attribute__((always_inline))
    inline auto
    evaluate( cache_map_type &c, const value_map_type & m, const Expr & e ) const {
	sb::mask_pack<> mpack;
	return expr::evaluator<value_map_type, cache_map_type, address_map_t,
			       Atomic>( c, m, m_aid_to_address )
	    .evaluate( e, mpack );
    }

    template<typename cache_map_type, typename value_map_type, typename Expr>
    __attribute__((always_inline))
    inline bool
    evaluate_bool( cache_map_type & c, const value_map_type & m, const Expr & expr ) const {
	// Check all lanes are active.
	// Type-check on the expression to see if it is constant true.
	// If so, omit the check.
	if constexpr ( expr::is_constant_true<Expr>::value )
	    return true;
	else {
	    auto r = evaluate( c, m, expr );
	    return !expr::is_false( r );
	}
    }

    template<typename Expr>
    inline auto
    replace_ptrs( Expr && expr ) const {
	auto aid_to_address = extract_pointer_set( std::forward<Expr>( expr ) );
	auto aid_replaced = map_replace_all( m_aid_to_address, aid_to_address );
	using replaced_map_t = decltype(aid_replaced);
	// Return this object when map is unchanged to avoid generation of
	// code by the C++ compiler that copies the address map
	if constexpr ( std::is_same_v<std::decay_t<replaced_map_t>,
				      std::decay_t<address_map_t>> )
	    return *this;
	else
	    return execution_environment<replaced_map_t>( aid_replaced );
    }

    address_map_t & get_map() { return m_aid_to_address; }

private:
    address_map_t m_aid_to_address;
};

template<typename... Expr>
GG_INLINE inline
auto create_execution_environment( Expr &&... expr ) {
    auto aid_to_address = extract_pointer_set( std::forward<Expr>( expr )... );
    using address_map_t = decltype(aid_to_address);
    return execution_environment<address_map_t>( aid_to_address );
}
    
template<typename PtrSet, typename... Expr>
GG_INLINE inline
auto create_execution_environment_with(
    const PtrSet & ptrset, Expr &&... expr ) {
/*
    using address_map_t = decltype(
	extract_pointer_set_with( ptrset, std::forward<Expr>( expr )... ) );
    return execution_environment<address_map_t>(
	std::forward<address_map_t>(
	    extract_pointer_set_with( ptrset, std::forward<Expr>( expr )... )
	    ) );
*/
    using address_map_t
	= typename expr::ast_ptrset::ptrset_list<PtrSet,Expr...>::map_type;
    execution_environment<address_map_t> env;
    expr::map_copy_entries( env.get_map(), ptrset ); // copy supplied pointers
    expr::ast_ptrset::ptrset_list<PtrSet,Expr...>::initialize(
	env.get_map(), expr... );
    return env;
	
}

template<typename Operator, typename Cache, typename WeightTy>
GG_INLINE inline
auto create_execution_environment_op( Operator & op, Cache & c, WeightTy * w ) {
    using ew_pset
	= typename expr::ast_ptrset::ptrset_pointer<
	    expr::aid_eweight, WeightTy, expr::map_new<>>::map_type;
    using op_pset
	= typename Operator::template ptrset<ew_pset>::map_type;
    using address_map_t
	= typename expr::ast_ptrset::ptrset_list<op_pset,Cache>::map_type;

    execution_environment<address_map_t> env;

    Operator::template ptrset<ew_pset>::initialize( env.get_map(), op );
    expr::ast_ptrset::ptrset_list<op_pset,Cache>::initialize(
	env.get_map(), c );
    expr::ast_ptrset::ptrset_pointer<
	expr::aid_eweight, WeightTy, expr::map_new<>>::initialize(
	    env.get_map(), w );

    return env;
}

template<typename Operator, typename Cache>
GG_INLINE inline
auto create_execution_environment_op( Operator & op, Cache & c ) {
    return create_execution_environment_op( op, c, (VID *)nullptr );
}
    
} // namespace eval
} // namespace expr

#endif // GRAPTOR_DSL_EVAL_ENVIRONMENT_H
