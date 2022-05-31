// -*- c++ -*-
#ifndef GRAPTOR_DSL_AST_CONSTANT_H
#define GRAPTOR_DSL_AST_CONSTANT_H

#include "graptor/dsl/ast/value.h"

namespace expr {

/**
 * Checks whether a value_kind represents a known constant
 * \param vkind: the value_kind that is checked
 * \return: true if the value is a known constant
 */
constexpr bool is_constant_kind( value_kind vkind ) {
    return vkind == vk_zero || vkind == vk_cstone
	|| vkind == vk_true || vkind == vk_false
	|| vkind == vk_truemask || vkind == vk_inc;
}

/**
 * constant: a constant, without vector length specification
 *
 * \tparam VKind: what constant is represented
 * \tparam T: type of the constant, or void if typeless
 */
template<value_kind VKind, typename T>
struct constant;

/**
 * constant: overload for a typed value that is constant throughout the
 *           expression.
 *
 * \tparam VKind: what constant is represented
 * \tparam T: type of the constant
 */
template<typename T>
struct constant<vk_any,T> : public expr_base {
    static constexpr value_kind vkind = vk_any;
    static constexpr op_codes opcode = op_constant;
    using type = T; // We do know what type this might be

    constexpr GG_INLINE constant( T val ) : m_val( val ) { }

    template<typename Tr>
    auto expand() {
	return value<Tr, vk_any>( m_val );
    }

    T get_value() const { return m_val; }

private:
    T m_val;
};

/**
 * constant: overload for a type-less, dimensionless, symbolic constant.
 *
 * \tparam VKind: what constant is represented
 */
template<value_kind VKind>
struct constant<VKind,void> : public expr_base {
    static constexpr value_kind vkind = VKind;
    static constexpr op_codes opcode = op_constant;

    static_assert( is_constant_kind(vkind), "illegal kind specifier" );

    constexpr GG_INLINE constant() { }

    template<typename Tr>
    static auto expand() {
	if constexpr ( vkind == vk_zero )
	    return value<Tr, vk_zero>();
	else if constexpr ( vkind == vk_cstone )
	    return value<Tr, vk_cstone>();
	else if constexpr ( vkind == vk_false )
	    return value<Tr, vk_false>();
	else if constexpr ( vkind == vk_true )
	    return value<Tr, vk_true>();
	else if constexpr ( vkind == vk_truemask )
	    return value<Tr, vk_truemask>();
	else if constexpr ( vkind == vk_inc )
	    return value<Tr, vk_inc>();
    }
};

static const constant<vk_zero,void> _0;
static const constant<vk_cstone,void> _1;
static const constant<vk_false,void> _false;
static const constant<vk_true,void> _true;
static const constant<vk_truemask,void> _1s;
static const constant<vk_inc,void> _0123;

template<typename T>
constexpr auto _c( T val ) {
    return constant<vk_any,T>( val );
}

/**
 * Expand a constant into a value given a context that sets the type
 *
 * The expression passed to the first argument is returned as-is if it
 * is not a constant.
 *
 * \param c constant expression to be expanded to value expression with
 *        data_type filled out
 * \param e expression that determines data_type
 */
template<typename Cst, typename Expr>
auto expand_cst( Cst c, Expr e ) {
    if constexpr ( c.opcode == op_constant ) {
	static_assert( !is_constant<Expr>::value,
		       "cannot infer data_type for constant" );
	return c.template expand<typename Expr::data_type>();
    } else
	return c;
}

template<typename DataType, value_kind VKind, typename T>
constexpr auto _xp( constant<VKind,T> c ) {
    return c.template expand<DataType>();
}


} // namespace expr

#endif // GRAPTOR_DSL_AST_CONSTANT_H
