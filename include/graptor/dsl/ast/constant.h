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
 * constant: a type-less, dimensionless, symbolic constant.
 *
 * \tparam VKind: what constant is represented
 */
template<value_kind VKind>
struct constant : public expr_base {
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

static const constant<vk_zero> _0;
static const constant<vk_cstone> _1;
static const constant<vk_false> _false;
static const constant<vk_true> _true;
static const constant<vk_truemask> _1s;
static const constant<vk_inc> _0123;

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
	return Cst::template expand<typename Expr::data_type>();
    } else
	return c;
}

} // namespace expr

#endif // GRAPTOR_DSL_AST_CONSTANT_H
