// -*- c++ -*-
#ifndef GRAPTOR_DSL_AST_VALUE_H
#define GRAPTOR_DSL_AST_VALUE_H

namespace expr {

/* value
 * A value, typically a scalar.
 * The kind represents the source index (vk_src), destination index (vk_dst),
 * a mask, or any other value (vk_any).
 */
template<typename Tr, value_kind VKind, typename Enable>
struct value : public expr_base {
    using data_type = Tr;
    using element_type = typename data_type::element_type;
    using type = element_type; // legacy
    static constexpr unsigned short VL = Tr::VL;
    static constexpr value_kind vkind = VKind;
    static constexpr op_codes opcode = op_value;

    constexpr GG_INLINE value() { }
};

/* value
 * Special case with value
 */
template<typename Tr, value_kind VKind>
struct value<Tr,VKind,std::enable_if_t<value_kind_has_value<VKind>::value>>
 : public expr_base {
    using data_type = Tr;
    using element_type = typename data_type::element_type;
    using type = element_type; // legacy
    static constexpr unsigned short VL = Tr::VL;
    static constexpr value_kind vkind = VKind;
    static constexpr op_codes opcode = op_value;

    constexpr GG_INLINE value( element_type data ) : m_data( data ) { }

    GG_INLINE type data() const { return m_data; }

private:
    element_type m_data;
};

template<typename Expr>
auto true_val( Expr e ) {
    constexpr unsigned short VL = Expr::VL;
    using Tr = typename Expr::data_type;
    using MTr = typename Tr::prefmask_traits;
    return add_mask( value<MTr, vk_true>(), get_mask( e ) );
}

template<typename Expr>
auto false_val( Expr e ) {
    constexpr unsigned short VL = Expr::VL;
    using T = typename Expr::type;
    using Tr = simd::ty<T,VL>;
    return add_mask( value<Tr, vk_false>(), get_mask( e ) );
}

template<typename Expr>
auto zero_val( Expr e ) {
    constexpr unsigned short VL = Expr::VL;
    using T = typename Expr::type;
    using Tr = simd::ty<T,VL>;
    return add_mask( value<Tr, vk_zero>(), get_mask( e ) );
}

template<typename Expr>
auto allones_val( Expr e ) {
    constexpr unsigned short VL = Expr::VL;
    using T = typename Expr::type;
    using Tr = simd::ty<T,VL>;
    return add_mask( value<Tr, vk_truemask>(), get_mask( e ) );
}

template<typename Expr>
auto constant_val_one( Expr e ) {
    constexpr unsigned short VL = Expr::VL;
    using T = typename Expr::type;
    using Tr = simd::ty<T,VL>;
    return add_mask( value<Tr, vk_cstone>(), get_mask( e ) );
}

template<typename Expr>
auto constant_val( Expr e, typename Expr::type v ) {
    constexpr unsigned short VL = Expr::VL;
    using T = typename Expr::type;
    using Tr = simd::ty<T,VL>;
    return add_mask( value<Tr, vk_any>( v ), get_mask( e ) );
}

template<typename T, typename Expr>
auto constant_val2( Expr e, T /*typename Expr::type*/ v ) {
    constexpr unsigned short VL = Expr::VL;
    using Tr = simd::ty<T,VL>;
    return add_mask( value<Tr, vk_any>( v ), get_mask( e ) );
}

template<typename Tr>
auto constant_val3( typename Tr::element_type v ) {
    constexpr unsigned short VL = Tr::VL;
    return value<Tr, vk_any>( v );
}


template<typename Expr>
auto increasing_val( Expr e ) {
    constexpr unsigned short VL = Expr::VL;
    using T = typename Expr::type;
    using Tr = simd::ty<T,VL>;
    return add_mask( value<Tr, vk_inc>(), get_mask( e ) );
}

/**********************************************************************
 * Utilities
 **********************************************************************/
/* Aux, used by rewriting rules for refop
 * May be extended to accept small alterations, e.g., cast, vectors, ...
 */
template<typename Tr, typename Expr>
constexpr auto replace_pid( value<Tr,vk_zero>, Expr e ) {
    return e;
}

/*
template<typename Tr, typename Ur, typename Expr>
constexpr auto replace_pid(
    binop<value<Tr,vk_zero>,value<Ur,vk_smk>,binop_mask>, Expr e ) {
    return add_mask( e, value<Ur,vk_smk>() );
}
*/

template<typename Tr, typename Expr, typename Mask>
constexpr auto replace_pid(
    binop<value<Tr,vk_zero>,Mask,binop_mask> b, Expr e ) {
    // return add_mask( e, b.data2() );
    return e;
}


} // namespace expr

#endif // GRAPTOR_DSL_AST_VALUE_H
