// -*- c++ -*-
#ifndef GRAPTOR_DSL_AST_DECL_H
#define GRAPTOR_DSL_AST_DECL_H

// Abstract language:
// v value
//   values can be of C++ type, or vectors
//   cpp_value
//   abstract_value
// unop v -> v
// v binop v -> v
// A[v] -> v (r-value)
// A[v] R= v (reduction) -> v (bools)
// rvalue = value; lvalue = ref
// 
// Example: reduce( lvalue(A,d), rvalue(B,s) )
// scalar materialisation: += ( &A[d], B[s] );
// v-s materialisation (CSC): s-reduce( lvalue(A,d), v-gather(B,s) )
// v-v materialisation: scatter( A,d, v-sum( gather(A,d), v-gather(B,s) ) )
// v-s + mask: s-reduce( lvalue(A,d), v-gather(B,s,mask,+-zero) )
//
// Rewrite:
//    reduce -> if tgt is v-lvalue, then v-reduce, else s-reduce
//           -> if rhs is v-lvalue, then apply reduce to rhs first, else take rhs
//   consider valid combinations for reduce: ss, vs, vv; sv not valid? (CSR)
//      ... sv could also be valid: broadcast s to v at reduce

namespace expr {

// Kinds of values
enum value_kind {
    vk_any = 0,
    vk_src = 1,
    vk_smk = 2,
    vk_dst = 3,
    vk_dmk = 4,
    vk_mask = 5,
    vk_true = 6,
    vk_false = 7,
    vk_zero = 8,
    vk_vid = 9,
    vk_pid = 10,
    vk_inc = 11,
    vk_cstone = 12,
    vk_truemask = 13,
    vk_edge = 14,
    vk_eweight = 15
};

template<value_kind vk>
struct value_kind_has_value : public std::false_type { };

template<>
struct value_kind_has_value<vk_any> : public std::true_type { };

// Types of AST nodes
enum op_codes {
    op_value,
    op_constant,
    op_array,
    op_bitarray,
    op_unop,
    op_binop,
    op_ternop,
    op_cacheop,
    op_storeop,
    op_ntstoreop,
    op_refop,
    op_maskrefop,
    op_redop,
    op_dfsaop,
    op_noop,
    op_scalar,
    op_NUM
};

template<typename Cst, typename Expr>
auto expand_cst( Cst c, Expr e );

/***********************************************************************
 * Pre-declarations
 ***********************************************************************/
struct expr_base { };

struct noop;

template<typename Tr, value_kind VKind, typename Enable = void>
struct value;

template<value_kind VKind, typename T = void>
struct constant;

template<typename E, typename UnOp>
struct unop;

template<typename E1, typename E2, typename BinOp>
struct binop;

template<typename S, typename U, typename C, typename TernOp>
struct ternop;

template<unsigned cid, typename Tr>
struct cacheop;

template<typename A, typename T, unsigned short VL_>
struct refop;

template<typename A, typename T, typename M, unsigned short VL_>
struct maskrefop;

template<typename E1, typename E2, typename RedOp>
struct redop;

template<typename S, typename U, typename C, typename DFSAOp>
struct dfsaop;

// Pre-declared binary operations
struct binop_mask;
struct binop_setmask;
struct binop_predicate;
struct binop_land;
struct binop_add;
struct binop_mul;

// Pre-declared unary operations
template<unsigned short VL, bool aligned = true>
struct unop_incseq;

template<unsigned short VL>
struct unop_broadcast;

template<typename T, unsigned short VL>
struct unop_cvt;

template<typename T>
struct unop_cvt_data_type;

template<typename RedOp>
struct unop_reduce;

template<unsigned short VL>
struct unop_switch_to_vector;

// Pre-declared reduction operations
struct redop_logicalor;
template<bool conditional = true>
struct redop_add;
struct redop_mul;

// Pre-declared DFSA operations
struct dfsaop_MIS;

/***********************************************************************
 * Traits
 ***********************************************************************/
template<typename T, typename Enable = void>
struct is_expr : public std::false_type { };

template<typename T>
struct is_expr<T,std::enable_if_t<std::is_class_v<T>>> {
    static constexpr bool value = ( T::opcode < op_NUM );
};

template<typename T>
constexpr bool is_expr_v = is_expr<T>::value;

template<typename T>
struct is_noop {
    static constexpr bool value = ( T::opcode == op_noop );
};

template<typename T>
struct is_not_noop {
    static constexpr bool value = ( T::opcode != op_noop );
};

template<typename T>
struct is_constant {
    static constexpr bool value = ( T::opcode == op_constant );
};

template<typename T>
struct is_value {
    static constexpr bool value = ( T::opcode == op_value );
};

template<typename T, value_kind VKind, typename Enable = void>
struct is_value_vk : std::false_type { };

template<typename Tr, value_kind VKind>
struct is_value_vk<Tr,VKind,typename std::enable_if<is_value<Tr>::value>::type> {
    static constexpr bool value = ( Tr::vkind == VKind );
};

template<typename T>
struct is_binop {
    static constexpr bool value = ( T::opcode == op_binop );
};

template<typename T, typename = void>
struct is_binop_mask : std::false_type { };

template<typename T>
struct is_binop_mask<T,typename std::enable_if<is_binop<T>::value>::type> {
    static constexpr bool value =
	std::is_same<typename T::op_type,binop_mask>::value;
};

template<typename T, typename = void>
struct is_binop_setmask : std::false_type { };

template<typename T>
struct is_binop_setmask<T,typename std::enable_if<is_binop<T>::value>::type> {
    static constexpr bool value =
	std::is_same<typename T::op_type,binop_setmask>::value;
};

template<typename T, typename = void>
struct is_binop_land : std::false_type { };

template<typename T>
struct is_binop_land<T,typename std::enable_if<is_binop<T>::value>::type> {
    static constexpr bool value =
	std::is_same<typename T::op_type,binop_land>::value;
};

template<typename T>
struct is_binop_arithmetic : std::false_type { };

template<>
struct is_binop_arithmetic<binop_add> : std::true_type { };

template<>
struct is_binop_arithmetic<binop_mul> : std::true_type { };

template<typename T>
struct is_refop {
    static constexpr bool value = ( T::opcode == op_refop );
};

template<typename T>
struct is_redop {
    static constexpr bool value = ( T::opcode == op_redop );
};

template<typename T>
struct is_dfsaop {
    static constexpr bool value = ( T::opcode == op_dfsaop );
};

template<typename T>
struct is_maskrefop {
    static constexpr bool value = ( T::opcode == op_maskrefop );
};

template<typename T>
struct is_cacheop {
    static constexpr bool value = ( T::opcode == op_cacheop );
};

template<typename E>
struct is_masked_cacheop : public std::false_type { };

template<unsigned cid, typename Tr, typename M>
struct is_masked_cacheop<binop<cacheop<cid,Tr>,M,binop_mask>>
    : public std::true_type { };

template<typename T>
struct is_storeop {
    static constexpr bool value = ( T::opcode == op_storeop );
};

template<typename T>
struct is_ntstoreop {
    static constexpr bool value = ( T::opcode == op_ntstoreop );
};

template<typename T>
struct is_unop {
    static constexpr bool value = ( T::opcode == op_unop );
};

template<typename T>
struct unop_is_incseq : std::false_type { };

template<unsigned short VL>
struct unop_is_incseq<unop_incseq<VL>> : std::true_type { };

template<typename T>
struct unop_is_cvt : std::false_type { };

template<typename T, unsigned short VL>
struct unop_is_cvt<unop_cvt<T,VL>> : std::true_type { };

template<typename T>
struct unop_is_cvt_data_type : std::false_type { };

template<typename Tr>
struct unop_is_cvt_data_type<unop_cvt_data_type<Tr>> : std::true_type { };

template<typename T, typename Enable = void>
struct is_unop_cvt_data_type : std::false_type { };

template<typename T>
struct is_unop_cvt_data_type<T,std::enable_if_t<is_unop<T>::value>>
    : unop_is_cvt_data_type<typename T::unop_type> { };

template<typename T>
struct unop_is_broadcast : std::false_type { };

template<unsigned short VL>
struct unop_is_broadcast<unop_broadcast<VL>> : std::true_type { };

template<typename T>
struct unop_is_reduce : std::false_type { };

template<typename RedOp>
struct unop_is_reduce<unop_reduce<RedOp>> : std::true_type { };

template<typename T, typename Enable = void>
struct is_unop_incseq : std::false_type { };

template<typename T>
struct is_unop_incseq<T,typename std::enable_if<is_unop<T>::value>::type>
    : unop_is_incseq<typename T::unop_type> { };

template<typename T>
struct is_not_unop_incseq {
    static constexpr bool value = ! is_unop_incseq<T>::value;
};

// TODO: used?
template<typename E1, typename E2, typename U = void>
using enable_if_compatible = std::enable_if<
    std::is_same<typename E1::type, typename E2::type>::value
    && ( E1::VL == E2::VL || E1::VL == 1 || E2::VL == 1), U>;

// Analysis

template<typename E>
struct is_scalar;

/* is_vector
 * Determine if an argument is a scalar or vector type. If vector type, also
 * calculate the vector length of the expression.
 */
template<typename E>
struct is_vector; // also calculate vector length

template<value_kind VKind, typename E>
struct is_vk_opt_mask : std::false_type { };

template<value_kind VKind, typename Tr, value_kind VKind2>
struct is_vk_opt_mask<VKind,value<Tr, VKind2>> {
    static constexpr bool value = (VKind == VKind2);
};

template<value_kind VKind, typename E1, typename E2>
struct is_vk_opt_mask<VKind,binop<E1,E2,binop_mask>> : is_vk_opt_mask<VKind,E1> { };

template<value_kind VKind, typename E, unsigned short VL>
struct is_vk_opt_mask<VKind,unop<E,unop_incseq<VL>>> : is_vk_opt_mask<VKind,E> { };

// Does an expression depend only on a value of kind VKind and constant values?
// This precludes, vk_src, vk_dst, etc if not mentioned, but allows
// vk_any, vk_zero, etc. vk_smk etc should appear only in the mask argument
// of binop_mask. Cacheop, refop, storeop, etc should not appear.
template<value_kind VKind, typename E>
struct depends_only_on : std::false_type { };

template<value_kind VKind, typename Tr, value_kind VKind2>
struct depends_only_on<VKind,value<Tr,VKind2>> {
    // Exclude value types that are looked up in a value_map during evaluation
    static constexpr bool value = VKind2 == VKind
	|| ( VKind2 != vk_src && VKind2 != vk_dst && VKind2 != vk_smk
	     && VKind2 != vk_dmk && VKind2 != vk_mask
	     && VKind2 != vk_vid && VKind2 != vk_pid
	     && VKind2 != vk_edge );
};

template<value_kind VKind, typename E, typename UnOp>
struct depends_only_on<VKind,unop<E,UnOp>>
    : public depends_only_on<VKind,E> { };

template<value_kind VKind, typename E1, typename E2, typename BinOp>
struct depends_only_on<VKind,binop<E1,E2,BinOp>> {
    static constexpr bool value
	= depends_only_on<VKind,E1>::value
	&& depends_only_on<VKind,E2>::value;
};

// This restored old behavior encapsulated in is_vk_opt_mask, however it
// is questionable that this is correct in general.
template<value_kind VKind, typename E1, typename E2>
struct depends_only_on<VKind,binop<E1,E2,binop_mask>>
    : public depends_only_on<VKind,E1> { };

// To determine if an expression is cacheable
template<value_kind VKind, typename E>
struct is_indexed_by_vk : std::false_type { };

template<value_kind VKind, typename A, typename T, unsigned short VL>
struct is_indexed_by_vk<VKind,refop<A,T,VL>> : depends_only_on<VKind,T> { };

template<typename E>
struct is_indexed_by_src : is_indexed_by_vk<expr::vk_src,E> { };

template<typename E>
struct is_indexed_by_dst : is_indexed_by_vk<expr::vk_dst,E> { };

template<typename Expr>
struct is_constant_false : std::false_type { };

template<typename Tr>
struct is_constant_false<value<Tr,vk_false>> : std::true_type { };

template<typename E, typename M>
struct is_constant_false<binop<E,M,binop_mask>> : is_constant_false<E> { };

template<typename Expr>
struct is_constant_true : std::false_type { };

template<typename Tr>
struct is_constant_true<value<Tr,vk_true>> : std::true_type { };

template<typename E, typename M>
struct is_constant_true<binop<E,M,binop_mask>> : is_constant_true<E> { };

template<typename E>
struct is_constant_true<unop<E,unop_reduce<redop_logicalor>>>
    : is_constant_true<E> { };

template<typename E, unsigned short VL>
struct is_constant_true<unop<E,unop_switch_to_vector<VL>>>
    : is_constant_true<E> { };

template<typename Expr>
struct is_constant_zero : std::false_type { };

template<typename Tr>
struct is_constant_zero<value<Tr,vk_zero>> : std::true_type { };

template<typename E, typename M>
struct is_constant_zero<binop<E,M,binop_mask>> : is_constant_zero<E> { };

template<typename E>
struct is_indexed_by_zero : std::false_type { };

template<typename A, typename T, unsigned short VL>
struct is_indexed_by_zero<refop<A,T,VL>> : is_constant_zero<T> { };

/***********************************************************************
 * Does an expression contain a binop_mask or maskrefop?
 ***********************************************************************/
template<typename E>
struct expr_contains_mask : std::false_type { };

template<typename E, typename UnOp>
struct expr_contains_mask<unop<E,UnOp>> : public expr_contains_mask<E> { };

template<typename E1, typename E2, typename BinOp>
struct expr_contains_mask<binop<E1,E2,BinOp>> {
    static constexpr bool value =
	std::is_same_v<BinOp,binop_mask>
	|| std::is_same_v<BinOp,binop_setmask>
	|| std::is_same_v<BinOp,binop_predicate>
	|| expr_contains_mask<E1>::value || expr_contains_mask<E2>::value;
};

template<typename E1, typename E2, typename E3, typename TernOp>
struct expr_contains_mask<ternop<E1,E2,E3,TernOp>> {
    static constexpr bool value =
	expr_contains_mask<E1>::value
	|| expr_contains_mask<E2>::value
	|| expr_contains_mask<E3>::value;
};

template<typename A, typename T, typename M, unsigned short VL>
struct expr_contains_mask<maskrefop<A,T,M,VL>> : public std::true_type { };

template<typename A, typename T, unsigned short VL>
struct expr_contains_mask<refop<A,T,VL>> : public expr_contains_mask<T> { };

template<typename E1, typename E2, typename RedOp>
struct expr_contains_mask<redop<E1,E2,RedOp>> {
    static constexpr bool value =
	expr_contains_mask<E1>::value || expr_contains_mask<E2>::value;
};

template<typename S, typename U, typename C, typename DFSAOp>
struct expr_contains_mask<dfsaop<S,U,C,DFSAOp>> {
    static constexpr bool value =
	expr_contains_mask<S>::value
	|| expr_contains_mask<U>::value
	|| expr_contains_mask<C>::value;
};

template<typename E>
constexpr bool expr_contains_mask_v = expr_contains_mask<E>::value;

/***********************************************************************
 * Does an expression contain specific vk?
 ***********************************************************************/
template<value_kind VKind, typename E>
struct expr_contains_vk : std::true_type { };

template<value_kind VKind>
struct expr_contains_vk<VKind,noop> : public std::false_type { };

template<value_kind VKind, typename Tr, value_kind vk>
struct expr_contains_vk<VKind,value<Tr,vk>> {
    static constexpr bool value = VKind == vk;
};

template<value_kind VKind, typename E, typename UnOp>
struct expr_contains_vk<VKind,unop<E,UnOp>>
    : public expr_contains_vk<VKind,E> { };

template<value_kind VKind, typename E1, typename E2, typename BinOp>
struct expr_contains_vk<VKind,binop<E1,E2,BinOp>> {
    static constexpr bool value =
	expr_contains_vk<VKind,E1>::value || expr_contains_vk<VKind,E2>::value;
};

template<value_kind VKind,
	 typename E1, typename E2, typename E3, typename TernOp>
struct expr_contains_vk<VKind,ternop<E1,E2,E3,TernOp>> {
    static constexpr bool value =
	expr_contains_vk<VKind,E1>::value
	|| expr_contains_vk<VKind,E2>::value
	|| expr_contains_vk<VKind,E3>::value;
};

template<value_kind VKind,
	 typename A, typename T, typename M, unsigned short VL>
struct expr_contains_vk<VKind,maskrefop<A,T,M,VL>> {
    static constexpr bool value =
	expr_contains_vk<VKind,T>::value
	|| expr_contains_vk<VKind,M>::value;
};

template<value_kind VKind, typename A, typename T, unsigned short VL>
struct expr_contains_vk<VKind,refop<A,T,VL>>
    : public expr_contains_vk<VKind,T> { };

template<value_kind VKind, typename E1, typename E2, typename RedOp>
struct expr_contains_vk<VKind,redop<E1,E2,RedOp>> {
    static constexpr bool value =
	expr_contains_vk<VKind,E1>::value || expr_contains_vk<VKind,E2>::value;
};

template<value_kind VKind, typename S, typename U, typename C, typename DFSAOp>
struct expr_contains_vk<VKind,dfsaop<S,U,C,DFSAOp>> {
    static constexpr bool value =
	expr_contains_vk<VKind,S>::value
	|| expr_contains_vk<VKind,U>::value
	|| expr_contains_vk<VKind,C>::value;
};

template<value_kind VKind, typename E>
constexpr bool expr_contains_vk_v = expr_contains_vk<VKind,E>::value;

/***********************************************************************
 * Expressions are identical if they have the same C++ type and they do
 * not contain data values (arrays with the same AID are assumed to contain
 * the same pointer).
 ***********************************************************************/
template<typename E1, typename E2>
struct is_identical {
    static constexpr bool value = std::is_same_v<E1,E2>
	&& !expr_contains_vk_v<vk_any,E1>;
};

/***********************************************************************
 * No-operation
 ***********************************************************************/
struct noop : public expr_base {
    using type = void;
    static constexpr unsigned short VL = 0;
    static constexpr op_codes opcode = op_noop;

    constexpr GG_INLINE noop() { }
};

static constexpr
noop make_noop() { return noop(); }

} // namespace expr

#endif // GRAPTOR_DSL_AST_DECL_H
