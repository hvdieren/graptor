// -*- c++ -*-
#ifndef GRAPTOR_DSL_AST_MEMREF_H
#define GRAPTOR_DSL_AST_MEMREF_H

#include <type_traits>

#include "graptor/encoding.h"
#include "graptor/utils.h"

namespace expr {

/* refop
 * A reference to an array element. Can become both an lvalue or an rvalue.
 */

/* array_aid
 * Internal IDs to disambiguate arrays. It was hoped that multiple AIDs
 * could be used for old and new frontiers in order to track where the
 * definition of the frontier came from. While we can map these to the
 * some pointers using extract_pointer_set(), this idea breaks down when
 * we assume that cached arrays are consistently remapped to the cache.
 * Unfortunately, that mechanism does not recognize that dicstinct AIDs
 * map to the same arrays.
 */
enum array_aid {
    aid_graph_index = -16,
    aid_frontier_old = -32,
    aid_frontier_old_f = -32,
    aid_frontier_old_2 = -32,
    aid_frontier_new = -48,
    aid_frontier_new_f = -48,
    aid_frontier_new_2 = -48,
    aid_frontier_new_2b = -48,
    aid_frontier_new_store = -48,
    aid_frontier_nactv = -64,
    aid_frontier_nacte = -80,
    aid_redir = -96,
    aid_utilization_active = -112,
    aid_utilization_vectors = -128,
    aid_graph_degree = -144,
    aid_frontier_a = -160,
    aid_emap_zerof = -176,
    aid_emap_let = -192,
    aid_rnd_state0 = -208,
    aid_rnd_state1 = -224,
    aid_rnd_tmp0 = -240,
    aid_rnd_tmp1 = -256,
    aid_rnd_tmp2 = -272,
    aid_udiv_p = -288,
    aid_udiv_m = -304,
    aid_priv = -1024 // a very big downward range is required
};

static inline constexpr array_aid cid_to_aid( unsigned int cid ) {
    return array_aid( unsigned(aid_priv) - cid * 16 );
}

static inline constexpr unsigned int aid_to_cid( array_aid aid ) {
    return ( unsigned(aid_priv) - unsigned(aid) ) / 16;
}

constexpr short aid_key( array_aid a ) {
    // Negative values alias, positive don't
    return short(a) < 0 ? - ( short(-a) >> 4 ) : short(a);
}

template<array_aid AID0, array_aid AID1>
struct aid_is_aliased {
    static constexpr bool value = aid_key(AID0) == aid_key(AID1);
};

constexpr bool aid_aliased( array_aid a0, array_aid a1 ) {
    return aid_key( a0 ) == aid_key( a1 );
}

template<typename T, typename U, short AID_,
	 typename Encoding = array_encoding<T>,
	 bool NT_ = false>
struct array_ro;

// array_intl has no data members.
template<typename T, typename U, short AID_ /*= -1*/,
	 typename Encoding = array_encoding<T>,
	 bool NT_ = false>
struct array_intl : expr_base {
    using type = T;
    using index_type = U;
    using encoding = Encoding;
    using storage_type = typename encoding::storage_type;
    static constexpr short AID = AID_;
    static constexpr bool NT = NT_;

    static constexpr op_codes opcode = op_array;

    template<short NewAID = AID>
    static auto replace_array( storage_type *ptr ) {
	if constexpr ( is_array_encoding_zero_v<encoding> ) {
	    using Enc = typename encoding::base_encoding;
	    return array_ro<type,index_type,NewAID,Enc,NT>( ptr );
	} else
	    return array_ro<type,index_type,NewAID,encoding,NT>( ptr );
    }

    // AST builder
    template<typename E>
    typename std::enable_if<std::is_same<index_type, typename E::type>::value,
			    refop<array_intl<T,U,AID,encoding,NT>, E, E::VL>>::type
	operator[] ( E idx ) const {
	return refop<array_intl<T,U,AID,encoding,NT>, E, E::VL>( *this, idx );
    }
};

template<typename T, typename U, short AID_ /*= -1*/,
	 typename Encoding, // = array_encoding<T>,
	 bool NT_> // = false>
struct array_ro : public array_intl<T,U,AID_,Encoding,NT_> {
    using parent_type = array_intl<T,U,AID_,Encoding,NT_>;
    using type = typename parent_type::type;
    using index_type = typename parent_type::index_type;
    using encoding = typename parent_type::encoding;
    using storage_type = typename parent_type::storage_type;

    static constexpr short AID = parent_type::AID;
    static constexpr bool NT = parent_type::NT;
    static constexpr op_codes opcode = op_array;

    constexpr array_ro( storage_type *ptr ) : m_ptr( ptr ) { }

    template<short NewAID = AID>
    static auto replace_array( storage_type *ptr ) {
	if constexpr ( is_array_encoding_zero_v<encoding> ) {
	    using Enc = typename encoding::base_encoding;
	    return array_ro<type,index_type,NewAID,Enc,NT>( ptr  );
	} else
	    return array_ro<type,index_type,NewAID,encoding,NT>( ptr  );
    }

    // Should be read-only/const in _ro, modifiable ref in array_update
    auto
    operator[] ( index_type idx ) {
	return encoded_element_ref<encoding, simd::detail::vdata_traits<type,1>,
				   index_type>( ptr(), idx );
    }

    // AST builder
    template<typename E>
    typename std::enable_if<std::is_same<index_type, typename E::type>::value,
			    refop<array_ro<T,U,AID,encoding,NT>, E, E::VL>>::type
    operator[] ( E idx ) const {
	return refop<array_ro<T,U,AID,encoding,NT>, E, E::VL>( *this, idx );
    }

    template<bool newNT>
    auto rebindNT() {
	return array_ro<type,index_type,AID,encoding,newNT>( ptr() );
    }

    template<typename S>
    auto observe() const {
	return array_ro<S,index_type,AID,encoding,NT>( ptr() );
    }

    auto observe_stored() const {
	return observe<typename encoding::stored_type>();
    }

    GG_INLINE storage_type *ptr() const { return m_ptr; }

private:
    storage_type * m_ptr;
};

template<typename T, typename U, short AID_ = -1>
struct bitarray_intl : expr_base {
    using type = void; // tell expression you won't get a value, only mask
    using stype = T;
    using index_type = U;
    static constexpr short AID = AID_;

    static constexpr op_codes opcode = op_bitarray;

    static_assert( !is_logical<T>::value, "logical types not allowed here" );

    constexpr bitarray_intl() { }

    // Should be read-only/const in _ro, modifiable ref in array_update
/*
    bool operator[] ( VID idx ) const {
	return ( m_ptr[idx/(sizeof(stype)*8)] >> (idx % (sizeof(stype)*8)) ) & 1;
    }
*/
    
    // AST builder
    template<typename E>
    typename std::enable_if<std::is_same<index_type, typename E::type>::value,
			    refop<bitarray_intl<T,U,AID>, E, E::VL>>::type
	operator[] ( E idx ) const {
	// static_assert( sizeof(T)*8 == E::VL, "vector length mismatch" );
	return refop<bitarray_intl<T,U,AID>, E, E::VL>( *this, idx );
    }
};

template<typename T, typename U, short AID_ = -1>
struct bitarray_ro : bitarray_intl<T,U,AID_> {
    using type = void; // tell expression you won't get a value, only mask
    using stype = T;
    using index_type = U;
    static constexpr short AID = AID_;

    static constexpr op_codes opcode = op_bitarray;

    static_assert( !is_logical<T>::value, "logical types not allowed here" );

    constexpr bitarray_ro( stype *ptr ) : m_ptr( ptr ) { }

    // Should be read-only/const in _ro, modifiable ref in array_update
    bool operator[] ( VID idx ) const {
	return ( m_ptr[idx/(sizeof(stype)*8)] >> (idx % (sizeof(stype)*8)) ) & 1;
    }
    
    // AST builder
    template<typename E>
    typename std::enable_if<std::is_same<index_type, typename E::type>::value,
			    refop<bitarray_ro<T,U,AID>, E, E::VL>>::type
	operator[] ( E idx ) {
	// static_assert( sizeof(T)*8 == E::VL, "vector length mismatch" );
	return refop<bitarray_ro<T,U,AID>, E, E::VL>( *this, idx );
    }

    GG_INLINE stype *ptr() { return m_ptr; }

private:
    stype * m_ptr;
};

template<frontier_type ftype, typename T, typename U, short AID,
	 typename Enc, bool NT = false, bool WithPtr = false,
	 typename Enable = void>
struct array_select {
    using type = array_intl<T,U,AID,Enc,NT>;
};

template<frontier_type ftype, typename T, typename U, short AID,
	 typename Enc, bool NT>
struct array_select<ftype,T,U,AID,Enc,NT,true,
		    std::enable_if_t<ftype != frontier_type::ft_bit
				     && ftype != frontier_type::ft_bit2>> {
    using type = array_ro<T,U,AID,Enc,NT>;
};

template<typename T, typename U, short AID, typename Enc>
struct array_select<frontier_type::ft_bit2,T,U,AID,Enc,false,false> {
    using type = array_intl<bitfield<2>,U,AID,array_encoding_bit<2>,false>;
};

template<typename T, typename U, short AID, typename Enc>
struct array_select<frontier_type::ft_bit2,T,U,AID,Enc,false,true> {
    using type = array_ro<bitfield<2>,U,AID,array_encoding_bit<2>,false>;
};

template<typename T, typename U, short AID, typename Enc>
struct array_select<frontier_type::ft_bit,T,U,AID,Enc,false,false> {
    using type = bitarray_intl<T,U,AID>;
};

template<typename T, typename U, short AID, typename Enc>
struct array_select<frontier_type::ft_bit,T,U,AID,Enc,true,false> {
    using type = bitarray_intl<T,U,AID>;
};

template<typename T, typename U, short AID, typename Enc>
struct array_select<frontier_type::ft_bit,T,U,AID,Enc,false,true> {
    using type = bitarray_ro<T,U,AID>;
};

template<typename T, typename U, short AID, typename Enc>
struct array_select<frontier_type::ft_bit,T,U,AID,Enc,true,true> {
    using type = bitarray_ro<T,U,AID>;
};

template<typename A, typename T>
inline auto make_refop( A a, T t );

template<typename A, typename T, unsigned short VL_>
struct refop : public expr_base {
    static constexpr unsigned short VL = VL_; // TODO: copy from T
    static constexpr op_codes opcode = op_refop;

    using type = typename A::type; // blended as vector if T is vector
    using data_type = simd::ty<type,VL>;
    using array_type = A;
    using index_type = typename T::type;
    using idx_type = T;
    using self_type = refop<A,T,VL>;


    // If VL > 1 and T::VL == 1 then we assume that the index is a scalar index
    // that is to indicate a set of consecutive locations starting at the
    // listed index.
    static_assert( VL == T::VL || T::VL == 1, "need matching VL" );

    GG_INLINE refop( A array, T idx ) : m_array( array ), m_idx( idx ) { }

    GG_INLINE const A & array() const { return m_array; }
    GG_INLINE const T & index() const { return m_idx; }

    template<typename Expr>
    constexpr auto replace_array( type * a, Expr e ) const {
	// It is assumed that m_idx is zero. We replace the zero by
	// vk_pid.
	return make_refop( array_type::replace_array( a ),
			   replace_pid( m_idx, e ) );
    }
    template<short AID, typename Expr>
    constexpr auto replace_array( type * a, Expr e ) const {
	// It is assumed that m_idx is zero. We replace the zero by
	// vk_pid.
	return make_refop( array_type::template replace_array<AID>( a ),
			   replace_pid( m_idx, e ) );
    }
    template<short AID, typename Enc, typename Expr>
    constexpr auto replace_array( type * a, Expr e ) const {
	// It is assumed that m_idx is zero. We replace the zero by
	// vk_pid.
	return make_refop( array_type::template replace_array<AID,Enc>( a ),
			   replace_pid( m_idx, e ) );
    }
    template<typename Expr>
    constexpr auto replace_index( Expr idx ) const {
	return make_refop( m_array, idx );
    }

    // TODO: for correctness reasons, abstract as reduce so that there is only
    //       one reduce operation per array, no mixing of min, or, add, ...
    //       A possible non-associative sequence is { r += a; r *= b; }
    //       (interleaving two such sequences on same r is meaningless)
    //       but { r += a; r += b; } is ok
    template<typename E>
    auto min( E rhs );

    template<typename E, typename C>
    auto min( E rhs, C cond );

    template<typename E>
    auto max( E rhs );

    template<typename E, typename C>
    auto max( E rhs, C cond );

    template<typename E>
    auto operator += ( E rhs );

    template<typename E>
    auto operator *= ( E rhs );

    template<typename E>
    auto operator = ( E rhs );

    template<typename E, typename C>
    auto assign_if( E rhs, C cond );

    template<typename E>
    auto setif( E rhs );
    
    template<typename E>
    auto count_down( E rhs );
    
    template<typename E>
    auto count_down_value( E rhs );
    
    template<typename E>
    auto add( E rhs );
    
private:
    A m_array;
    T m_idx;
};

template<typename A, typename T>
auto make_refop( A a, T t ) {
    return refop<A,T,T::VL>( a, t );
}

template<typename A, typename T, typename M, unsigned short VL_>
struct maskrefop : public expr_base {
    static constexpr unsigned short VL = VL_; // TODO: copy from T

    using type = typename A::type; // blended as vector if T is vector
    using data_type = simd::ty<type,VL>;
    using array_type = A;
    using index_type = typename T::type;
    using mask_type = typename M::type;
    using idx_type = T;

    static constexpr op_codes opcode = op_maskrefop;

    static_assert( VL == T::VL, "need matching VL" );

    GG_INLINE maskrefop( A array, T idx, M mask )
	: m_array( array ), m_idx( idx ), m_mask( mask ) { }

    GG_INLINE const A & array() const { return m_array; }
    GG_INLINE const T & index() const { return m_idx; }
    GG_INLINE const M & mask() const { return m_mask; }
    
private:
    A m_array;
    T m_idx;
    M m_mask;
};

template<typename A, typename T, typename M>
auto make_maskrefop( A a, T t, M m ) {
    return maskrefop<A,T,M,T::VL>( a, t, m );
}

/************************************************************************
 * scalar value, typically used for reductions.
 *
 * The scalar type holds the value itself. It is non-copyable such that the
 * pointer to the value remains unique. Every use or dereference of the scalar
 * should immediately translate to a refop on an array_ro that points to the
 * actual scalar value. When creating the refop, the context of that creation
 * is used to infer the vector length of the refop.
 *
 * @param <T> type of the scalar value
 * @param <AID_> unique array ID for memory disambiguation
 ************************************************************************/
template<typename T, unsigned short AID_>
struct scalar : public expr_base, private NonCopyable<scalar<T,AID_>> {
    static constexpr op_codes opcode = op_scalar;
    static constexpr unsigned short AID = AID_;

    using type = T;			//!< type of scalar value
    using self_type = scalar<T,AID>;	//!< our own type

    /** Constructor: default initialisation
     */
    GG_INLINE scalar() { }

    /** Constructor: value initialisation
     *
     * @param[in] v initial value for the scalar
     */
    GG_INLINE scalar( type v ) : m_val( v ) { }

    /*! Create a refop for use in ASTs
     *
     * @tparam <VL> Vector length of resulting expression
     */
    template<unsigned short VL>
    auto make_refop() {
	return expr::make_refop(
	    array_ro<type,VID,AID,array_encoding<type>,false>( ptr() ),
	    value<simd::ty<VID,VL>,vk_zero>() );
    }

    /*! Retrieve writeable reference to actual scalar value.
     */
    type & operator * () { return *ptr(); }

    /*! Retrieve read-only reference to actual scalar value.
     */
    const type & operator * () const { return *ptr(); }

    /*! Retrieve writeable pointer to the actual scalar value.
     */
    type *ptr() { return &m_val; }

    /*! Retrieve read-only pointer to the actual scalar value.
     */
    const type *ptr() const { return &m_val; }

    /*! Retrieve value of the actual scalar.
     */
    operator type () const { return *ptr(); }

    /*! Assign a value to the scalar; use outside of ASTs
     *
     * @param[in] t The new value
     */
    const self_type & operator = ( type t ) { *ptr() = t; return *this; }

    /*! Create an AST node for a reduction operation using 'min'
     *
     * @tparam <E> The type of the RHS of the AST node
     * @param[in] rhs The RHS of the AST node
     */
    template<typename E>
    auto min( E rhs );

    /*! Create an AST node for a reduction operation using 'max'
     *
     * @tparam <E> The type of the RHS of the AST node
     * @param[in] rhs The RHS of the AST node
     */
    template<typename E>
    auto max( E rhs );

    /*! Create an AST node for a reduction operation using '+'
     *
     * The AST node will return a boolean indication that the value was
     * updated.
     *
     * @tparam <E> The type of the RHS of the AST node
     * @param[in] rhs The RHS of the AST node
     */
    template<typename E>
    auto operator += ( E rhs );

    /*! Create an AST node for a reduction operation using '*'
     *
     * @tparam <E> The type of the RHS of the AST node
     * @param[in] rhs The RHS of the AST node
     */
    template<typename E>
    auto operator *= ( E rhs );

    /*! Create an AST node for a reduction operation using '+'
     *
     * The AST node will return the updated value.
     *
     * @tparam <E> The type of the RHS of the AST node
     * @param[in] rhs The RHS of the AST node
     */
    template<typename E>
    auto add( E rhs );
    
private:
    type m_val; //!< The actual scalar value
};

/* cacheop
 * reference a cached value.
 * Note: this class uses the simd::detail::*_traits classes as type
 */
template<unsigned cid, typename Tr>
struct cacheop : public expr_base {
    using data_type = Tr;
    using type = typename data_type::member_type; // void for bitmask!

    static constexpr op_codes opcode = op_cacheop;
    static constexpr unsigned short VL = data_type::VL;

    cacheop() { }
};

template<bool nt, typename R, typename T>
struct storeop;

/* storeop
 * A store operation against a ref. Only for internal usage.
 */
template<typename R, typename T>
// disabled because bitarray_ro sets type to void
// typename = typename std::enable_if<
// std::is_same<typename R::type, typename T::type>::value>::type>
struct storeop<false,R,T> : public expr_base {
    using type = typename R::type; // same as T::type
    using data_type = typename R::data_type::prefmask_traits;

    static constexpr op_codes opcode = op_storeop;
    static constexpr unsigned short VL = T::VL;

    storeop( R ref, T val ) : m_ref( ref ), m_val( val ) { }

    const R & ref() const { return m_ref; }
    const T & value() const { return m_val; }

    template<typename VTr1, typename MTr1,
	     typename VTr2, typename MTr2,
	     layout_t Layout1, layout_t Layout2,
	     typename I, typename Enc, bool NT>
    __attribute__((always_inline))
    static inline auto
    evaluate( lvalue<VTr1,I,MTr1,Enc,NT,Layout1> l,
	      rvalue<VTr2,Layout2,MTr2> r ) {
	static_assert( !std::is_void_v<VTr1>, "require a type to store" );

	if constexpr ( std::is_void_v<VTr2> ) {
	    static_assert( simd::detail::is_mask_traits<VTr1>::value,
			   "a mask should be stored to a mask location" );
	
	    // Convert rvalue to correct type, matching lvalue
	    auto v = r.mask().template convert_data_type<VTr1>();
	    auto m = l.mask();

	    // Store the value conditionally for those lanes enabled by the mask
	    l.value().store( v, m );
	    return make_rvalue( v, m );
	} else {
	    // Convert rvalue to correct type, matching lvalue
	    auto v = r.value().template convert_data_type<VTr1>();
	    auto m = force_mask<VTr1>( join_mask<VTr1>( l.mask(), r.mask() ) );

	    // Store the value conditionally for those lanes enabled by the mask
	    l.value().store( v, m );
	    return make_rvalue( v, m );
	}
    }

    template<typename VTr1, layout_t Layout1,
	     typename VTr2, layout_t Layout2,
	     typename I, typename Enc, bool NT, typename MPack>
    __attribute__((always_inline))
    static inline auto
    evaluate( sb::lvalue<VTr1,I,Enc,NT,Layout1> l,
	      sb::rvalue<VTr2,Layout2> r,
	      const MPack & mpack ) {
	static_assert( !std::is_void_v<VTr1>, "require a type to store" );

	// Convert rvalue to correct type, matching lvalue
	auto v = r.value().template convert_data_type<VTr1>();
	using MTr = typename VTr1::prefmask_traits;
	if constexpr ( sb::is_empty_mask_pack_v<MPack> ) {
	    // Store all lanes
	    l.value().store( v );
	    return make_rvalue( v.true_mask(), mpack );
	} else if constexpr ( MPack::template has_mask<MTr>() ) {
	    auto m = mpack.template get_mask<MTr>();

	    // Store the value conditionally for those lanes enabled by the mask
	    l.value().store( v, m );
	    return make_rvalue( m, mpack );
	} else {
	    auto mpack2 = mpack.template clone_and_add<MTr>();
	    auto m = mpack2.template get_mask<MTr>();

	    // Store the value conditionally for those lanes enabled by the mask
	    l.value().store( v, m );
	    return make_rvalue( m, mpack2 );
	}
    }

private:
    R m_ref;
    T m_val;
};

template<typename R, typename T>
auto make_storeop( R r, T t ) {
    if constexpr ( t.opcode == op_constant )
	return make_storeop( r, expand_cst( t, r ) );
    else
	return storeop<false,R,T>( r, t );
}

/* ntstoreop
 * A store operation against a ref. Only for internal usage.
 */
template<typename R, typename T>
// disabled because bitarray_ro sets type to void
// typename = typename std::enable_if<
// std::is_same<typename R::type, typename T::type>::value>::type>
struct storeop<true,R,T> : public expr_base {
    using type = typename R::type; // same as T::type

    static constexpr op_codes opcode = op_ntstoreop;
    static constexpr unsigned short VL = T::VL;

    storeop( R ref, T val ) : m_ref( ref ), m_val( val ) { }

    const R & ref() const { return m_ref; }
    const T & value() const { return m_val; }

    template<typename VTr, typename MTr1, typename MTr2, typename I,
	     typename Enc, bool NT,
	     layout_t Layout1,  layout_t Layout2>
    __attribute__((always_inline))
    static inline auto
    evaluate( lvalue<VTr,I,MTr1,Enc,NT,Layout1> l, rvalue<VTr,Layout2,MTr2> r ) {
	auto mask = l.mask() && r.mask();
	// using M = typename decltype(mask)::member_type;
	// l.value().template store<M>( r.value(), mask );
	l.value().ntstore( r.value(), mask );
	return make_rvalue( r.value(), mask );
    }

    template<typename VTr, typename I, typename Enc, bool NT,
	     layout_t Layout1,  layout_t Layout2>
    __attribute__((always_inline))
    static inline auto
    evaluate( lvalue<VTr,I,void,Enc,NT,Layout1> l, rvalue<VTr,Layout2,void> r ) {
	l.value().ntstore( r.value() );
	return r;
    }

    template<typename VTr, typename I, typename MTr, typename Enc, bool NT,
	     layout_t Layout1,  layout_t Layout2>
    __attribute__((always_inline))
    static inline auto
    evaluate( lvalue<VTr,I,void,Enc,NT,Layout1> l, rvalue<void,Layout2,MTr> r,
	      typename std::enable_if<simd::matchVLtt<VTr,MTr>::value
	      && simd::detail::is_mask_traits<VTr>::value
	      && VTr::W != MTr::W>::type * = nullptr ) {
	l.value().ntstore( r.mask().template convert<VTr>() );
	return r;
    }

    template<typename VTr, typename I, typename Enc, bool NT,
	     layout_t Layout1,  layout_t Layout2>
    __attribute__((always_inline))
    static inline auto
    evaluate( lvalue<VTr,I,void,Enc,NT,Layout1> l, rvalue<void,Layout2,VTr> r ) {
	l.value().ntstore( r.mask() );
	return r;
    }

private:
    R m_ref;
    T m_val;
};

template<typename R, typename T>
auto make_ntstoreop( R r, T t ) {
    return storeop<true,R,T>( r, t );
}

template<bool nt, typename R, typename T>
auto make_storeop_like( R r, T t ) {
    return storeop<nt,R,T>( r, t );
}

/**=====================================================================*
 * Syntactic sugar for scalars
 *======================================================================*/
template<unsigned short AID, typename Tr>
static constexpr auto make_scalar() {
    return
	array_intl<typename Tr::element_type, VID, AID,
		    array_encoding<void>, false>()
	[value<simd::ty<VID,Tr::VL>, vk_zero>()];
}

template<unsigned short AID, typename T, unsigned short VL>
static constexpr auto make_scalar() {
    return make_scalar<AID,simd::ty<T,VL>>();
}

} // namespace expr

#endif // GRAPTOR_DSL_AST_MEMREF_H
