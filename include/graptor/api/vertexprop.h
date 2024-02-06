// -*- C++ -*-
#ifndef GRAPTOR_API_VERTEXPROP_H
#define GRAPTOR_API_VERTEXPROP_H

#include "graptor/utils.h"
#include "graptor/encoding.h"
#include "graptor/primitives.h"

namespace api {

/************************************************************************
 * Representation of a vertex property. Can be used both as a syntax tree
 * element and as a native C++ array.
 *
 * Memory is allocated at construction time and has to be freed explicitly
 * by calling #del.
 * Memory allocation proceeds using a balanced partitioned allocation.
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

    /** Constructor: create a vertex property.
     *
     * @param[in] part graph partitioner
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

    void set( VID v, T t ) {
	encoding::template store<simd::ty<T,1>>( mem.get(), v, t );
    }

    // Should relay on encoding for get/set
    [[deprecated("not reliable with bitfields")]]
    typename encoding::storage_type * get_ptr() const {
	return mem.get();
    }

    void fill( const partitioner & part, T && val ) {
	if constexpr ( !std::is_same_v<encoding,array_encoding<T>> )
	    assert( 0 && "NYI - encoding" );
	fill_by_partition<T>( part, mem.get(), std::forward<T>( val ) );
    }

    const char * get_name() const { return m_name; }

private:
    mm::buffer<typename encoding::storage_type> mem;	//!< memory buffer
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
 * Interleaved representation of two vertex properties.
 *
 * Memory is allocated at construction time and has to be freed explicitly
 * by calling #del.
 * Memory allocation proceeds using a balanced partitioned allocation.
 * The array size as well as the dimensions of partitioning are taken
 * from the partitioner object supplied to the constructor.
 *
 * @see mm
 * @param <T0> the first type of array elements
 * @param <T1> the second type of array elements
 * @param <U> the type of the array index
 * @param <AID0_> an integral identifier that uniquely identifies the
 *                first property array
 * @param <AID1_> an integral identifier that uniquely identifies the
 *                second property array
 * @param <MaxVL_> maximum vector length - interleaving factor
 ************************************************************************/
#if 0
template<typename T0, typename T1, typename U,
	 short AID0_, short AID1_,
	 unsigned short MaxVL = MAX_VL>
class vertexprop2 : private NonCopyable<vertexprop2<T0,T1,U,AID0_,AID1_,MaxVL>> {
public:
    using type0 = T0; 	 	 	//!< first type of array elements
    using type1 = T1; 	 	 	//!< second type of array elements
    using index_type = U; 	 	//!< index type of property
    using encoding = array_encoding_intlv2<T0,T1,MaxVL>; //!< data encoding
    using encoding0 = typename encoding::ifc_encoding<0>;
    using encoding1 = typename encoding::ifc_encoding<1>;
    static constexpr short AID0 = AID0_; //!< array ID of first property
    static constexpr short AID1 = AID1_; //!< array ID of second property

    /** Constructor: create an vertex property.
     *
     * @param[in] part graph partitioner
     * @param[in] name explanation string for debugging
     */
    vertexprop2( const partitioner & part, const char * name )
	: mem( part.scale( 2 ), name ),
	  m_name( name ) { }

    //! Release memory
    void del() {
	mem.del( m_name );
    }

    /*! Obtaining proxy vertex property
     *
     * @tparam A property index
     * @return A vertexprop object representing the property
     */
    template<unsigned short A>
    auto get_property() {
	return vertexprop<A, std::conditional_t<A==0,type0,type1>, index_type,
			  A == 0 ? AID0 : AID1,
			  typename encoding::ifc_encoding<A>,
			  MaxVL>( encoding::get_base<A>( mem.get() ) );
	// TODO: avoid add of MaxVL to index for A == 1 by taking different base poiter
    }
    
    const char * get_name() const { return m_name; }

private:
    mm::buffer<typename encoding::stored_type> mem;	//!< memory buffer
    const char * m_name;	//!< explanatory name describing vertex property
};
#endif

/************************************************************************
 * Representation of an interleaved vertex property. Can be used both as
 * a syntax tree element and as a native C++ array.
 *
 * Memory is not held by this object. #del has no impact.
 *
 * @see mm
 * @param <A> the accessible property from the interleaved set
 * @param <T0> the first type of array elements
 * @param <T1> the second type of array elements
 * @param <U> the type of the array index
 * @param <AID_> an integral identifier that uniquely identifies the
 *               property array
 * @param <MaxVL_> maximum vector length - interleaving factor
 ************************************************************************/
#if 0
template<unsigned short A, typename T0, typename T1, typename U,
	 short AID_, unsigned short MaxVL_>
class vertexprop<std::conditional_t<A==0,T0,T1>,U,AID_,
    array_encoding_intlv2_ifc<A,T0,T1,MaxVL_>>
    : private NonCopyable<vertexprop<
	std::conditional_t<A==0,T0,T1>,U,AID_,
	array_encoding_intlv2_ifc<A,T0,T1,MaxVL_>> {
public:
    using type = std::conditional_t<A==0,T0,T1>; //!< type of array elements
    using index_type = U; 	 	//!< index type of property
    using encoding = array_encoding_intlv2_ifc<A,T0,T1,MaxVL_>; //!< data encoding
    static constexpr short AID = AID_;	//!< array ID of property
    //! type of array syntax tree for this property
    using array_ty = expr::array_ro<type,index_type,AID,encoding,false>;

    /** Constructor: create an vertex property.
     *
     * @param[in] ptr base pointer of array
     * @param[in] name explanation string for debugging
     */
    vertexprop( typename encoding::storage_type * ptr, const char * name )
	: m_ptr( ptr ), m_name( name ) { }

    //! Release memory - no-op
    void del() { }

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
	return array_ty( m_ptr )[e];
    }

    /*! Subscript operator overload for native C++ array operation.
     * Indexes the array and returns the value at index #v of the array.
     * This operator returns an r-value; it cannot be used to modify array contents.
     *
     * @param[in] v array index
     * @return value found at array index #e
     */
    T operator[] ( VID v ) const {
	return encoding::template load<simd::ty<T,1>>( m_ptr, v );
    }

    typename encoding::stored_type * get_ptr() const {
	return m_ptr;
    }

    const char * get_name() const { return m_name; }

private:
    typename encoding::storage_type * m_ptr; //!< memory buffer, not owned
    const char * m_name;	//!< explanatory name describing vertex property
};
#endif

} // namespace api

#endif // GRAPTOR_API_VERTEXPROP_H
