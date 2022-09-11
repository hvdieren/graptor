// -*- C++ -*-
#ifndef GRAPTOR_API_EDGEPROP_H
#define GRAPTOR_API_EDGEPROP_H

#include "graptor/utils.h"
#include "graptor/encoding.h"

namespace api {

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
	// TODO: use encoding::allocate + adjust to use mm::buffer
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
    mm::buffer<typename encoding::storage_type> mem; //!< memory buffer
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

#endif // GRAPTOR_API_EDGEPROP_H
