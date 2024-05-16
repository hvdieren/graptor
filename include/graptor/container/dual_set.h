// -*- c++ -*-
#ifndef GRAPTOR_CONTAINER_DUAL_SET_H
#define GRAPTOR_CONTAINER_DUAL_SET_H

/*!**********************************************************************
 * \file dual_set.h
 * \brief Defines abstract types that model sets of elements in a dual
 *        representation consisting of a sequential list (indicated by
 *        iterators) and a hashed data structure.
 ************************************************************************/

namespace graptor {

/*! \brief Type traits that identifies if a data type is a hash set, i.e.,
 *         it has a method \c contains.
 * \tparam S The type for which it is determined if it is a hash set.
 */
template<typename S>
struct is_hash_set {
    static constexpr bool value = requires( const S & s ) {
	s.contains( typename S::type(0) );
    };
};

/*! \brief Variable that identifies if a data type is a hash set, i.e.,
 *         it has a method \c contains.
 * \tparam S The type for which it is determined if it is a hash set.
 */
template<typename S>
constexpr bool is_hash_set_v = is_hash_set<S>::value;

/*! \brief Type traits that identifies if a data type is a hash table, i.e.,
 *         it has a method \c lookup.
 * \tparam S The type for which it is determined if it is a hash table.
 */
template<typename S>
struct is_hash_table {
    static constexpr bool value = requires( const S & s ) {
	s.lookup( typename S::type(0) );
    };
};

/*! \brief Variable that identifies if a data type is a hash table, i.e.,
 *         it has a method \c lookup.
 * \tparam S The type for which it is determined if it is a hash table.
 */
template<typename S>
constexpr bool is_hash_table_v = is_hash_table<S>::value;

/*! \brief Type traits that identifies if a data type is a hash set supporting
 *         simultaneous lookup of multiple elements using vectorization, i.e.,
 *         it has a method \c multi_contains.
 * \tparam S The type for which it is determined if it is a multi-hash set.
 */
template<typename S>
template<typename S>
struct is_multi_hash_set {
    static constexpr bool value = requires( const S & s ) {
	s.template multi_contains<typename S::type,8,target::mt_vmask>(
	    vector_type_traits_vl<typename S::type,8>::setzero(),
	    target::mt_vmask() );
    };
};

/*! \brief Variable that identifies if a data type is a hash set supporting
 *         simultaneous lookup of multiple elements using vectorization, i.e.,
 *         it has a method \c multi_contains.
 * \tparam S The type for which it is determined if it is a multi-hash set.
 */
template<typename S>
constexpr bool is_multi_hash_set_v = is_multi_hash_set<S>::value;

/*! \brief Type traits that identifies if a data type is a hash table supporting
 *         simultaneous lookup of multiple elements using vectorization, i.e.,
 *         it has a method \c multi_lookup.
 * \tparam S The type for which it is determined if it is a multi-hash table.
 */
template<typename S>
struct is_multi_hash_table {
    static constexpr bool value = requires( const S & s ) {
	s.template multi_lookup<typename S::type,8>(
	    vector_type_traits_vl<typename S::type,8>::setzero() );
    };
};

/*! \brief Variable that identifies if a data type is a hash table supporting
 *         simultaneous lookup of multiple elements using vectorization, i.e.,
 *         it has a method \c multi_lookup.
 * \tparam S The type for which it is determined if it is a multi-hash table.
 */
template<typename S>
constexpr bool is_multi_hash_table_v = is_multi_hash_table<S>::value;

/*! \brief A dual representation of a set as a sequential collection Seq
 *         and a hash set Hash.
 *
 * The sequence is copied (by value) as it is small and assumed not to
 * own its content (e.g., \p array_slice). The hash set is assumed to
 * own its content, so a reference is kept.
 *
 * If the sequence should not be copied, it is possible to instantiate this
 * type such that the type \p Seq is a reference type.
 *
 * The hashed representation may also be a hash table, which is considered
 * as a set of key-value pairs, with unique keys. In this case, the sequential
 * representation may only store the keys and not values.
 *
 * It is possible also to store a hashed representation as a sequential type,
 * which may be useful when this representation provides constant-time
 * iterators.
 *
 * \tparam Seq Sequential set representation
 * \tparam Hash Hash-based set representation
 *
 * The \a Seq type must have a member-type called \c type that identifies the
 * type of the values held.
 *
 * \sa make_dual_set
 */
template<typename Seq, typename Hash>
struct dual_set {
    using seq_type = Seq; 	 	 	 	//!< sequential set type
    using hash_type = Hash;	   			//!< hashed set type
    using type = std::remove_cv_t<typename seq_type::type>; //!< element type

    /*! Constructor taking a sequential and a hashed set
     *
     * The two sets should contain the same elements.
     * \param seq Sequential set representation
     * \param hash Hashed set representation
     */
    dual_set( seq_type seq, const hash_type & hash )
	: m_seq( seq ), m_hash( hash ) { }

    //! Returns the size of the set, i.e., a count of elements.
    auto size() const { return m_seq.size(); }
    
    //! Returns an iterator to the beginning of the sequential representation
    auto begin() { return m_seq.begin(); }
    //! Returns a constant iterator to the beginning of the sequential
    //  representation
    const auto begin() const { return m_seq.begin(); }
    //! Returns an iterator to the end of the sequential representation
    auto end() { return m_seq.end(); }
    //! Returns a constant iterator to the end of the sequential representation
    const auto end() const { return m_seq.end(); }

    //! Returns a reference to the sequential representation
    seq_type get_seq() const { return m_seq; }

    //! Returns a reference to the hashed representation
    const hash_type & get_hash() const { return m_hash; }

    /*! Checks if set contains an element.
     *
     * Requires is_hash_set_v<hash_type>.
     *
     * The hashed representation is used as this should be more efficient.
     * \param value The value checked for presence
     * \return True if the value is a member of the set; false otherwise.
     */
    bool contains( type value ) const { return m_hash.contains( value ); }

    /*! Performs a lookup on a hash table.
     *
     * Requires is_hash_table_v<hash_type>.
     *
     * Where the hashed set is also a hash table (the set contains key-value
     * pairs with unique keys), return the value component.
     * \param value The value checked for presence
     * \return The corresponding value stored by the hash table.
     */
    auto lookup( type value ) const { return m_hash.lookup( value ); }

    /*! Checks if a number of elements are stored in the hash set.
     *
     * Requires is_multi_hash_set_v<hash_type>.
     *
     * \tparam U The type of the elements, normally the same as \p type.
     * \tparam VL The vector length, or number of elements probed.
     * \tparam MT Determines the type of the return value.
     *
     * \param index Elements checked for
     * \param mt A placeholder value whose type determines the kind of mask
     *           to return.
     * \return A mask of kind \a MT, where \a MT is a \se target::mkind.
     */
    template<typename U, unsigned short VL, typename MT>
    auto
    multi_contains( typename vector_type_traits_vl<U,VL>::type index,
		    MT mt ) const {
	return m_hash.template multi_contains<U,VL,MT>( index, mt );
    }

    /*! Performs simultaneous lookup of a number of elements in the hash table.
     *
     * Requires is_multi_hash_table_v<hash_type>.
     *
     * \tparam U The type of the elements, normally the same as \p type.
     * \tparam VL The vector length, or number of elements probed.
     *
     * \param index Elements looked up
     * \return A vector holding the corresponding values.
     */
 
    template<typename U, unsigned short VL>
    auto
    multi_lookup( typename vector_type_traits_vl<U,VL>::type index ) const {
	return m_hash.template multi_lookup<U,VL>( index );
    }

    //! Trim the sequential representation from the right-hand side.
    dual_set<seq_type,hash_type> trim_r( const type * r ) const {
	return dual_set<seq_type,hash_type>( m_seq.trim_r( r ), m_hash );
    }

    //! Trim the sequential representation from the left- and right-hand sides.
    dual_set<seq_type,hash_type> trim_range( type lo, type hi ) const {
	return dual_set<seq_type,hash_type>( m_seq.trim_range( lo, hi ),
					     m_hash );
    }

private:
    seq_type m_seq; 	 	//!< Sequential representation
    const hash_type & m_hash;	//!< Hashed representation
};

/*! \brief Short-hand construction method for a \p dual_set
 * \sa dual_set
 */
template<typename Seq, typename Hash>
auto make_dual_set( Seq && seq, const Hash & hash ) {
    return dual_set<Seq,Hash>( seq, hash );
}

} // namespace graptor

#endif // GRAPTOR_CONTAINER_DUAL_SET_H
