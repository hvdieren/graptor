// -*- c++ -*-
#ifndef GRAPTOR_CONTAINER_MAYBE_DUAL_SET_H
#define GRAPTOR_CONTAINER_MAYBE_DUAL_SET_H

/*!**********************************************************************
 * \file maybe_dual_set.h
 * \brief Defines abstract types that model sets of elements in a dual
 *        representation consisting of (potentially) a sequential list
 *        (indicated by iterators) and (potentially) a hashed data structure.
 *        At least one of these two representations must be present.
 ************************************************************************/

namespace graptor {

/*! \brief A dual representation of a set as a sequential collection Seq
 *         and a hash set Hash, with one possibly absent.
 *
 * The sequence is copied (by value) as it is small and assumed not to
 * own its content (e.g., \p array_slice). The hash set is assumed to
 * own its content, so a pointer is kept.
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
struct maybe_dual_set {
    using self_type = maybe_dual_set<Seq,Hash>;		//!< our own type
    using seq_type = Seq; 	 	 	 	//!< sequential set type
    using hash_type = Hash;	   			//!< hashed set type
    using type = 	 	 	 	 	//!< element type
	std::remove_cv_t<typename std::decay_t<seq_type>::type>;

    /*! Constructor taking a sequential and a hashed set
     *
     * The two sets should contain the same elements.
     * \param seq Sequential set representation
     * \param hash Hashed set representation
     */
    maybe_dual_set( seq_type seq, const hash_type & hash )
	: m_seq( seq ), m_hash( &hash ) { }

    /*! Constructor taking a sequential set
     *
     * \param seq Sequential set representation
     */
    maybe_dual_set( seq_type seq )
	: m_seq( seq ), m_hash( nullptr ) { }

    /*! Constructor taking a hashed set
     *
     * \param hash Hashed set representation
     */
    maybe_dual_set( const hash_type & hash )
	: m_seq( Seq() ), m_hash( &hash ) { }

    //! Returns the size of the set, i.e., a count of elements.
    // Must return the size of the sequential representation if available,
    // in order to conform with trimming operations.
    size_t size() const {
	return m_seq.size() > 0 || m_hash == nullptr
	    ? m_seq.size() : m_hash->size();
    }
    
    /*! Returns the range of the set, i.e., the difference between largest
     * and smallest.
     * Can only be performed on the sequential representation and may thus
     * return 0 if the sequential representation is absent.
     */
    auto range() const { return m_seq.range(); }
    
    //! Returns an iterator to the beginning of the sequential representation
    auto begin() { return m_seq.begin(); }
    /*! Returns a constant iterator to the beginning of the sequential
     *  representation */
    const auto begin() const { return m_seq.begin(); }
    //! Returns an iterator to the end of the sequential representation
    auto end() { return m_seq.end(); }
    //! Returns a constant iterator to the end of the sequential representation
    const auto end() const { return m_seq.end(); }

    //! Returns first element. Requires non-empty set.
    auto front() const { return m_seq.front(); }
    //! Returns last element. Requires non-empty set.
    auto back() const { return m_seq.back(); }

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
    bool contains( type value ) const { return m_hash->contains( value ); }

    /*! Performs a lookup on a hash table.
     *
     * Requires is_hash_table_v<hash_type>.
     *
     * Where the hashed set is also a hash table (the set contains key-value
     * pairs with unique keys), return the value component.
     * \param value The value checked for presence
     * \return The corresponding value stored by the hash table.
     */
    auto lookup( type value ) const { return m_hash->lookup( value ); }

    /*! Checks if a number of elements are stored in the hash set.
     *
     * Requires is_multi_hash_set_v<hash_type>.
     *
     * \tparam U The type of the elements, normally the same as type.
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
	return m_hash->template multi_contains<U,VL,MT>( index, mt );
    }

    /*! Performs simultaneous lookup of a number of elements in the hash table.
     *
     * Requires is_multi_hash_table_v<hash_type>.
     *
     * \tparam U The type of the elements, normally the same as type.
     * \tparam VL The vector length, or number of elements probed.
     *
     * \param index Elements looked up
     * \return A vector holding the corresponding values.
     */
 
    template<typename U, unsigned short VL>
    auto
    multi_lookup( typename vector_type_traits_vl<U,VL>::type index ) const {
	return m_hash->template multi_lookup<U,VL>( index );
    }

    //! Trim the sequential representation from the left-hand side.
    self_type trim_l( const type * l ) const {
	return self_type( m_seq.trim_l( l ), *m_hash );
    }

    //! Trim the sequential representation from the right-hand side.
    self_type trim_r( const type * r ) const {
	return self_type( m_seq.trim_r( r ), *m_hash );
    }

    //! Trim the sequential representation from the left- and right-hand sides.
    self_type trim_range( type lo, type hi ) const {
	return self_type( m_seq.trim_range( lo, hi ), *m_hash );
    }

    //! Trim the sequential representation from the left-hand side.
    self_type trim_front( type lo ) const {
	return self_type( m_seq.trim_front( lo ), *m_hash );
    }

    //! Trim the sequential representation from the left- and right-hand sides.
    self_type trim_back( type hi ) const {
	return self_type( m_seq.trim_back( hi ), *m_hash );
    }

    //! Is the sequential representation valid?
    bool has_sequential() const {
	return m_seq.size() > 0 || m_hash == nullptr || m_hash->size() == 0;
    }

    //! Is the hash set representation valid
    bool has_hash_set() const {
	return m_hash != nullptr;
    }

private:
    seq_type m_seq; 	 	//!< Sequential representation
    const hash_type * m_hash;	//!< Hashed representation
};

} // namespace graptor

#endif // GRAPTOR_CONTAINER_MAYBE_DUAL_SET_H
