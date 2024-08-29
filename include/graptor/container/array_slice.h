// -*- c++ -*-
#ifndef GRAPTOR_CONTAINER_ARRAY_SLICE_H
#define GRAPTOR_CONTAINER_ARRAY_SLICE_H

namespace graptor {

// TODO: this is pretty much like a C++ std::ranges::view, adopt that
// functionality
template<typename T, typename I = size_t>
struct array_slice {
    using type = T;
    using index_type = I;
    
    array_slice( const type * begin_, const type * end_ )
	: m_begin( begin_ ), m_end( end_ ) { }
    array_slice( const type * begin_, index_type len_ )
	: m_begin( begin_ ), m_end( std::next( begin_, len_ ) ) { }

    size_t size() const { return std::distance( m_begin, m_end ); }
    
    /*! Returns the range of the set, i.e., the difference between largest
     * and smallest. Assumes sequence is sorted.
     */
    type range() const {
	return m_begin != m_end ? *std::prev( m_end ) - *m_begin: 0;
    }
    
    const type * begin() const { return m_begin; }
    const type * end() const { return m_end; }

    // Assumes non-empty list
    type front() const { return *begin(); }
    type back() const { return *std::prev( end() ); }

    type at( size_t idx ) const { return *std::next( m_begin, idx ); }

    array_slice<type,index_type> trim_r( const type * r ) const {
	return array_slice<type,index_type>( m_begin, r );
    }
    array_slice<type,index_type> trim_l( const type * l ) const {
	return array_slice<type,index_type>( l, m_end );
    }

    array_slice<type,index_type> trim_range( type lo, type hi ) const {
	const type * b = begin();
	const type * e = end();
	if( lo > front() )
	    b = std::lower_bound( begin(), end(), lo );
	if( hi < back() )
	    e = std::upper_bound( b, end(), hi );
	return array_slice<type,index_type>( b, e );
    }

    // Assumes non-empty list; retain lo in list but nothing smaller
    array_slice<type,index_type> trim_front( type lo ) const {
	if( lo < front() )
	    return *this;
	else
	    return array_slice<type,index_type>(
		std::lower_bound( begin(), end(), lo ), m_end );
    }

    // Assumes non-empty list; retain hi in list but nothing larger
    array_slice<type,index_type> trim_back( type hi ) const {
	if( hi > back() )
	    return *this;
	else
	    return array_slice<type,index_type>(
		m_begin, std::upper_bound( begin(), end(), hi ) );
    }

private:
    const type * m_begin, * m_end;
};

template<typename T, typename I = size_t>
array_slice<T,I> make_array_slice( const T * b, const T * e ) {
    return array_slice<T,I>( b, e );
}

template<typename T, typename I = size_t>
array_slice<T,I> make_array_slice( const std::vector<T> & v ) {
    return array_slice<T,I>( &*v.begin(), &*v.end() );
}

} // namespace graptor

#endif // GRAPTOR_CONTAINER_ARRAY_SLICE_H
