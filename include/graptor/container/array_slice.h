// -*- c++ -*-
#ifndef GRAPTOR_CONTAINER_ARRAY_SLICE_H
#define GRAPTOR_CONTAINER_ARRAY_SLICE_H

namespace graptor {

template<typename T, typename I = size_t>
struct array_slice {
    using type = T;
    using index_type = I;

    array_slice( const type * begin_, const type * end_ )
	: m_begin( begin_ ), m_end( end_ ) { }
    array_slice( const type * begin_, index_type len_ )
	: m_begin( begin_ ), m_end( begin_ + len_ ) { }

    index_type size() const { return m_end - m_begin; }
    
    const type * begin() const { return m_begin; }
    const type * end() const { return m_end; }

    array_slice<T,I> trim_r( const type * r ) const {
	return array_slice<T,I>( m_begin, r );
    }

    array_slice<T,I> trim_range( type lo, type hi ) const {
	auto b = std::lower_bound( begin(), end(), lo );
	auto e = std::upper_bound( b, end(), hi );
	return array_slice<T,I>( b, e );
    }

private:
    const type * m_begin, * m_end;
};

template<typename T, typename I = size_t>
array_slice<T,I> make_array_slice( const T * b, const T * e ) {
    return array_slice<T,I>( b, e );
}

} // namespace graptor

#endif // GRAPTOR_CONTAINER_ARRAY_SLICE_H
