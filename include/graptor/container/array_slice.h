// -*- c++ -*-
#ifndef GRAPTOR_CONTAINER_ARRAY_SLICE_H
#define GRAPTOR_CONTAINER_ARRAY_SLICE_H

namespace graptor {

template<typename T, typename I>
struct array_slice {
    using type = T;
    using index_type = I;

    array_slice( type * begin_, type * end_ )
	: m_begin( begin_ ), m_end( end_ ) { }
    array_slice( type * begin_, index_type len_ )
	: m_begin( begin_ ), m_end( begin_ + len_ ) { }

    index_type size() const { return m_end - m_begin; }
    
    type * begin() { return m_begin; }
    const type * begin() const { return m_begin; }
    type * end() { return m_end; }
    const type * end() const { return m_end; }

private:
    T * m_begin, * m_end;
};

} // namespace graptor

#endif // GRAPTOR_CONTAINER_ARRAY_SLICE_H
