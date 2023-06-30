// -*- C++ -*-
#ifndef GRAPTOR_CONTAINER_RANGE_ITERATOR_H
#define GRAPTOR_CONTAINER_RANGE_ITERATOR_H

namespace graptor {

template<typename T>
class range_iterator : public std::iterator<
    std::input_iterator_tag,	// iterator_category
    T,	  			// value_type
    T,				// difference_type
    const T*, 	 		// pointer
    T 	 	 		// reference
    > {
public:
    using type = T;

public:
    explicit range_iterator( type pos = 0 ) : m_pos( pos ) { }
    range_iterator& operator++() { m_pos++; return *this; }
    range_iterator operator++( int ) {
	range_iterator retval = *this; ++(*this); return retval;
    }
    bool operator == ( range_iterator other ) const {
	return m_pos == other.m_pos;
    }
    bool operator != ( range_iterator other ) const {
	return !( *this == other );
    }
    typename range_iterator::reference operator*() const {
	return m_pos;
    }

private:
    type m_pos;
};

} // namespace graph

#endif // GRAPTOR_CONTAINER_RANGE_ITERATOR_H

