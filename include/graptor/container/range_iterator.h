// -*- C++ -*-
#ifndef GRAPTOR_CONTAINER_RANGE_ITERATOR_H
#define GRAPTOR_CONTAINER_RANGE_ITERATOR_H

namespace graptor {

template<typename T>
class range_iterator {
public:
    using type = T;
    // iterator traits
    using iterator_category = std::input_iterator_tag;
    using value_type = type;
    using difference_type = type;
    using pointer = const type*;
    using reference = type;

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

