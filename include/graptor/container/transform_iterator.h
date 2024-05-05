// -*- C++ -*-
#ifndef GRAPTOR_CONTAINER_TRANSFORM_ITERATOR_H
#define GRAPTOR_CONTAINER_TRANSFORM_ITERATOR_H

#include <iterator>

namespace graptor {

// Assumes std::iterator_traits<Iterator>::iterator_category is compatible
// with input_iterator_tag
template<typename Iterator, typename Transform>
class transform_iterator {
public:
    // iterator traits
    using iterator_category = std::input_iterator_tag;
    using value_type = std::iter_value_t<Iterator>;
    using difference_type = std::iter_difference_t<Iterator>;
    using pointer = std::iter_value_t<Iterator>;
    using reference = std::iter_common_reference_t<Iterator>;

public:
    explicit transform_iterator( Iterator it, Transform transform )
	: m_it( it ), m_transform( transform ) { }
    transform_iterator& operator++() { m_it++; return *this; }
    transform_iterator operator++( int ) {
	transform_iterator retval = *this; ++(*this); return retval;
    }
    bool operator == ( transform_iterator other ) const {
	return m_it == other.m_it;
    }
    bool operator != ( transform_iterator other ) const {
	return !( *this == other );
    }
    value_type operator*() const {
	return m_transform( *m_it );
    }

private:
    Iterator m_it;
    Transform m_transform;
};

template<typename Iterator, typename Transform>
auto make_transform_iterator( Iterator && it, Transform && transform ) {
    return transform_iterator<std::decay_t<Iterator>,std::decay_t<Transform>>(
	std::forward<Iterator>( it ),
	std::forward<Transform>( transform ) );
}

} // namespace graptor

#endif // GRAPTOR_CONTAINER_TRANSFORM_ITERATOR_H

