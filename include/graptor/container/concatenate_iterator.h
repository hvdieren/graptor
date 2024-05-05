// -*- C++ -*-
#ifndef GRAPTOR_CONTAINER_CONCATENATE_ITERATOR_H
#define GRAPTOR_CONTAINER_CONCATENATE_ITERATOR_H

namespace graptor {

// Assumes std::iterator_traits<Iterator?>::iterator_category is compatible
// with input_iterator_tag
template<typename Iterator1, typename Iterator2>
class concatenate_iterator {
public:
    using type = typename Iterator1::value_type;
    static_assert( std::is_same_v<type,typename Iterator2::value_type>,
		   "Iterators must refer to the same types" );

    // iterator traits
    using iterator_category = std::input_iterator_tag;
    using value_type = typename Iterator1::value_type;
    using difference_type = typename Iterator1::difference_type;
    using pointer = typename Iterator1::pointer;
    using reference = typename Iterator1::reference;

public:
    explicit concatenate_iterator( Iterator1 b, Iterator1 e, Iterator2 i )
	: m_it1_begin( b ), m_it1_end( e ), m_it2( i ) { }
    concatenate_iterator& operator++() {
	if( m_it1_begin == m_it1_end )
	    m_it2++;
	else
	    m_it1_begin++;
	return *this;
    }
    concatenate_iterator operator++( int ) {
	concatenate_iterator retval = *this; ++(*this); return retval;
    }
    bool operator == ( concatenate_iterator other ) const {
	return m_it1_begin == other.m_it1_begin
	    && m_it2 == m_it2
	    && m_it1_end == other.m_it1_end;
    }
    bool operator != ( concatenate_iterator other ) const {
	return !( *this == other );
    }
    typename concatenate_iterator::reference operator*() const {
	return m_it1_begin == m_it1_end ? *m_it2 : *m_it1_begin;
    }

private:
    Iterator1 m_it1_begin;
    Iterator1 m_it1_end;
    Iterator2 m_it2;
};

template<typename Iterator1, typename Iterator2>
auto make_concatenate_iterator(
    Iterator1 && b, Iterator1 && e, Iterator2 && i ) {
    return concatenate_iterator<
	std::decay_t<Iterator1>,std::decay_t<Iterator2>>(
	    std::forward<Iterator1>( b ),
	    std::forward<Iterator1>( e ),
	    std::forward<Iterator2>( i ) );
}

} // namespace graptor

#endif // GRAPTOR_CONTAINER_CONCATENATE_ITERATOR_H

