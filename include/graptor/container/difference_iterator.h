// -*- C++ -*-
#ifndef GRAPTOR_CONTAINER_UTILS_H
#define GRAPTOR_CONTAINER_UTILS_H

namespace graptor {

template<typename Iter>
class difference_iterator : public std::iterator<
    std::input_iterator_tag,	// iterator_category
    typename std::iterator_traits<Iter>::value_type,
    typename std::iterator_traits<Iter>::difference_type,
    typename std::iterator_traits<Iter>::pointer,
    typename std::iterator_traits<Iter>::reference
    > {
public:
    explicit difference_iterator( Iter it ) : m_it( it ) { }
    difference_iterator& operator++() { m_it++; return *this; }
    difference_iterator operator++( int ) {
	difference_iterator retval = *this; ++(*this); return retval;
    }
    bool operator == ( difference_iterator other ) const {
	return m_it == other.m_it;
    }
    bool operator != ( difference_iterator other ) const {
	return !( *this == other );
    }
    typename difference_iterator::value_type operator*() const {
	Iter m_next = std::next( m_it, 1 );
	return *m_next - *m_it;
    }

private:
    Iter m_it;
};

template<typename Iter>
class pairwise_difference_iterator : public std::iterator<
    std::input_iterator_tag,	// iterator_category
    typename std::iterator_traits<Iter>::value_type,
    typename std::iterator_traits<Iter>::difference_type,
    typename std::iterator_traits<Iter>::pointer,
    typename std::iterator_traits<Iter>::reference
    > {
public:
    explicit pairwise_difference_iterator( Iter it1, Iter it2 )
	: m_it1( it1 ), m_it2( it2 ) { }
    pairwise_difference_iterator& operator++() {
	m_it1++; m_it2++; return *this;
    }
    pairwise_difference_iterator operator++( int ) {
	pairwise_difference_iterator retval = *this; ++(*this); return retval;
    }
    bool operator == ( pairwise_difference_iterator other ) const {
	return m_it1 == other.m_it1 && m_it2 == other.m_it2;
    }
    bool operator != ( pairwise_difference_iterator other ) const {
	return !( *this == other );
    }
    typename pairwise_difference_iterator::value_type operator*() const {
	return *m_it2 - *m_it1;
    }

    typename pairwise_difference_iterator::difference_type operator - (
	pairwise_difference_iterator other ) const {
	return m_it1 - other.m_it1;
    }

private:
    Iter m_it1;
    Iter m_it2;
};

template<typename Iter>
auto find_maximum( Iter start, Iter end ) {
    return std::max_element( start, end );
}

} // namespace graptor

#endif // GRAPTOR_CONTAINER_UTILS_H

