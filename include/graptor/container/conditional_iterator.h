// -*- C++ -*-
#ifndef GRAPTOR_CONTAINER_CONDITIONAL_ITERATOR_H
#define GRAPTOR_CONTAINER_CONDITIONAL_ITERATOR_H

namespace graptor {

// Assumes std::iterator_traits<Iterator>::iterator_category is compatible
// with input_iterator_tag
template<typename Iterator, typename Cond>
class conditional_iterator {
public:
    using type = typename Iterator::value_type;

    // iterator traits
    using iterator_category = std::input_iterator_tag;
    using value_type = typename Iterator::value_type;
    using difference_type = typename Iterator::difference_type;
    using pointer = typename Iterator::pointer;
    using reference = typename Iterator::reference;

public:
    explicit conditional_iterator( Iterator it, Cond cond )
	: m_it( it ), m_cond( cond ) { }
    conditional_iterator& operator++() {
	do {
	    m_it++;
	} while( !m_cond( m_it ) );
	return *this;
    }
    conditional_iterator operator++( int ) {
	conditional_iterator retval = *this; ++(*this); return retval;
    }
    bool operator == ( conditional_iterator other ) const {
	return m_it == other.m_it;
    }
    bool operator != ( conditional_iterator other ) const {
	return !( *this == other );
    }
    typename conditional_iterator::reference operator*() const {
	return *m_it;
    }

private:
    Iterator m_it;
    Cond m_cond;
};

template<typename Iterator, typename Cond>
auto make_conditional_iterator( Iterator && it, Cond && cond ) {
    return conditional_iterator<Iterator,Cond>(
	std::forward<Iterator>( it ),
	std::forward<Cond>( cond ) );
}

} // namespace graptor

#endif // GRAPTOR_CONTAINER_CONDITIONAL_ITERATOR_H

