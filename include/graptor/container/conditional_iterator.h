// -*- C++ -*-
#ifndef GRAPTOR_CONTAINER_CONDITIONAL_ITERATOR_H
#define GRAPTOR_CONTAINER_CONDITIONAL_ITERATOR_H

#include<iterator>

namespace graptor {

// Assumes std::iterator_traits<Iterator>::iterator_category is compatible
// with input_iterator_tag
template<typename Iterator, typename Cond>
class conditional_iterator {
public:
    // iterator traits
    using iterator_category = std::input_iterator_tag;
    using value_type = std::iter_value_t<Iterator>;
    using difference_type = std::iter_difference_t<Iterator>;
    using pointer = std::iter_value_t<Iterator>;
    using reference = std::iter_common_reference_t<Iterator>;

public:
    explicit conditional_iterator( Iterator it, Iterator it_end, Cond cond )
	: m_it( it ), m_it_end( it_end ), m_cond( cond ) {
	while( m_it != m_it_end && !m_cond( *m_it ) )
	    ++m_it;
    }
    conditional_iterator& operator++() {
	do {
	    ++m_it;
	} while( m_it != m_it_end && !m_cond( *m_it ) );
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
    Iterator m_it_end;
    Cond m_cond;
};

template<typename Iterator, typename Cond>
auto make_conditional_iterator( Iterator && it, Iterator && end,
				Cond && cond ) {
    return conditional_iterator<Iterator,Cond>(
	std::forward<Iterator>( it ),
	std::forward<Iterator>( end ),
	std::forward<Cond>( cond ) );
}

} // namespace graptor

#endif // GRAPTOR_CONTAINER_CONDITIONAL_ITERATOR_H

