// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_SIMPLE_CONDITIONAL_ITERATOR_H
#define GRAPHGRIND_GRAPH_SIMPLE_CONDITIONAL_ITERATOR_H

namespace graptor {

namespace graph {

template<typename Iterator, typename Cond>
class conditional_iterator : public std::iterator<
    std::input_iterator_tag,		// iterator_category
    typename Iterator::value_type,	// value_type
    typename Iterator::difference_type,	// difference_type
    typename Iterator::pointer,		// pointer
    typename Iterator::reference	// reference
    > {
public:
    using type = typename Iterator::value_type;

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
	return m_it;
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

} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_CONDITIONAL_ITERATOR_H

