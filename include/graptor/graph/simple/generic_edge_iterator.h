// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_SIMPLE_GENERIC_EDGE_ITERATOR_H
#define GRAPHGRIND_GRAPH_SIMPLE_GENERIC_EDGE_ITERATOR_H

namespace graptor {

namespace graph {

template<typename lVID, typename lEID>
class generic_edge_iterator : public std::iterator<
    std::input_iterator_tag,	// iterator_category
    std::pair<lVID,lVID>, 	// value_type
    lEID,			// difference_type
    const std::pair<lVID,lVID> *,	// pointer
    std::pair<lVID,lVID> 		// reference
    > {
public:
    using VID = lVID;
    using EID = lEID;

public:
    explicit generic_edge_iterator(
	VID vpos, EID epos,
	const EID * const index, const VID * const edges )
	: m_vpos( vpos ), m_epos( epos ),
	  m_index( index ), m_edges( edges ) { }
    generic_edge_iterator& operator++() {
	++m_epos;
	if( m_epos == m_index[m_vpos+1] )
	    ++m_vpos;
	return *this;
    }
    generic_edge_iterator operator++( int ) {
	generic_edge_iterator retval = *this; ++(*this); return retval;
    }
    bool operator == ( generic_edge_iterator other ) const {
	return m_epos == other.m_epos;
    }
    bool operator != ( generic_edge_iterator other ) const {
	return !( *this == other );
    }
    typename generic_edge_iterator::reference operator*() const {
	return std::make_pair( m_vpos, m_edges[m_epos] );
    }

private:
    VID m_vpos;
    EID m_epos;
    const EID * const m_index;
    const VID * const m_edges;
};

} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_GENERIC_EDGE_ITERATOR_H

