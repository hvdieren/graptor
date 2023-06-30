// -*- C++ -*-
#ifndef GRAPTOR_CONTAINER_DOUBLE_INDEX_EDGE_ITERATOR_H
#define GRAPTOR_CONTAINER_DOUBLE_INDEX_EDGE_ITERATOR_H

namespace graptor {

template<typename lVID, typename lEID>
class double_index_edge_iterator : public std::iterator<
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
    explicit double_index_edge_iterator(
	VID vpos, EID epos,
	const EID * const start_index,
	const EID * const end_index,
	const VID * const edges )
	: m_vpos( vpos ), m_epos( epos ),
	  m_start_index( start_index ),
	  m_end_index( end_index ),
	  m_edges( edges ) { }
    double_index_edge_iterator& operator++() {
	++m_epos;
	if( m_epos == m_end_index[m_vpos] )
	    m_epos = m_start_index[++m_vpos];
	return *this;
    }
    double_index_edge_iterator operator++( int ) {
	double_index_edge_iterator retval = *this; ++(*this); return retval;
    }
    bool operator == ( double_index_edge_iterator other ) const {
	return m_epos == other.m_epos;
    }
    bool operator != ( double_index_edge_iterator other ) const {
	return !( *this == other );
    }
    typename double_index_edge_iterator::reference operator*() const {
	return std::make_pair( m_vpos, m_edges[m_epos] );
    }

private:
    VID m_vpos;
    EID m_epos;
    const EID * const m_start_index;
    const EID * const m_end_index;
    const VID * const m_edges;
};

} // namespace graptor

#endif // GRAPTOR_CONTAINER_DOUBLE_INDEX_EDGE_ITERATOR_H

