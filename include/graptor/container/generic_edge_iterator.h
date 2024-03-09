// -*- C++ -*-
#ifndef GRAPTOR_CONTAINER_GENERIC_EDGE_ITERATOR_H
#define GRAPTOR_CONTAINER_GENERIC_EDGE_ITERATOR_H

namespace graptor {

template<typename lVID, typename lEID>
class generic_edge_iterator {
public:
    using VID = lVID;
    using EID = lEID;

    // iterator traits
    using iterator_category = std::input_iterator_tag;
    using value_type = std::pair<VID,VID>;
    using difference_type = EID;
    using pointer = const std::pair<VID,VID> *;
    using reference = std::pair<VID,VID>;

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

} // namespace graptor

#endif // GRAPTOR_CONTAINER_GENERIC_EDGE_ITERATOR_H

