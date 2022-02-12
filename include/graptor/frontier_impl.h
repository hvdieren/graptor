// -*- c++ -*-
#ifndef GRAPTOR_FRONTIER_IMPL_H
#define GRAPTOR_FRONTIER_IMPL_H

#include "graptor/frontier.h"

// Helper class to add up out-degree for all active vertices
template<class vertex>
class GoutDegree
{
    const graph<vertex> &PG;
    bool * dense;

public:
    GoutDegree( const graph<vertex>& pg, bool * pdense )
	: PG(pg), dense(pdense) { }
    // Note: adding up multiple degrees may overflow VID width in some graphs
    std::pair<VID, EID> operator()( VID i ) {
        return make_pair( (VID)dense[i],
			  dense[i] ? (EID)PG.V[i].getOutDegree() : (EID)0 );
    }
};
template<class vertex, unsigned short W>
class GoutDegreeL
{
    const graph<vertex> &PG;
    logical<W> * dense;

public:
    GoutDegreeL( const graph<vertex>& pg, logical<W> * pdense )
	: PG(pg), dense(pdense) { }
    // Note: adding up multiple degrees may overflow VID width in some graphs
    std::pair<VID, EID> operator()( VID i ) {
	bool d = dense[i] != logical<W>::false_val();
        return make_pair( (VID)d, d ? (EID)PG.V[i].getOutDegree() : (EID)0 );
    }
};
template<class vertex>
class GoutDegreeV
{
    graph<vertex> G;
    VID *s;

public:
    GoutDegreeV(graph<vertex> G_, VID* s_) : G(G_), s(s_) { }
    // Note: adding up multiple degrees may overflow VID width in some graphs
    EID operator()( VID i ) {
        return G.V[s[i]].getOutDegree();
    }
};
class GoutDegreeCSx
{
    const GraphCSx &PG;
    bool * dense;

public:
    GoutDegreeCSx( const GraphCSx &pg, bool * pdense )
	: PG(pg), dense(pdense) { }
    // Note: adding up multiple degrees may overflow VID width in some graphs
    std::pair<VID, EID> operator()( VID i ) {
	const EID *idx = PG.getIndex();
	EID deg = idx[i+1] - idx[i];
        return make_pair( (VID)dense[i], dense[i] ? deg : (EID)0 );
    }
};
class GoutDegreeCSxBit
{
    const GraphCSx &PG;
    unsigned char * dense;

public:
    GoutDegreeCSxBit( const GraphCSx &pg, unsigned char * pdense )
	: PG(pg), dense(pdense) { }
    // Note: adding up multiple degrees may overflow VID width in some graphs
    std::pair<VID, EID> operator()( VID i ) {
	const EID *idx = PG.getIndex();
	EID deg = idx[i+1] - idx[i];
	unsigned char d = ( idx[i/sizeof(unsigned char)]
			    >> (i % sizeof(unsigned char)) ) & (unsigned char)1;
        return make_pair( (VID)d, d ? deg : (EID)0 );
    }
};
template<unsigned short W>
class GoutDegreeCSxL
{
    const GraphCSx &PG;
    logical<W> * dense;

public:
    GoutDegreeCSxL( const GraphCSx &pg, logical<W> * pdense )
	: PG(pg), dense(pdense) { }
    // Note: adding up multiple degrees may overflow VID width in some graphs
    pair<VID, EID> operator()( VID i ) {
	const EID *idx = PG.getIndex();
	EID deg = idx[i+1] - idx[i];
	bool d = dense[i] != logical<W>::false_val();
        return make_pair( (VID)d, d ? deg : (VID)0 );
    }
};
class GoutDegreeCSxV
{
    const GraphCSx & G;
    VID *s;

public:
    GoutDegreeCSxV( const GraphCSx & G_, VID* s_ ) : G( G_ ), s( s_ ) { }
    // Note: adding up multiple degrees may overflow VID width in some graphs
    EID operator()( VID i ) {
	const EID *idx = G.getIndex();
	EID deg = idx[s[i]+1] - idx[s[i]];
        return deg;
    }
};

template<class vertex>
void frontier::calculateActiveCounts( graph<vertex> G, VID n ) {
    // Calculate statistics on active vertices and their out-degree
    switch( ftype ) {
    case frontier_type::ft_true: break;
    case frontier_type::ft_bool:
    {
	bool * d = get_b().get();
	std::pair<VID,EID> p = sequence::reduce<VID>(
	    (VID)0, (VID)nv, GoutDegree<vertex>(G, d));
	nactv = p.first;
	nacte = p.second;
	break;
    }
    case frontier_type::ft_bit:
    case frontier_type::ft_logical1:
    case frontier_type::ft_logical2:
	assert( 0 && "NYI" );
        break;
    case frontier_type::ft_logical4:
    {
	logical<4> * d = get_l<4>().get();
	std::pair<VID,EID> p = sequence::reduce<VID>(
	    (VID)0, (VID)nv, GoutDegreeL<vertex,4>(G, d));
	nactv = p.first;
	nacte = p.second;
	break;
    }
    case frontier_type::ft_logical8:
    {
	logical<8> * d = get_l<8>().get();
	std::pair<VID,EID> p = sequence::reduce<VID>(
	    (VID)0, (VID)nv, GoutDegreeL<vertex,8>(G, d));
	nactv = p.first;
	nacte = p.second;
	break;
    }
    case frontier_type::ft_sparse:
    {
	assert( n != ~(VID)0
		&& "sparse case requires non-default value for n" );

	nactv = n;
	if( nactv == 0 )
	    nacte = 0;
	else
	    nacte = sequence::reduce<EID>((VID)0, nactv, addF<VID>(),
					  GoutDegreeV<vertex>(G, get_s()));
	break;
    }
    case frontier_type::ft_unbacked:
    case frontier_type::ft_bit2:
    default: UNREACHABLE_CASE_STATEMENT;
    }
}

    // Note: this function must be passed the CSR graph
void frontier::calculateActiveCounts( GraphCSx G, VID from, VID to ) {
    // Calculate statistics on active vertices and their out-degree
    switch( ftype ) {
    case frontier_type::ft_true: break;
    case frontier_type::ft_bool:
    {
	bool * d = get_b().get();
	std::pair<VID,EID> p = sequence::reduceSerial<VID>(
	    (VID)from, (VID)to, GoutDegreeCSx(G, d));
	__sync_fetch_and_add( &nactv, p.first );
	__sync_fetch_and_add( &nacte, p.second );
	break;
    }
    case frontier_type::ft_bit:
    {
	unsigned char * d = get_bit().get();
	std::pair<VID,EID> p = sequence::reduceSerial<VID>(
	    (VID)from, (VID)to, GoutDegreeCSxBit(G, d));
	__sync_fetch_and_add( &nactv, p.first );
	__sync_fetch_and_add( &nacte, p.second );
	break;
    }
    case frontier_type::ft_logical1:
    case frontier_type::ft_logical2:
	assert( 0 && "NYI" );
	break;
    case frontier_type::ft_logical4:
    {
	logical<4> * d = get_l<4>().get();
	std::pair<VID,EID> p = sequence::reduceSerial<VID>(
	    (VID)from, (VID)to, GoutDegreeCSxL<4>(G, d));
	__sync_fetch_and_add( &nactv, p.first );
	__sync_fetch_and_add( &nacte, p.second );
	break;
    }
    case frontier_type::ft_logical8:
    {
	logical<8> * d = get_l<8>().get();
	std::pair<VID,EID> p = sequence::reduceSerial<VID>(
	    (VID)from, (VID)to, GoutDegreeCSxL<8>(G, d));
	__sync_fetch_and_add( &nactv, p.first );
	__sync_fetch_and_add( &nacte, p.second );
	break;
    }
    case frontier_type::ft_sparse:
	assert( 0 && "Not applicable" );
	break;
    case frontier_type::ft_unbacked:
    case frontier_type::ft_bit2:
    default: UNREACHABLE_CASE_STATEMENT;
    }
}

template<typename GraphType>
void frontier::calculateActiveCounts( const GraphType & G ) {
    calculateActiveCounts( G.getCSR(), G.get_partitioner(), ~(VID)0 );
}

// Note: this function must be passed the CSR graph
void frontier::calculateActiveCounts( const GraphCSx & G,
				      const partitioner & part,
				      VID n ) {
    // Calculate statistics on active vertices and their out-degree
    switch( ftype ) {
    case frontier_type::ft_true: break;
    case frontier_type::ft_bool:
	calculateActiveCounts_tmpl( part, G.getDegree(), get_b().get() );
	break;
    case frontier_type::ft_bit:
    {
	// TODO: this is erroneous - may read uninitialised values in the
	//       array of bits
	unsigned char * d = get_bit().get();
	std::pair<VID,EID> p = sequence::reduce<VID>( (VID)0, (VID)nv,
						      GoutDegreeCSxBit(G, d));
	nactv = p.first;
	nacte = p.second;
	break;
    }
    case frontier_type::ft_logical1:
	calculateActiveCounts_tmpl( part, G.getDegree(), get_l<1>().get() );
	break;
    case frontier_type::ft_logical2:
	calculateActiveCounts_tmpl( part, G.getDegree(), get_l<2>().get() );
	break;
    case frontier_type::ft_logical4:
	calculateActiveCounts_tmpl( part, G.getDegree(), get_l<4>().get() );
	break;
    case frontier_type::ft_logical8:
	calculateActiveCounts_tmpl( part, G.getDegree(), get_l<8>().get() );
	break;
    case frontier_type::ft_sparse:
    {
	assert( n != ~(VID)0
		&& "sparse case requires non-default value for n" );

	nactv = n;
	if( nactv == 0 )
	    nacte = 0;
	else
	    nacte = sequence::reduce<EID>(VID(0), nactv, addF<EID>(),
					  GoutDegreeCSxV(G, get_s()));
	break;
    }
    case frontier_type::ft_unbacked:
    case frontier_type::ft_bit2:
    default: UNREACHABLE_CASE_STATEMENT;
    }
}

template<typename FlagsTy>
void frontier::calculateActiveCounts_tmpl(
    const partitioner & part, const VID * outdeg, const FlagsTy * flags ) {
    int npart = part.get_num_partitions();
    VID * vsum = new VID[npart];
    EID * esum = new EID[npart];

    map_partition(
	part, [&]( int p ) {
		  VID s = part.start_of( p );
		  VID e = part.end_of( p );
		  VID nv = 0;
		  EID ne = 0;
		  for( VID v=s; v < e; ++v ) {
		      bool a = false;
		      if constexpr ( is_logical_v<FlagsTy> ) {
			  if( flags[v] >> (8*sizeof(FlagsTy)-1) )
			      a = true;
		      } else {
			  if( !!flags[v] )
			      a = true;
		      }
		      if( a ) {
			  nv += 1;
			  ne += outdeg[v];
		      }
		  }
		  vsum[p] = nv;
		  esum[p] = ne;
	      } );

    nactv = vsum[0];
    nacte = esum[0];

    for( int p=1; p < npart; ++p ) {
	nactv += vsum[p];
	nacte += esum[p];
    }

    delete[] vsum;
    delete[] esum;
}

template<typename From, typename To>
void frontier::convert_logical( const partitioner & part,
				From * fromp, To * top ) {
    // Should work for bool too
    expr::array_ro<From,VID,expr::aid_frontier_old> src( fromp );
    expr::array_ro<To,VID,expr::aid_frontier_new> dst( top );
    make_lazy_executor( part )
	.vertex_map( [&]( auto v ) { return dst[v] = src[v]; } )
	.materialize();
}

inline std::ostream & operator << ( std::ostream & os, const frontier & F ) {
    VID n = F.nVertices();
    VID nv = F.nActiveVertices();
    bool * f = F.template getDense<frontier_type::ft_bool>();
    os << " (" << F.getType() << ") ";
    if( f ) {
	os << "frontier [B] #" << nv << ":";
	for( VID v=0; v < n; ++v )
	    os << ' ' << ( f[v] ? 'X' : '.' );
	return os;
    }
    {
	logical<4> * f = F.template getDense<frontier_type::ft_logical4>();
	if( f ) {
	    os << "frontier [L4] #" << nv << ":";
	    for( VID v=0; v < n; ++v )
		os << ( f[v] ? 'X' : '.' );
	    return os;
	}
    }
    {
	logical<1> * f = F.template getDense<frontier_type::ft_logical1>();
	if( f ) {
	    os << "frontier [L1] #" << nv << ":";
	    for( VID v=0; v < n; ++v )
		os << ( f[v] ? 'X' : '.' );
	    return os;
	}
    }
    {
	VID * f = F.getSparse();
	if( f ) {
	    os << "frontier [S] #" << nv << ":";
	    for( VID v=0; v < nv && v < 200; ++v )
		os << ' ' << f[v];
	    if( nv >= 200 )
		os << " ...";
	    return os;
	}
    }
    os << "frontier [?] #" << nv << ": (NYI)";
    return os;
}


#endif // GRAPTOR_FRONTIER_IMPL_H
