// -*- c++ -*-
#ifndef GRAPTOR_FRONTIER_IMPL_H
#define GRAPTOR_FRONTIER_IMPL_H

#include "graptor/frontier/frontier.h"
#include "graptor/primitives.h"
#include "graptor/dsl/primitives.h"

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
		  VID s = part.start_of_vbal( p );
		  VID e = part.end_of_vbal( p );
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
    // TODO: duplicates copy_cast
    expr::array_ro<From,VID,expr::aid_frontier_old> src( fromp );
    expr::array_ro<To,VID,expr::aid_frontier_new> dst( top );
    make_lazy_executor( part )
	.vertex_map( [&]( auto v ) { return dst[v] = src[v]; } )
	.materialize();
}

template<typename GraphType>
void frontier::merge_or( const GraphType & G, frontier & f ) {
    // Nothing to do
    if( f.nActiveVertices() == 0 )
	return;

    switch( ftype ) {
    case frontier_type::ft_true: break;
    case frontier_type::ft_sparse:
	if constexpr ( std::is_same_v<GraphType,GraphCSx> )
	    merge_or_sparse( G, f );
	else
	    merge_or_sparse( G.getCSR(), f );
	break;
    case frontier_type::ft_logical1:
	merge_or_ds( G, getDense<frontier_type::ft_logical1>(), f );
	break;
    case frontier_type::ft_logical4:
	merge_or_ds( G, getDense<frontier_type::ft_logical4>(), f );
	break;
    case frontier_type::ft_bool:
    case frontier_type::ft_logical2:
    case frontier_type::ft_logical8:
    case frontier_type::ft_bit:
    case frontier_type::ft_bit2:
	assert( 0 && "NYI - frontier::merge_or" );
	break;
    case frontier_type::ft_unbacked:
    default: UNREACHABLE_CASE_STATEMENT;
    }
}

/************************************************************************
 * \brief logical-or-merge the frontier f into this
 *
 * This operation incrementally sets true values for those vertices
 * that are listed in the frontier f and are absent from this.
 * It minimizes writes to memory.
 * The code below is agnostic to the type of the frontier f.
 *
 * \tparam GraphType type of the graph object
 * \tparam LHSTy pointer to array of values for the dense frontier this
 *
 * \param G the graph object
 * \param lhs_p pointer to dense array of logical values for frontier this
 * \param f frontier to perform logical-or with
 ************************************************************************/
template<typename GraphType, typename LHSTy>
void frontier::merge_or_ds( const GraphType & G, LHSTy * lhs_p,
			    frontier & f ) {
    const VID * degree_p = nullptr;
    if constexpr ( std::is_same_v<GraphType,GraphCSx> )
	degree_p = G.getDegree();
    else
	degree_p = G.getCSR().getDegree();
    expr::array_ro<VID,VID,expr::aid_graph_degree,array_encoding<VID>,false> degree( const_cast<VID *>( degree_p ) );
    // Should be possible to extend to 1- and 2-bit frontiers in 'this'
    // by passing a suitable encoding here.
    expr::array_ro<LHSTy,VID,expr::aid_frontier_new,array_encoding<LHSTy>,false> lhs( lhs_p );
    VID v_new = 0;
    EID e_new = 0;
    expr::array_ro<VID,VID,expr::aid_frontier_nactv,array_encoding<VID>,false> nactv_( &v_new );
    expr::array_ro<EID,VID,expr::aid_frontier_nacte,array_encoding<EID>,false> nacte_( &e_new );

    make_lazy_executor( G.get_partitioner() )
	.vertex_scan(
	    f, [&]( auto v ) {
		return expr::let<expr::aid_frontier_a>(
		    lhs[v],
		    [&]( auto nf ) {
			// Not sure if the add_predicate construct works
			// in case of 1-bit or 2-bit frontiers
			using Tr = typename decltype(lhs[v])::data_type;
			using MTr = typename Tr::prefmask_traits;
			auto mask = !expr::make_unop_cvt_to_mask<MTr>( nf );
			return expr::make_seq(
			    nacte_[expr::_0(v)] +=
			    expr::_p(
				expr::make_unop_cvt_type<EID>( degree[v] ),
				mask ),
			    nactv_[expr::_0(v)] +=
			    expr::_p( expr::_1(nactv_[expr::_0(v)]),
						 mask ),
			    lhs[v] = expr::_p(
				expr::true_val( lhs[v] ),
				mask )
			    // expr::_true(lhs[v])
			    );
		    } );
	    } )
	.materialize();
    
    nactv += v_new;
    nacte += e_new;
}

void frontier::merge_or_sparse( const GraphCSx & G, frontier & f ) {
    std::cerr << "WARNING: sequential loop in frontier::merge_or_sparse()\n";

    using std::swap;
    
    // Worst-case memory allocation
    frontier newf = frontier::sparse(
	G.numVertices(),
	nActiveVertices() + f.nActiveVertices() );
    VID * newf_p = newf.getSparse();
    VID * lhs_p = getSparse();
    VID * rhs_p = f.getSparse();

    sort( &lhs_p[0], &lhs_p[nActiveVertices()] );
    sort( &rhs_p[0], &rhs_p[f.nActiveVertices()] );

    VID ne = 0;
    const VID * l = lhs_p;
    const VID * le = &lhs_p[nActiveVertices()];
    const VID * r = rhs_p;
    const VID * re = &rhs_p[f.nActiveVertices()];
    VID * o = newf_p;
    for( ; l != le; ++o ) {
	if( r == re ) {
	    for( ; l != le; ++l ) {
		ne += G.getDegree( *l );
		*o++ = *l;
	    }
	    break;
	}
	if( *r < *l ) {
	    ne += G.getDegree( *r );
	    *o = *r++;
	} else {
	    ne += G.getDegree( *l );
	    *o = *l;
	    if( *r == *l )
		++r;
	    ++l;
	}
    }
    for( ; r != re; ++r ) {
	ne += G.getDegree( *r );
	*o++ = *r;
    }

    newf.nactv = o - newf_p;
    newf.nacte = ne;

    swap( *this, newf );
    newf.del();
}

template<typename GraphType, typename LHSTy, typename RHSTy>
void frontier::merge_or_tmpl( const GraphType & G, LHSTy * lhs_p, RHSTy * rhs_p ) {
    const VID * degree_p = nullptr;
    if constexpr ( std::is_same_v<GraphType,GraphCSx> )
	degree_p = G.getDegree();
    else
	degree_p = G.getCSR().getDegree();
    expr::array_ro<VID,VID,expr::aid_graph_degree,array_encoding<VID>,false> degree( const_cast<VID *>( degree_p ) );
    expr::array_ro<LHSTy,VID,expr::aid_frontier_new,array_encoding<LHSTy>,false> lhs( lhs_p );
    expr::array_ro<LHSTy,VID,expr::aid_frontier_old,array_encoding<RHSTy>,false> rhs( rhs_p );
    nactv = 0;
    nacte = 0;
    expr::array_ro<VID,VID,expr::aid_frontier_nactv,array_encoding<VID>,false> nactv_( &nactv );
    expr::array_ro<EID,VID,expr::aid_frontier_nacte,array_encoding<EID>,false> nacte_( &nacte );

    make_lazy_executor( G.get_partitioner() )
	.vertex_scan( [&]( auto v ) {
	    return expr::let<expr::aid_frontier_a>(
		lhs[v] || rhs[v],
		[&]( auto nf ) {
		    // Not sure if the add_predicate construct works in case of
		    // 1-bit or 2-bit frontiers
		    using Tr = simd::detail::mask_preferred_traits_type<
                        LHSTy, decltype(v)::VL>;
		    auto mask = expr::make_unop_cvt_to_mask<Tr>( nf );
		    return expr::make_seq(
			nacte_[expr::_0(v)] +=
			    expr::add_predicate(
				expr::make_unop_cvt_type<EID>( degree[v] ),
				mask ),
			nactv_[expr::_0(v)] +=
			    expr::add_predicate( expr::_1(v), mask ),
			lhs[v] = nf );
		} );
	} )
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

/***********************************************************************
 * Frontier type conversions
 ***********************************************************************/

void frontier::toBool( const partitioner & part ) {
    switch( ftype ) {
    case frontier_type::ft_true: break;
    case frontier_type::ft_unbacked: break;
    case frontier_type::ft_bool: break;
    case frontier_type::ft_bit:
    case frontier_type::ft_bit2:
	assert( 0 && "NYI" );
	break;
    case frontier_type::ft_logical1:
    {
	// std::cerr << "FRONTIER CONVERT frontier from logical<1> to bool\n";
	// No need to reallocate memory. Just reinterpret the values,
	// and remap 0->0 and ~0 -> 1. One may argue whether any action
	// is required at all as usually ~0 is interpreted as true.
	// However, ~0 != true, so it depends on coding style.
	logical<1> * p = get_l<1>().get();
	bool * cp = reinterpret_cast<bool *>( p );
	copy_cast<bool,logical<1>>( part, cp, p );
	ftype = frontier_type::ft_bool;
	break;
    }
    case frontier_type::ft_logical2:
    {
	assert( 0 && "NYI" );
	break;
    }
    case frontier_type::ft_logical4:
    {
	// std::cerr << "FRONTIER CONVERT frontier from logical<4> to bool\n";
	mmap_ptr<logical<4>> l4( std::move( get_l<4>() ) );
	mmap_ptr<bool> & b = get_b();
	new ( &b ) mmap_ptr<bool>();
	b.allocate( numa_allocation_partitioned( part ) );
	copy_cast<bool,logical<4>>( part, b.get(), l4.get() );

	l4.del();
	ftype = frontier_type::ft_bool;
	break;
    }
    case frontier_type::ft_logical8:
    {
	// std::cerr << "FRONTIER CONVERT frontier from logical<8> to bool\n";
	mmap_ptr<logical<8>> l8( std::move( get_l<8>() ) );
	mmap_ptr<bool> & b = get_b();
	new ( &b ) mmap_ptr<bool>();
	b.allocate( numa_allocation_partitioned( part ) );
	copy_cast<bool,logical<8>>( part, b.get(), l8.get() );

	l8.del();
	ftype = frontier_type::ft_bool;
	break;
    }
    case frontier_type::ft_sparse:
    {
	// std::cerr << "FRONTIER CONVERT frontier from sparse to bool\n";
	VID * s = get_s();
	mmap_ptr<bool> & b = get_b();
	new ( &b ) mmap_ptr<bool>();
	b.allocate( numa_allocation_partitioned( part ) );
	bool *bp = b.get();
	clear_by_partition( part, bp );
	parallel_loop( (VID)0, nactv, [&]( VID i ) { bp[s[i]] = 1; } );
	delete[] s;
	ftype = frontier_type::ft_bool;
	break;
    }
    default: UNREACHABLE_CASE_STATEMENT;
    }
}

void frontier::toBit( const partitioner & part ) {
    switch( ftype ) {
    case frontier_type::ft_true: break;
    case frontier_type::ft_unbacked: break;
    case frontier_type::ft_bool:
    {
	partitioner cpart = part.contract( sizeof(unsigned char) );
	mmap_ptr<bool> bo( std::move( get_b() ) );
	mmap_ptr<unsigned char> & b = get_bit();
	new ( &b ) mmap_ptr<unsigned char>();
	b.allocate( numa_allocation_partitioned( cpart ) );
	copy_cast<bitfield<1>,bool>(
	    part, reinterpret_cast<bitfield<1>*>( b.get() ), bo.get() );

	bo.del();
	ftype = frontier_type::ft_bit;
	break;
    }
    case frontier_type::ft_bit: break;
    case frontier_type::ft_bit2: break;
    case frontier_type::ft_logical1:
    case frontier_type::ft_logical2:
	assert( 0 && "NYI" );
	break;
    case frontier_type::ft_logical4:
    {
	partitioner cpart = part.contract( sizeof(unsigned char) );
	mmap_ptr<logical<4>> l4( std::move( get_l<4>() ) );
	mmap_ptr<unsigned char> & b = get_bit();
	new ( &b ) mmap_ptr<unsigned char>();
	b.allocate( numa_allocation_partitioned( part ) );
	copy_cast<bitfield<1>,logical<4>>(
	    part, reinterpret_cast<bitfield<1> *>( b.get() ), l4.get() );

	l4.del();
	ftype = frontier_type::ft_bit;
	break;
    }
    case frontier_type::ft_logical8:
    {
	assert( 0 && "NYI" );
	mmap_ptr<logical<8>> l8( std::move( get_l<8>() ) );
	mmap_ptr<bool> & b = get_b();
	new ( &b ) mmap_ptr<bool>();
	b.allocate( numa_allocation_partitioned( part ) );
	copy_cast<bitfield<1>,logical<8>>(
	    part, reinterpret_cast<bitfield<1> *>( b.get() ), l8.get() );

	l8.del();
	ftype = frontier_type::ft_bit;
	break;
    }
    case frontier_type::ft_sparse:
    {
	partitioner cpart = part.contract( sizeof(unsigned char) );
	VID * s = get_s();
	mmap_ptr<unsigned char> & b = get_bit();
	new ( &b ) mmap_ptr<unsigned char>();
	b.allocate( numa_allocation_partitioned( cpart ) );
	unsigned char *bp = b.get();
	clear_by_partition( cpart, bp );

	if( nactv > 4096 ) {
	    using Enc = array_encoding_bit<1>;
	    using data_type = simd::ty<bitfield<1>,1>;
	    parallel_loop( (VID)0, nactv, [&]( VID i ) {
		VID v = s[i];
		auto l = Enc::extract<data_type>( Enc::ldcas<data_type>( bp, v ) );
		auto ll = bitfield<1>( 1 );
		while( !Enc::cas<data_type>( bp, v, l, ll ) ) {
		    l = Enc::extract<data_type>( Enc::ldcas<data_type>( bp, v ) );
		}
	    } );
	} else {
	    for( VID i=0; i < nactv; i++ ) {
		VID idx = s[i] / 8;
		bp[idx] |= ((unsigned char)1) << ( s[i] % 8 );
	    }
	}
	
	delete[] s;
	ftype = frontier_type::ft_bit;
	break;
    }
    default: UNREACHABLE_CASE_STATEMENT;
    }
}

void frontier::toBit2( const partitioner & part ) {
    switch( ftype ) {
    case frontier_type::ft_true: break;
    case frontier_type::ft_unbacked: break;
    case frontier_type::ft_bool:
    {
	partitioner cpart = part.contract( 4 );
	mmap_ptr<bool> bo( std::move( get_b() ) );
	mmap_ptr<unsigned char> & b = get_bit2();
	new ( &b ) mmap_ptr<unsigned char>();
	b.allocate( numa_allocation_partitioned( cpart ) );
	std::cerr << "WARNING: sequential loop in frontier::toBit()\n";
	for( VID v=0; v < nv; v += 4 ) {
	    unsigned char m = 0;
	    for( VID vs=0; vs < 8; ++vs ) {
		bool val = bo.get()[v+8-vs];
		m <<= 2;
		m |= (val == 0 ? 0 : 3);
	    }
	    b.get()[v/8] = m;
	}

	bo.del();
	ftype = frontier_type::ft_bit;
	break;
    }
    case frontier_type::ft_bit:
    {
	assert( 0 && "NYI" );
	break;
    }
    case frontier_type::ft_bit2: break;
    case frontier_type::ft_logical1:
    case frontier_type::ft_logical2:
	assert( 0 && "NYI" );
	break;
    case frontier_type::ft_logical4:
    {
	assert( 0 && "NYI" );
	partitioner cpart = part.contract( sizeof(unsigned char) );
	mmap_ptr<logical<4>> l4( std::move( get_l<4>() ) );
	mmap_ptr<unsigned char> & b = get_bit2();
	new ( &b ) mmap_ptr<unsigned char>();
	b.allocate( numa_allocation_partitioned( part ) );
	unsigned short * bp = reinterpret_cast<unsigned short *>( b.get() );

#if __AVX512F__
	std::cerr << "WARNING: sequential loop in frontier::toBit2()\n";
	for( VID v=0; v < nv; v += 16 ) {
	    auto vv = simd::vector<logical<4>,16>::load_from( &l4.get()[v] );
	    simd::mask<0,16> m = vv.template asmask<simd::detail::mask_bit_traits<16>>();
	    bp[v/16] = m.get(); // div 8 because unsigned char *
	}
#else
	assert( 0 && "Not supported" );
#endif

	l4.del();
	ftype = frontier_type::ft_bit;
	break;
    }
    case frontier_type::ft_logical8:
    {
	assert( 0 && "NYI" );
	mmap_ptr<logical<8>> l8( std::move( get_l<8>() ) );
	mmap_ptr<bool> & b = get_b();
	new ( &b ) mmap_ptr<bool>();
	b.allocate( numa_allocation_partitioned( part ) );
	std::cerr << "WARNING: sequential loop in frontier::toBit2()\n";
	for( VID v=0; v < nv; ++v )
	    b.get()[v] = (bool)l8.get()[v];

	l8.del();
	ftype = frontier_type::ft_bool;
	break;
    }
    case frontier_type::ft_sparse:
    {
	VID * s = get_s();
	mmap_ptr<unsigned char> & b = get_bit2();
	new ( &b ) mmap_ptr<unsigned char>();
	b.allocate( numa_allocation_partitioned( part ) );
	unsigned char *bp = b.get();

	partitioner cpart = part.contract( 4 ); // 8 bits / 2 bits per elm
	clear_by_partition( cpart, bp );

	// TODO: race conditions?
	parallel_loop( (VID)0, nactv, [&]( VID i ) {
	    bp[s[i]/4] |= 3 << ((s[i]%4)*2);
	} );
	delete[] s;
	ftype = frontier_type::ft_bit2;
	break;
    }
    default: UNREACHABLE_CASE_STATEMENT;
    }
}

void frontier::toLogical2( const partitioner & part ) {
    switch( ftype ) {
    case frontier_type::ft_true: break;
    case frontier_type::ft_unbacked: break;
    case frontier_type::ft_bool:
    case frontier_type::ft_bit:
    case frontier_type::ft_logical8:
	assert( 0 && "NYI" );
	break;
    case frontier_type::ft_logical1:
    {
	convert_logical<1,2>( part );
	ftype = frontier_type::ft_logical2;
	break;
    }
    case frontier_type::ft_logical4:
    {
	convert_logical<4,2>( part );
	ftype = frontier_type::ft_logical2;
	break;
    }
    case frontier_type::ft_logical2:
	break; // nothing to do
    case frontier_type::ft_sparse:
    {
	VID * s = get_s();
	mmap_ptr<logical<2>> & l2 = get_l<2>();
	new ( &l2 ) mmap_ptr<logical<2>>();
	l2.allocate( numa_allocation_partitioned( part ) );
	logical<2> *l2p = l2.get();
	clear_by_partition( part, l2p );
	std::cerr << "WARNING: sequential loop in frontier::toLogical2()\n";
	/*parallel_*/for( VID i=0; i < nactv; i++ ) {
	    assert( !l2p[s[i]] );
	    l2p[s[i]] = logical<2>::true_val();
	}

	delete[] s;
	ftype = frontier_type::ft_logical2;
	break;
    }
    default: UNREACHABLE_CASE_STATEMENT;
    }
}

void frontier::toLogical4( const partitioner & part ) {
    switch( ftype ) {
    case frontier_type::ft_true: break;
    case frontier_type::ft_unbacked: break;
    case frontier_type::ft_bool:
    {
	// std::cerr << "FRONTIER CONVERT frontier from bool to logical<4>\n";
	mmap_ptr<bool> b( std::move( get_b() ) );
	mmap_ptr<logical<4>> & l4 = get_l<4>();
	new ( &l4 ) mmap_ptr<logical<4>>();
	l4.allocate( numa_allocation_partitioned( part ) );
	logical<4> * l4p = l4.get();
	bool * bp = b.get();
	copy_cast<logical<4>,bool>( part, l4.get(), bp );

	b.del();
	ftype = frontier_type::ft_logical4;
	break;
    }
    case frontier_type::ft_bit:
    {
	// partitioner cpart = part.contract( sizeof(unsigned char) );
	mmap_ptr<unsigned char> b( std::move( get_bit() ) );
	mmap_ptr<logical<4>> & l4 = get_l<4>();
	new ( &l4 ) mmap_ptr<logical<4>>();
	l4.allocate( numa_allocation_partitioned( part ) );
	copy_cast<logical<4>,bitfield<1>>(
	    part, l4.get(), reinterpret_cast<bitfield<1> *>( b.get() ) );
	b.del();
	ftype = frontier_type::ft_logical4;
	break;
    }
    case frontier_type::ft_logical4:
	break; // nothing to do
    case frontier_type::ft_msb4:
	// Leave as is; works nearly same way as logical<4>, so why change?
	break;
    case frontier_type::ft_logical1:
    {
	convert_logical<1,4>( part );
	ftype = frontier_type::ft_logical4;
	break;
    }
    case frontier_type::ft_logical8:
    {
	convert_logical<8,4>( part );
	ftype = frontier_type::ft_logical4;
	break;
    }
    case frontier_type::ft_logical2:
    {
	convert_logical<2,4>( part );
	ftype = frontier_type::ft_logical4;
	break;
    }
    case frontier_type::ft_sparse:
    {
	// std::cerr << "FRONTIER CONVERT frontier from sparse to logical<4>\n";
	VID * s = get_s();
	mmap_ptr<logical<4>> & l4 = get_l<4>();
	new ( &l4 ) mmap_ptr<logical<4>>();
	l4.allocate( numa_allocation_partitioned( part ) );
	logical<4> *l4p = l4.get();
	// map_vertexL( part, [&](VID v) { l4p[v]=logical<4>::false_val(); } );
	clear_by_partition( part, l4p );
	parallel_loop( (VID)0, nactv, [&]( VID i ) {
	    // Assertion meaningful only when prior sparse edgemaps
	    // are guaranteed to remove duplicates
	    // assert( !l4p[s[i]] );
	    l4p[s[i]] = logical<4>::true_val();
	} );

	delete[] s;
	ftype = frontier_type::ft_logical4;
	break;
    }
    default: UNREACHABLE_CASE_STATEMENT;
    }
}

void frontier::toLogical8( const partitioner & part ) {
    switch( ftype ) {
    case frontier_type::ft_true: break;
    case frontier_type::ft_unbacked: break;
    case frontier_type::ft_bool: break;
    {
	// std::cerr << "FRONTIER CONVERT frontier from bool to logical<8>\n";
	mmap_ptr<bool> b( std::move( get_b() ) );
	mmap_ptr<logical<8>> & l8 = get_l<8>();
	new ( &l8 ) mmap_ptr<logical<8>>();
	l8.allocate( numa_allocation_partitioned( part ) );
	logical<8> * l8p = l8.get();
	bool * bp = b.get();
	copy_cast<logical<8>,bool>( part, l8.get(), bp );

	b.del();
	ftype = frontier_type::ft_logical8;
	break;
    }
    case frontier_type::ft_bit:
    case frontier_type::ft_logical1:
    case frontier_type::ft_logical2:
    case frontier_type::ft_logical4:
	assert( 0 && "NYI" );
	break;
    case frontier_type::ft_sparse:
    {
	// std::cerr << "FRONTIER CONVERT frontier from sparse to logical<8>\n";
	VID * s = get_s();
	mmap_ptr<logical<8>> & l8 = get_l<8>();
	new ( &l8 ) mmap_ptr<bool>();
	l8.allocate( numa_allocation_partitioned( part ) );
	logical<8> *l8p = l8.get();
	// map_vertexL( part, [&](VID v) { l8p[v]=logical<8>::false_val(); } );
	clear_by_partition( part, l8p );
	parallel_loop( (VID)0, nactv, [&]( VID i ) {
	    l8p[s[i]] = logical<8>::true_val();
	} );

	delete[] s;
	ftype = frontier_type::ft_logical8;
	break;
    }
    default: UNREACHABLE_CASE_STATEMENT;
    }
}

#endif // GRAPTOR_FRONTIER_IMPL_H
