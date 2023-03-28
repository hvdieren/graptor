// -*- c++ -*-
// A Pattern Decomposed Graph
#include "graptor/graptor.h"
#include "graptor/api.h"

#include "graptor/graph/contract/vertex_set.h"
#include "graptor/graph/GraphCSx.h"
#include "graptor/graph/GraphPDG.h"

#define NOBENCH
#define MD_ORDERING 0
#define VARIANT 11
#define FUSION 1
#include "../bench/KC_bucket.C"
#undef FUSION
#undef VARIANT
#undef MD_ORDERING
#undef NOBENCH

enum cvt_pdg_variable_name {
    var_dcount = var_kc_num + 0,
    var_max = var_kc_num + 1,
    var_min = var_kc_num + 2,
    var_covered = var_kc_num + 3,
    var_counts = var_kc_num + 4
};

class Cliqueling {
public:
    Cliqueling( VID v, const VID * nI, const VID * nE ) {
	m_vertices.push( v );
	m_candidates.push( nI, nE );
    } 
    Cliqueling( contract::vertex_set<VID> && v,
		contract::vertex_set<VID> && c ) {
	m_vertices.swap( std::forward<contract::vertex_set<VID>>( v ) );
	m_candidates.swap( std::forward<contract::vertex_set<VID>>( c ) );
    }
    Cliqueling( VID v, contract::vertex_set<VID> && c ) {
	m_vertices.push( v );
	m_candidates.swap( std::forward<contract::vertex_set<VID>>( c ) );
    }
    Cliqueling() { }

    bool is_compatible( VID v ) const {
	return m_candidates.contains( v );
    }

    bool contains( VID v ) const {
	return m_vertices.contains( v ) || m_candidates.contains( v );
    }

    bool members_contained( const Cliqueling & C ) const {
	// Based on assumption of partitions of vertex sets
	// i.e. C.m_vertices intersect C.m_candidates is empty
	assert( C.m_candidates.intersection_size( C.m_vertices ) == 0 );
	VID im = C.m_vertices.intersection_size( m_vertices );
	VID ic = C.m_candidates.intersection_size( m_vertices );
	return im + ic == m_vertices.size();
    }

    // Size of clique after inserting vertex v
    size_t added_size( VID v, const VID * nI, const VID * nE ) const {
	return m_candidates.intersection_size( nI, nE - nI )
	    + m_vertices.size() + 1;
    }

    void add_vertex( VID v, const VID * nI, const VID * nE ) {
	contract::vertex_set<VID> ins = m_candidates.intersect( nI, nE-nI );
	std::swap( ins, m_candidates );
	m_vertices.add( v );
    }

    bool is_complete() const {
	return m_candidates.size() == 0;
    }

    size_t size() const {
	return m_vertices.size();
    }
    bool empty() const {
	return m_vertices.empty();
    }
    void clear() {
	// Clear and release memory
	contract::vertex_set<VID> me, mc; // empty
	swap( me, mc );
    }
    size_t potential() const {
	return m_candidates.size();
    }

    void swap( contract::vertex_set<VID> & M,
	       contract::vertex_set<VID> & C ) {
	using std::swap;
	swap( m_vertices, M );
	swap( m_candidates, C );
    }

    void swap_candidates( contract::vertex_set<VID> & C ) {
	using std::swap;
	swap( m_candidates, C );
    }

    void swap( VID u, contract::vertex_set<VID> & C ) {
	using std::swap;
	m_vertices.clear();
	m_vertices.push( u );
	swap( m_candidates, C );
    }

    void swap( Cliqueling & c ) {
	using std::swap;
	swap( m_vertices, c.m_vertices );
	swap( m_candidates, c.m_candidates );
    }

    void intersect_candidates( Cliqueling * cl,
			       contract::vertex_set<VID> & C,
			       contract::vertex_set<VID> & Cur,
			       contract::vertex_set<VID> & Cvr ) {
	m_candidates.intersect( cl->m_candidates, C, Cur, Cvr );
    }

    void union_vertices( Cliqueling * cl ) {
	m_vertices.add( cl->m_vertices.begin(), cl->m_vertices.end() );
    }
    
    friend ostream & operator << ( ostream & os, const Cliqueling & c );

    const auto & get_vertices() const { return m_vertices; }
    auto & get_vertices() { return m_vertices; }
    const auto & get_candidates() const { return m_candidates; }
    auto & get_candidates() { return m_candidates; }

private:
    contract::vertex_set<VID> m_vertices;
    contract::vertex_set<VID> m_candidates;
};

ostream & operator << ( ostream & os, const Cliqueling & c ) {
    for( auto && v : c.m_vertices )
	os << ' ' << v;
    os << " | ";
    for( auto && v : c.m_candidates )
	os << ' ' << v;
    return os;
}

class EdgeCliquePartition {
public:
    EdgeCliquePartition( const GraphCSx & G,
			 const VID * const order,
			 const VID * const rev_order )
	: m_G( G ),
	  m_order( order ),
	  m_rev_order( rev_order ),
	  m_vertex_cliques( G.numEdges(), numa_allocation_interleaved() ),
	  m_singleton( G.numVertices(), numa_allocation_interleaved() ),
	  m_num_2cliques( 0 ) {
	m_cliquelings.resize( G.numEdges() );
	parallel_loop( EID(0), G.numEdges(), [&]( EID i ) {
	    m_vertex_cliques[i] = i;
	} );
	parallel_loop( VID(0), G.numVertices(), [&]( VID i ) {
	    m_singleton[i] = ~intptr_t(0);
	} );
    }
    ~EdgeCliquePartition() {
/*
	for( auto && c : m_cliquelings ) {
	    if( c )
		delete reinterpret_cast<Cliqueling *>( c );
	}
	m_cliquelings.del();
*/
	m_vertex_cliques.del();
	m_singleton.del();
    }

    void find_cliques() {
	VID n = m_G.numVertices();
	EID m = m_G.numEdges();

	timer tm;
	tm.start();

	// TODO: order by decreasing coreness

	// parallel_loop( VID(0), n, [&]( VID v ) {
	for( VID i=0; i < n; ++i ) {
	    VID v = m_order[i];
	    const EID * const index = m_G.getIndex();
	    const VID * const edges = m_G.getEdges();

	    EID e = index[v];
	    EID ee = index[v+1];

	    VID * ngh = new VID[ee-e];
	    std::copy( &edges[index[v]], &edges[index[v+1]], ngh );
	    std::sort( &ngh[0], &ngh[ee-e], [&]( VID a, VID b ) {
		return m_rev_order[a] < m_rev_order[b];
	    } );
	    
	    // for( ; e != ee; ++e ) {
	    for( VID x=0; x < ee-e; ++x ) {
		// VID u = edges[e];
		VID u = ngh[x];
		// Only visit each edge once. Ignore self-edges.
		// Assumes neighbour list is sorted in increasing order.
		if( m_rev_order[u] >= m_rev_order[v] )
		    break;

		// TODO: Use an ordering rule to keep trees flat
		VID a = v;
		VID b = u;
		// if( rev_order[a] > rev_order[b] ) {
		if( a > b ) {
		    std::swap( a, b );
		}

		// m_locks[a].lock();
		// m_locks[b].lock();

		process_edge( a, b );

		// m_locks[b].unlock();
		// m_locks[a].unlock();
	    }

	    delete[] ngh;

	    if( i % 10000 == 0 ) {
		std::cerr << "@" << tm.next() << " i=" << i << ": ";
		sizes();
	    } else if( i % 1000 == 0 ) {
		std::cerr << " i=" << i << "\n";
	    }
	    // check_partitions();
	} // );

	tm.stop();
	std::cerr << "total time edge clique partition: " << tm.total() << "\n";

	std::cerr << "reporting...\n";

	report();

	std::cerr << "checking edges...\n";

	check_edges();
    }

    auto count_cliques( VID min_size ) const {
	EID m = m_G.numEdges();
	VID nc = 0;
	EID nm = 0;
	EID ne = 0;
	for( EID e=0; e < m; ++e ) {
	    if( m_vertex_cliques[e] == e && !m_cliquelings[e].empty() ) {
		VID sz = m_cliquelings[e].size();
		if( sz >= min_size ) {
		    nc++;
		    nm += sz;
		    ne += EID( sz - 1 ) * EID( sz );
		}
	    }
	}

	return std::make_tuple( nc, nm, ne );
    }
    
    void write_cliques( CliqueList<VID,EID,VID> & cliques,
			CompressedList<VID,EID> & remainder,
			VID min_size ) {
	VID n = m_G.numVertices();
	EID m = m_G.numEdges();
	EID * cindex = cliques.get_corpus().get_index();
	VID * cmembers = cliques.get_corpus().get_members();
	EID * lindex = cliques.get_links().get_index();
	VID * lmembers = cliques.get_links().get_members();
	EID * rindex = remainder.get_index();
	VID * redges = remainder.get_members();
	EID nc = 0;
	EID nm = 0;
	EID ne = 0;

	std::fill( &cindex[0], &cindex[cliques.get_num_cliques()], 0 );
	std::fill( &lindex[0], &lindex[n], 0 );
	std::fill( &rindex[0], &rindex[n], 0 );

	// First, a counting exercise
	for( EID e=0; e < m; ++e ) {
	    if( m_vertex_cliques[e] == e && !m_cliquelings[e].empty() ) {
		VID sz = m_cliquelings[e].size();
		if( sz >= min_size ) {
		    // Another clique, to count offset in members list
		    cindex[nc++] = sz;
		    
		    // All of the vertices have a link to a clique
		    for( auto u : m_cliquelings[e].get_vertices() )
			lindex[u]++;
		} else {
		    // Singular edges
		    auto I = m_cliquelings[e].get_vertices().begin();
		    auto F = std::next( I, 1 );
		    if( sz == 2 ) {
			rindex[*I]++;
			rindex[*F]++;
		    } else if( sz == 3 ) {
			auto G = std::next( F, 1 );

			rindex[*I] += 2;
			rindex[*F] += 2;
			rindex[*G] += 2;
		    } else
			assert( 0 && "NYI" );
		}
	    }
	}

	assert( nc == cliques.get_num_cliques() );

	// Sum up the counts
	EID mc = sequence::plusScan( cindex, cindex, nc );
	assert( mc == cliques.get_num_members() );
	cindex[nc] = mc;

	EID ml = sequence::plusScan( lindex, lindex, n );
	assert( ml == cliques.get_num_links() );
	lindex[n] = ml;

	EID mr = sequence::plusScan( rindex, rindex, n );
	assert( mr == remainder.get_num_members() );
	rindex[n] = mr;

	// Place data
	nc = 0;
	for( EID e=0; e < m; ++e ) {
	    if( m_vertex_cliques[e] == e && !m_cliquelings[e].empty() ) {
		VID sz = m_cliquelings[e].size();
		if( sz >= min_size ) {
		    // Fill in clique info
		    std::copy( m_cliquelings[e].get_vertices().begin(),
			       m_cliquelings[e].get_vertices().end(),
			       &cmembers[cindex[nc]] );
		    // Set link
		    for( auto u : m_cliquelings[e].get_vertices() )
			lmembers[lindex[u]++] = nc;

		    // Clique is done
		    ++nc;
		} else {
		    // Singular edges
		    auto I = m_cliquelings[e].get_vertices().begin();
		    auto F = std::next( I, 1 );
		    if( sz == 2 ) {
			redges[rindex[*I]++] = *F;
			redges[rindex[*F]++] = *I;
		    } else if( sz == 3 ) {
			auto G = std::next( F, 1 );

			redges[rindex[*I]++] = *F;
			redges[rindex[*I]++] = *G;
			redges[rindex[*F]++] = *I;
			redges[rindex[*F]++] = *G;
			redges[rindex[*G]++] = *I;
			redges[rindex[*G]++] = *F;
		    } else
			assert( 0 && "NYI" );
		}
	    }
	}

	// Resetting indices
	for( VID c=0; c < nc; ++c )
	    cindex[c] = cindex[c+1];
	cindex[nc] = mc;

	EID lprev = 0;
	EID rprev = 0;
	for( VID v=0; v < n; ++v ) {
	    EID tmp = lindex[v];
	    lindex[v] = lprev;
	    lprev = tmp;
	    tmp = rindex[v];
	    rindex[v] = rprev;
	    rprev = tmp;
	}
	assert( lprev == ml );
	assert( rprev == mr );
	lindex[n] = lprev;
	rindex[n] = rprev;
    }

    void check_edges() {
	VID n = m_G.numVertices();
	EID m = m_G.numEdges();
	const EID * const index = m_G.getIndex();
	const VID * const edges = m_G.getEdges();

	VID self_edges = 0;
	for( VID v=0; v < n; ++v ) {
	    EID e = index[v];
	    EID ee = index[v+1];
	    for( ; e != ee; ++e ) {
		VID u = edges[e];

		const Cliqueling & c = find_neighbour( v, u );
		assert( c.get_candidates().size() == 0 );

		if( u == v )
		    ++self_edges;
	    }
	}

	std::cerr << "self_edges: " << self_edges << "\n";
    }

    void check_partitions() {
	VID n = m_G.numVertices();
	EID m = m_G.numEdges();
	const EID * const index = m_G.getIndex();
	const VID * const edges = m_G.getEdges();

	for( VID v=0; v < n; ++v ) {
	    if( m_singleton[v] == ~intptr_t(0) )
		continue;

	    EID e = index[v];
	    EID ee = index[v+1];
	    size_t sz = 0;
	    for( ; e != ee; ++e ) {
		intptr_t p = tree_find( m_vertex_cliques[e] );
		Cliqueling & c = m_cliquelings[p];
		if( c.empty() )
		    break;
		sz += c.get_vertices().size() - 1
		    + c.get_candidates().size();
	    }
	    assert( sz == 0 || sz == index[v+1] - index[v] );
	}
    }

    void sizes() const {
	EID m = m_G.numEdges();
	size_t sm = 0, sc = 0;
	for( EID e=0; e < m; ++e ) {
	    if( m_vertex_cliques[e] == e && !m_cliquelings[e].empty() ) {
		sm += m_cliquelings[e].get_vertices().size();
		sc += m_cliquelings[e].get_candidates().size();
	    }
	}
	std::cerr << "sizes: M: " << sm << " C: " << sc << "\n";
    }
    
    void report() const {
	std::map<size_t,size_t> histo;
	EID m = m_G.numEdges();
	size_t non_empty_candidates = 0;
	size_t unassigned_edges = 0;
	for( EID e=0; e < m; ++e ) {
	    if( m_vertex_cliques[e] == e && !m_cliquelings[e].empty() ) {
		size_t sz = m_cliquelings[e].size();
		histo[sz]++;
		if( m_cliquelings[e].get_candidates().size() != 0 )
		    ++non_empty_candidates;
	    } else
		unassigned_edges += m_cliquelings[e].get_candidates().size();
	}

	std::cerr << "histogram:\n";
	EID covered_e = 0;
	for( auto && p : histo ) {
	    size_t sz = p.first;
	    size_t edges = ( sz - 1 ) * sz;
	    std::cerr << sz << ": " << p.second
		      << " with " << edges << " edges per clique\n";
	    covered_e += edges * p.second;
	}
	std::cerr << "ignored 2-cliques: " << m_num_2cliques << "\n";
	std::cerr << "non-empty candidates: " << non_empty_candidates << "\n";
	std::cerr << "unassigned edges: " << unassigned_edges << "\n";
	std::cerr << "covered edges: " << covered_e << "\n";
	std::cerr << "edges: " << m << "\n";
    }

private:
    /**
     * With u < v, u and v are neighbours
     */
    void process_edge( VID u, VID v ) {
	Cliqueling & Cu = find_cliqueling( u, v );
	Cliqueling & Cv = find_cliqueling( v, u );

	// std::cerr << "Cu " << u << ": @" << get_id( Cu ) << " " << Cu << '\n';
	// std::cerr << "Cv " << v << ": @" << get_id( Cv ) << " " << Cv << '\n';

	// check_neighbour_count( u );
	// check_neighbour_count( v );

	if( get_id( Cu ) == get_id( Cv ) ) {
	    // std::cerr << u << " and " << v << " in same cliqueling\n";
	} else if( compatible( Cu, Cv ) ) {
	    // std::cerr << u << ',' << v << " have compatible cliquelings\n";
	    // Merge cliquelings:
	    // * Members is union of members
	    // * Candidates is intersection of candidates
	    // * Cu->candidates \ intersection and Cv->candidates \ intersection
	    //   create new cliquelings.
	    contract::vertex_set<VID> M, C, Cur, Cvr;
	    merge_union( Cu.get_vertices(), Cv.get_vertices(), M );
	    merge_candidates( M, Cu.get_candidates(), Cv.get_candidates(),
			      C, Cur, Cvr );

	    assert( M.intersection_empty( Cur ) );
	    assert( M.intersection_empty( Cvr ) );

	    // Update Cu to be the merged cliquelings; make Cv point to Cu
	    Cu.swap( M, C );
	    link_cliqueling( Cv, Cu );
	    // std::cerr << "merged: @" << get_id( Cu ) << ' ' << Cu << '\n';

	    // TODO: left-ever edges can be merged all into one partition
	    //       of neighbours (aka cliqueling) if such a cliqueling has
	    //       only one vertex in the members set
	    //       This increases chances for finding large cliques

	    // Create left-over cliqueling for Cu, if any
	    if( !Cur.empty() ) {
		distribute_candidates( M, Cur );
		// std::cerr << "left-over Cur: @" << get_id( c ) << " " << c << '\n';
	    }

	    M.swap( Cv.get_vertices() );
	    Cv.clear(); // Cv no longer necessary

	    // Create left-over cliqueling for Cv, if any
	    if( !Cvr.empty() ) {
		distribute_candidates( M, Cvr );
		// std::cerr << "left-over Cvr: @" << get_id( c ) << " " << c << '\n';
	    }
	} else {
	    // std::cerr << "edge " << u << ',' << v << " is 2-clique\n";
	    ++m_num_2cliques;
	}

	// check_neighbour_count( u );
	// check_neighbour_count( v );
    }

    bool compatible( const Cliqueling & Cu, const Cliqueling & Cv ) const {
	if( Cu.members_contained( Cv ) ) {
	    assert( Cv.members_contained( Cu ) );
	    return true;
	} else {
	    assert( !Cv.members_contained( Cu ) );
	    return false;
	}
    }

    Cliqueling & find_cliqueling( VID v, VID ngh ) {
	const EID * const index = m_G.getIndex();
	const VID * const edges = m_G.getEdges();
	EID e=index[v], ee=index[v+1];
	for( ; e != ee; ++e ) {
	    intptr_t p = tree_find( m_vertex_cliques[e] );
	    Cliqueling & c = m_cliquelings[p];
	    if( c.contains( ngh ) )
		return c;
	    if( c.empty() )
		break;
	}
	assert( e < ee ); // at most as many cliquelings as neighbours
	assert( e == index[v] ); // 1st cliqueling contains all ngh

	// std::cerr << "Creating first cliqueling v=" << v << " e=" << e << "\n";
	Cliqueling & c
	    = allocate_cliqueling( e, v, &edges[index[v]], &edges[ee] );
	m_singleton[v] = e;
	return c;
    }

    const Cliqueling & find_neighbour( VID v, VID ngh ) {
	const EID * const index = m_G.getIndex();
	const VID * const edges = m_G.getEdges();
	EID e=index[v], ee=index[v+1];
	for( ; e != ee; ++e ) {
	    intptr_t p = tree_find( m_vertex_cliques[e] );
	    Cliqueling & c = m_cliquelings[p];
	    if( c.get_vertices().contains( ngh ) )
		return c;
	    if( c.empty() )
		break;
	}
	assert( 0 && "checking: neighbour not found" );
    }

    void check_neighbour_count( VID v ) {
	const EID * const index = m_G.getIndex();
	const VID * const edges = m_G.getEdges();
	EID e=index[v], ee=index[v+1];
	VID cnt = 0;
	contract::vertex_set<VID> all;
	std::cerr << "*** PARTITIONS for " << v << "\n";
	for( ; e != ee; ++e ) {
	    intptr_t p = tree_find( m_vertex_cliques[e] );
	    Cliqueling & c = m_cliquelings[p];
	    if( c.empty() )
		break;
	    assert( c.get_vertices().contains( v ) );
	    cnt += c.get_vertices().size() - 1 + c.get_candidates().size();

	    std::cerr << "   " << c << "\n";

	    contract::vertex_set<VID> tmp1, tmp2;
	    merge_union( c.get_vertices(), c.get_candidates(), tmp1 );
	    merge_union( tmp1, all, tmp2 );
	    all.swap( tmp2 );
	}
	assert( cnt == index[v+1] - index[v] );
	assert( all.size() == cnt+1 );
	std::cerr << "*** PARTITIONS for " << v << " done\n";
    }

    // Merge the candidates in C with the candidate sets of every v in M
    // Basically, returning the vertices to their sets. Some cliques may be
    // possible between them, needs to be figured out. Retry in a second loop
    // over all edges?
    void
    distribute_candidates( const contract::vertex_set<VID> & M,
			   contract::vertex_set<VID> & C ) {
	for( auto && v : M ) {
	    merge_neighbours( v, C );
	}
    }

    // A cliqueling with |M={v}| == 1, merge the candidates in C with other
    // candidate sets.
    void
    merge_neighbours( VID v, contract::vertex_set<VID> & C ) {
	const EID * const index = m_G.getIndex();
	const VID * const edges = m_G.getEdges();

	// A quick fix to avoid linear search: cache the location of the
	// set with |M|==1 (there is only one) so that we can find in O(1)
	if( m_singleton[v] != ~intptr_t(0) ) {
	    EID e = m_singleton[v];
	    intptr_t p = tree_find( m_vertex_cliques[e] );
	    Cliqueling & c = m_cliquelings[p];
	    size_t sz = c.get_vertices().size();
	    if( sz == 1 ) {
		assert( v == *c.get_vertices().begin() );
		c.get_candidates().add( C );
		return;
	    }	
	}
	
	// Linear search
	EID e=index[v], ee=index[v+1];
	for( ; e != ee; ++e ) {
	    intptr_t p = tree_find( m_vertex_cliques[e] );
	    Cliqueling & c = m_cliquelings[p];
	    size_t sz = c.get_vertices().size();
	    if( sz == 0 ) {
		if( p == e )
		    break;
	    } else if( sz == 1 ) {
		assert( v == *c.get_vertices().begin() );
		c.get_candidates().add( C );
		m_singleton[v] = e;
		return;
	    }	
	}

	// There are no cliquelings where the member set == { v }
	// create_cliqueling( v, C );
	assert( e < ee );
	assert( m_cliquelings[e].empty() );
	allocate_cliqueling( e, v, C );
	m_singleton[v] = e;
    }

    template<typename... Args>
    Cliqueling & allocate_cliqueling( EID e, Args... args ) {
	assert( m_vertex_cliques[e] == e );
	assert( m_cliquelings[e].empty() );
	Cliqueling c( std::forward<Args>( args )... );
	m_cliquelings[e].swap( c );
	return m_cliquelings[e];
    }

    void link_cliqueling( const Cliqueling & from, const Cliqueling & to ) {
	// union
	intptr_t f = get_id( from );
	intptr_t t = get_id( to );
	m_vertex_cliques[f] = t;
    }

    intptr_t get_id( const Cliqueling & c ) const {
	return &c - &m_cliquelings[0];
    }
    
    intptr_t tree_find( intptr_t p ) {
/*
	// intptr_t p = q;
	while( p != m_vertex_cliques[p] ) {
	    assert( m_cliquelings[p].get_vertices().empty() );
	    assert( m_cliquelings[p].get_candidates().empty() );
	    p = m_vertex_cliques[p];
	}
	// m_vertex_cliques[q] = p;
	return p;
*/
	intptr_t u = p;
	while( true ) {
	    intptr_t v = m_vertex_cliques[u];
	    intptr_t w = m_vertex_cliques[v];
	    if( v == w )
		return v;
	    else {
		m_vertex_cliques[u] = w;
		// __sync_val_compare_and_swap( &m_vertex_cliques[u], v, w );
		u = v;
	    }
	}
    }

    // It is assumed that Mu and Mv have empty intersection (?)
    void
    merge_union( const contract::vertex_set<VID> & Mu,
		 const contract::vertex_set<VID> & Mv,
		 contract::vertex_set<VID> & M ) {
	M.resize( Mu.size() + Mv.size() );
	const VID * bu = Mu.begin();
	const VID * eu = Mu.end();
	const VID * bv = Mv.begin();
	const VID * ev = Mv.end();
	VID * p = M.begin();
	while( bu != eu && bv != ev ) {
	    if( *bu == *bv ) {
		// std::cerr << "NOTE: member lists have common element: "
		// << *bu << "\n";
		*p++ = *bu;
		++bu;
		++bv;
	    } else if( *bu < *bv )
		*p++ = *bu++;
	    else
		*p++ = *bv++;
	}
	while( bu != eu )
	    *p++ = *bu++;
	while( bv != ev )
	    *p++ = *bv++;

	// Shrink set if there were common elements
	M.resize( p - M.begin() );
    }

    // C = ( Cu intersect Cv ) \ M
    // Cur = Cu \ C
    // Cvr = Cv \ C
    // such that C union Cur union Cvr = ( Cu union Cv ) \ M
    //
    // TODO: could also decide to move candidates in C to both
    //       Cur and Cvr (possibly in such a way to keep the largest cliqueling)
    //   + pre-calculate sizes of C, Cur, Cvr
    //   + decide on actions depending on those numbers
    //   + avoid 2-cliques (don't build them)
    //     -> Theorem? if our technique creates M=2-clique w C=empty
    //                 then that is the best possible choice anyway???
    //   + don't merge if C < Cur or Cvr
    //   + if we do that, will we need multiple passes over edge list?
    void
    merge_candidates( const contract::vertex_set<VID> & M,
		      const contract::vertex_set<VID> & Cu,
		      const contract::vertex_set<VID> & Cv,
		      contract::vertex_set<VID> & C,
		      contract::vertex_set<VID> & Cur,
		      contract::vertex_set<VID> & Cvr ) {
	// Worst-case size for each output vertex set
	size_t csz = std::min( Cu.size(), Cv.size() );
	C.resize( csz );
	Cur.resize( Cu.size() );
	Cvr.resize( Cv.size() );

	const VID * bu = Cu.begin();
	const VID * eu = Cu.end();
	const VID * bv = Cv.begin();
	const VID * ev = Cv.end();
	const VID * bm = M.begin();
	const VID * em = M.end();
	VID * p = C.begin();
	VID * pur = Cur.begin();
	VID * pvr = Cvr.begin();
	while( bu != eu && bv != ev ) {
	    if( *bu == *bv ) {
		// Check if element is in M
		while( bm != em && *bm < *bu ) // could use lower_bound
		    ++bm; 
		if( bm != em && *bm == *bu ) { // drop element; should not exist
		    /*
		    std::cerr << "NOTE: candidate lists have common element "
			      << "with members-union: " << *bu << " with:\n"
			      << "M: " << M << "\n"
			      << "Cu: " << Cu << "\n"
			      << "Cv: " << Cv << "\n";
		    */
		} else
		    *p++ = *bu; // in intersection and not in M

		++bu;
		++bv;
	    } else if( *bu < *bv ) {
		while( bm != em && *bm < *bu )
		    ++bm; 
		if( bm != em && *bm == *bu ) {
		    // drop - looking for intersection
		    ++bu;
		} else
		    *pur++ = *bu++;
	    } else {
		while( bm != em && *bm < *bv )
		    ++bm; 
		if( bm != em && *bm == *bv ) {
		    // drop - looking for intersection
		    ++bv;
		} else
		    *pvr++ = *bv++;
	    }
	}
	while( bu != eu ) {
	    while( bm != em && *bm < *bu )
		++bm; 
	    if( bm != em && *bm == *bu ) {
		// drop - looking for intersection
		++bu;
	    } else
		*pur++ = *bu++;
	}
	while( bv != ev ) {
	    while( bm != em && *bm < *bv )
		++bm; 
	    if( bm != em && *bm == *bv ) {
		// drop - looking for intersection
		++bv;
	    } else
		*pvr++ = *bv++;
	}

	// Shrink set if there were common elements
	C.resize( p - C.begin() );
	Cur.resize( pur - Cur.begin() );
	Cvr.resize( pvr - Cvr.begin() );
    }

private:
    const GraphCSx & m_G;
    const VID * const m_order;
    const VID * const m_rev_order;
    mm::buffer<intptr_t> m_vertex_cliques;
    mm::buffer<intptr_t> m_singleton;
    std::vector<Cliqueling> m_cliquelings;
    size_t m_num_2cliques;
};

void
sort_order( VID * order, VID * rev_order,
	    const VID * const coreness,
	    VID n,
	    VID K ) {
    VID * histo = new VID[K+1];
    std::fill( &histo[0], &histo[K+1], 0 );

    // Histogram
    for( VID v=0; v < n; ++v ) {
	VID c = coreness[v];
	assert( c <= K );
	histo[K-c]++;
    }

    // Prefix sum
    VID sum = sequence::plusScan( histo, histo, K+1 );

    // Place in order
    for( VID v=0; v < n; ++v ) {
	VID c = coreness[v];
	VID pos = histo[K-c]++;
	order[pos] = v;
	rev_order[v] = pos;
    }

    delete[] histo;
}

class GraphBuilder {
public:
    template<short VarName>
    GraphBuilder( const partitioner & part, const GraphCSx & G, frontier & f,
		  api::vertexprop<VID,VID,VarName> & dcount )
	: g2s( G.numVertices(), numa_allocation_interleaved() ),
	  s2g( f.nActiveVertices(), numa_allocation_interleaved() ) {
	VID n = G.numVertices();
	EID m = G.numEdges();
	const EID * const gindex = G.getIndex();
	const VID * const gedges = G.getEdges();
	VID ns = f.nActiveVertices();

	assert( G.isSymmetric() );

	// Initialize
	std::fill( &g2s[0], &g2s[n], (VID)~0 );
	
	// create array with vertex IDs, will be sorted
	frontier fs = f.copySparse( part );
	const VID * const s = fs.getSparse();
	assert( std::is_sorted( &s[0], &s[ns] ) && "list must be sorted" );
	mm::buffer<EID> idx( ns+1, numa_allocation_interleaved() );
	parallel_loop( (VID)0, ns, [&]( VID i ) {
	    s2g[i] = s[i]; // copy
	    idx[i] = dcount[s[i]]; // copy selected values
	    g2s[s[i]] = i; // reverse map
	} );

	parallel_loop( (VID)0, ns, [&]( VID v ) {
	    assert( v == g2s[s2g[v]] );
	} );

	// Prefix sum of degrees becomes index array
	EID ms = sequence::plusScan( idx.get(), idx.get(), ns );
	idx[ns] = ms;

	// Construct selected graph
	new ( &S ) GraphCSx( ns, ms, -1, true, false );
	EID * index = S.getIndex();
	VID * edges = S.getEdges();

	// Copy - TODO: avoid
	std::copy( &idx[0], &idx[ns+1], index );

	if( ms == 0 ) {
	    S.build_degree();
	    idx.del();
	    return;
	}

	// Prepare lookup
	f.toDense<bool>( part );
	const bool * const fb = f.getDense<bool>();

	// Place neighbours
	parallel_loop( (VID)0, ns, [&]( VID v ) {
	    VID vg = s2g[v];
	    EID se = index[v];
	    for( EID e=gindex[vg], ee=gindex[vg+1]; e < ee; ++e ) {
		VID u = gedges[e];
		if( fb[u] ) // or g2s[u] != ~0
		    edges[se++] = g2s[u];
	    }
	    assert( se == index[v+1] && "error in neighbour count" );
	} );

	S.build_degree();
	idx.del();
	fs.del();
    }
    ~GraphBuilder() {
	S.del();
	g2s.del();
	s2g.del();
    }

    void remap( contract::vertex_set<VID> & s ) {
	for( auto I=s.begin(), E=s.end(); I != E; ++I ) {
	    *I = s2g[*I];
	}
    }

    const VID * get_g2s() const { return g2s.get(); }
    const VID * get_s2g() const { return s2g.get(); }
    const GraphCSx & get_graph() const { return S; }

private:
    GraphCSx S;
    mm::buffer<VID> g2s;
    mm::buffer<VID> s2g;
};

template<typename GraphType2>
GraphBuilder
prune_graph_degree( const GraphType2 & G,
		    VID target_omega,
		    const KCv<GraphType2> & kcore ) {
    const partitioner &part = G.get_partitioner();
    VID n = G.numVertices();
    EID m = G.numEdges();

    api::vertexprop<VID,VID,var_dcount> dcount( part, "dcount" );

    // Vertex property with coreness
    auto & coreness = kcore.getCoreness();

    make_lazy_executor( part )
	.vertex_map( [&]( auto v ) { return dcount[v] = _0; } )
	.materialize();

    // Assumes target_omega >= 2
    frontier sel;
    api::edgemap(
	G,
	// Only do this for relevant vertices
	api::filter( api::dst, api::strong,
		     [&]( auto d ) {
			 return coreness[d] > _c( target_omega-2 );
		     } ),
	// Select all vertices with sufficiently high coreness
	api::record( sel, api::method,
		     [&]( auto v ) {
			 return coreness[v] > _c( target_omega-2 );
		     },
		     api::strong ),
	// Count number of neighbours with sufficiently high coreness
	// We know it is at least dcount[d], but we need an accurate count
	// to assist the graph builder.
	api::relax( [&]( auto s, auto d, auto e ) {
	    return dcount[d] += _p( _1(dcount[d]),
				    coreness[s] > _c( target_omega-2 ) );
	} )
	)
	.materialize();

    assert( sel.nActiveVertices() > 0 && "oops - pruned too hard" );

    GraphBuilder CB( part, G.getCSR(), sel, dcount );

    dcount.del();

    return CB;
}

template<short VarName1, typename Iter>
bool is_not_covered( 
    GraphCSRAdaptor & GA,
    VID v,
    Iter && I, Iter && E,
    api::edgeprop<VID,EID,VarName1> & covered ) {
    const EID * const index = GA.getCSR().getIndex();
    const VID * const edges = GA.getCSR().getEdges();
    const VID * const nb = &edges[index[v]];
    const VID * const ne = &edges[index[v+1]];
    return std::all_of( I, E, [&]( auto u ) {
	const VID * ep = std::lower_bound( ne, nb, u );
	EID e = ep - edges;
	return covered[e] == 0;
    } );
}

template<short VarName1>
bool is_not_covered( 
    GraphCSRAdaptor & GA,
    VID v,
    contract::vertex_set<VID> & R,
    api::edgeprop<VID,EID,VarName1> & covered ) {
    return is_not_covered( GA, v, R.begin(), R.end(), covered );
}
    

template<short VarName1, short VarName2>
void BronKerbosch( 
    GraphCSRAdaptor & GA,
    VID K,
    int depth,
    api::edgeprop<VID,EID,VarName1> & covered,
    api::vertexprop<VID,VID,VarName2> & counts,
    std::vector<contract::vertex_set<VID>> & cliques,
    contract::vertex_set<VID> & R,
    contract::vertex_set<VID> & P,
    contract::vertex_set<VID> & X ) {

    // std::cerr << "BK R=" << R << " P=" << P << " K=" << K << "\n";

    if( R.size() + P.size() < K )
	return;

    if( P.size() == 0 ) {
	// Maximal clique?
	// if( X.size() == 0 ) {
	    std::for_each( R.begin(), R.end(), [&]( VID v ) {
		counts.get_ptr()[v]++;
	    } );
	    cliques.push_back( R );
	    // std::cerr << "found clique in R\n";
	    return;
	// }

	// Already visited before
	return;
    }

    const EID * const index = GA.getCSR().getIndex();
    const VID * const edges = GA.getCSR().getEdges();

    for( VID I=0, E=P.size(); I != E; ++I ) {
	VID v = P.get( I );
	// some edge already covered?
	if( !is_not_covered( GA, v, R, covered ) ) {
	    // std::cerr << "v=" << v << " is covered wrt " << R << "\n";
	    continue;
	}

	R.push( v );
	contract::vertex_set<VID> Pv
	    = P.intersect( &edges[index[v]], index[v+1] - index[v] );
	contract::vertex_set<VID> Xv
	    = X.intersect( &edges[index[v]], index[v+1] - index[v] );
	
	BronKerbosch( GA, K, depth+1, covered, counts, cliques, R, Pv, Xv );
	R.pop();
	P.remove( v );
	--I;
	--E;
	X.push( v );
    }
}

template<short VarName1, short VarName2, short VarName3>
void explore_cliques(
    GraphCSRAdaptor & GA,
    VID v,
    VID K,
    const api::vertexprop<VID,VID,VarName1> & coreness,
    api::edgeprop<VID,EID,VarName2> & covered,
    api::vertexprop<VID,VID,VarName3> & counts,
    std::vector<contract::vertex_set<VID>> & cliques ) {

    const partitioner & part = GA.get_partitioner();
    const VID n = GA.numVertices();
    const EID m = GA.numEdges();
    const EID * const index = GA.getCSR().getIndex();
    const VID * const edges = GA.getCSR().getEdges();

    // Pick up the relevant neighbours. N will be in sort order because
    // neighbour lists are.
    contract::vertex_set<VID> R, P, X;
    for( EID e=index[v], ee=index[v+1]; e != ee; ++e ) {
	VID u = edges[e];
	if( coreness[u] >= K && covered[e] == 0 ) {
	    P.push( u );
	    // if( u < v )
	    // X.push( u );
	}
    }

    R.push( v );
    
    BronKerbosch( GA, K, 1, covered, counts, cliques, R, P, X );
}

template<short VarName1>
auto
characterise(
    const contract::vertex_set<VID> & s,
    api::vertexprop<VID,VID,VarName1> & counts ) {
    VID none = 0, total = 0;
    std::for_each( s.begin(), s.end(),
		   [&]( VID v ) {
		       if( counts[v] == 1 )
			   ++none;
		       total += counts[v];
		   } );
    return std::make_pair( none, total );
}

template<short VarName1>
bool
is_possible(
    GraphCSRAdaptor & GA,
    const contract::vertex_set<VID> & s,
    api::edgeprop<VID,EID,VarName1> & covered ) {

    auto I = s.begin();
    auto E = s.end();

    while( I != E ) {
	VID v = *I;
	++I;
	if( !is_not_covered( GA, v, I, E, covered ) )
	    return false;
    }
    return true;
}

template<short VarName1>
void
mark_edges(
    GraphCSRAdaptor & GA,
    const contract::vertex_set<VID> & s,
    api::edgeprop<VID,EID,VarName1> & covered ) {

    const EID * const index = GA.getCSR().getIndex();
    const VID * const edges = GA.getCSR().getEdges();

    std::for_each( s.begin(), s.end(), [&]( VID v ) {
	const VID * const nb = &edges[index[v]];
	const VID * const ne = &edges[index[v+1]];
	std::for_each( s.begin(), s.end(), [&]( VID u ) {
	    if( u != v ) {
		const VID * ep = std::lower_bound( ne, nb, u );
		EID e = ep - edges;
		covered.get_ptr()[e] = 1;
	    }
	} );
    } );
}

template<short VarName1, short VarName2, short VarName3>
void find_cliques_of_size(
    GraphCSRAdaptor & GA,
    VID K,
    const api::vertexprop<VID,VID,VarName1> & coreness,
    api::edgeprop<VID,EID,VarName2> & covered,
    api::vertexprop<VID,VID,VarName3> & counts,
    std::vector<contract::vertex_set<VID>> & all_cliques ) {


    const partitioner & part = GA.get_partitioner();
    const VID n = GA.numVertices();
    const EID m = GA.numEdges();

    frontier ftrue = frontier::all_true( n, m );

    // First, select all vertices with coreness equal to K.
    // If wanting to have K=multiple of 4, could expand to range [K,K+4)
    frontier F;
    make_lazy_executor( part )
	.vertex_filter( GA, ftrue, F,
			[&]( auto v ) { return coreness[v] == _c( K ); } )
	.materialize();

    if( F.nActiveVertices() == 0 ) {
	std::cerr << "No vertices present with coreness " << K << "\n";
	return;
    }

    // Search cliques
    F.toSparse( part );
    const VID * const list = F.getSparse();
    const VID nlist = F.nActiveVertices();

    std::cerr << "exploring cliques...\n";
    std::vector<contract::vertex_set<VID>> cliques;
    
    for( VID i=0; i < nlist; ++i ) {
	VID v = list[i];
	explore_cliques( GA, v, K, coreness, covered, counts, cliques );
    }

    std::cerr << "sorting " << cliques.size() << " cliques...\n";
    // Sort cliques in order of priority, considering in first instance
    // those with vertices with only a single clique found, and
    // those with minimum total clique counts as secondary sort criterion.
    std::sort( cliques.begin(), cliques.end(),
	       [&]( contract::vertex_set<VID> & a,
		    contract::vertex_set<VID> & b ) {
		   auto ca = characterise( a, counts );
		   auto cb = characterise( b, counts );
		   if( ca.first < cb.first )
		       return true;
		   else if( ca.first > cb.first )
		       return false;
		   if( ca.second < cb.second )
		       return true;
		   else if( ca.second > cb.second )
		       return false;
		   else
		       return *a.begin() < *b.begin();
	       } );
    
    // Pick cliques
    std::cerr << "marking cliques...\n";
    for( auto && c : cliques ) {
	if( is_possible( GA, c, covered ) ) {
	    mark_edges( GA, c, covered );
	    all_cliques.push_back( c );
	    std::cerr << "C: " << c << "\n";
	}
    }
}

template<short VarName1, short VarName2, short VarName3>
auto
find_cliques(
    GraphCSRAdaptor & GA,
    VID K,
    const api::vertexprop<VID,VID,VarName1> & coreness,
    api::edgeprop<VID,EID,VarName2> & covered,
    api::vertexprop<VID,VID,VarName3> & counts ) {

    EID n = GA.numVertices();
    std::vector<contract::vertex_set<VID>> cliques;

    for( VID k=4; k <= K; ++k ) {
	std::cerr << "\n *** Cliques of size " << k << " out of "
		  << K << "\n\n";
	std::fill( &counts.get_ptr()[0], &counts.get_ptr()[n], 0 );
	find_cliques_of_size( GA, k, coreness, covered, counts, cliques );
    }

    return cliques;
}

template<short VarName>
void
write_cliques( const GraphCSRAdaptor & GA,
	       const std::vector<contract::vertex_set<VID>> & clist,
	       CliqueList<VID,EID,VID> & cliques,
	       CompressedList<VID,EID> & remainder,
	       const api::edgeprop<VID,EID,VarName> & covered,
	       VID min_size ) {
    VID n = GA.numVertices();
    EID m = GA.numEdges();
    const EID * const gindex = GA.getCSR().getIndex();
    const VID * const gedges = GA.getCSR().getEdges();
    EID * cindex = cliques.get_corpus().get_index();
    VID * cmembers = cliques.get_corpus().get_members();
    EID * lindex = cliques.get_links().get_index();
    VID * lmembers = cliques.get_links().get_members();
    EID * rindex = remainder.get_index();
    VID * redges = remainder.get_members();
    EID nc = 0;
    EID nm = 0;
    EID ne = 0;

    std::fill( &cindex[0], &cindex[cliques.get_num_cliques()], 0 );
    std::fill( &lindex[0], &lindex[n], 0 );
    std::fill( &rindex[0], &rindex[n], 0 );

    assert( min_size == 4 );

    // First, a counting exercise
    for( auto && c : clist ) {
	VID sz = c.size();
	if( sz >= min_size ) {
	    // Another clique, to count offset in members list
	    cindex[nc++] = sz;

	    // All of the vertices have a link to a clique
	    for( auto u : c )
		lindex[u]++;
	} else {
	    // Singular edges
	    auto I = c.begin();
	    auto F = std::next( I, 1 );
	    if( sz == 2 ) {
		rindex[*I]++;
		rindex[*F]++;
	    } else if( sz == 3 ) {
		auto G = std::next( F, 1 );

		rindex[*I] += 2;
		rindex[*F] += 2;
		rindex[*G] += 2;
	    } else
		assert( 0 && "NYI" );
	}
    }

    // We haven't touched upon the uncovered edges yet...
    for( VID v=0; v < n; ++v ) {
	for( EID e=gindex[v], ee=gindex[v+1]; e != ee; ++e ) {
	    if( covered[e] == 0 )
		rindex[v]++;
	}
    }

    assert( nc == cliques.get_num_cliques() );

    // Sum up the counts
    EID mc = sequence::plusScan( cindex, cindex, nc );
    assert( mc == cliques.get_num_members() );
    cindex[nc] = mc;

    EID ml = sequence::plusScan( lindex, lindex, n );
    assert( ml == cliques.get_num_links() );
    lindex[n] = ml;

    EID mr = sequence::plusScan( rindex, rindex, n );
    assert( mr == remainder.get_num_members() );
    rindex[n] = mr;

    // Place data
    nc = 0;
    for( auto && c : clist ) {
	size_t sz = c.size();
	if( sz >= min_size ) {
	    // Fill in clique info
	    std::copy( c.begin(), c.end(), &cmembers[cindex[nc]] );
	    // Set link
	    for( auto u : c )
		lmembers[lindex[u]++] = nc;

	    // Clique is done
	    ++nc;
	} else {
	    // Singular edges
	    auto I = c.begin();
	    auto F = std::next( I, 1 );
	    if( sz == 2 ) {
		redges[rindex[*I]++] = *F;
		redges[rindex[*F]++] = *I;
	    } else if( sz == 3 ) {
		auto G = std::next( F, 1 );

		redges[rindex[*I]++] = *F;
		redges[rindex[*I]++] = *G;
		redges[rindex[*F]++] = *I;
		redges[rindex[*F]++] = *G;
		redges[rindex[*G]++] = *I;
		redges[rindex[*G]++] = *F;
	    } else
		assert( 0 && "NYI" );
	}
    }

    // uncovered edges
    for( VID v=0; v < n; ++v ) {
	for( EID e=gindex[v], ee=gindex[v+1]; e != ee; ++e ) {
	    if( covered[e] == 0 )
		redges[rindex[v]++] = gedges[e];
	}
    }

    // Resetting indices
    VID cprev = 0;
    for( VID c=0; c < nc; ++c ) {
	VID tmp = cindex[c];
	cindex[c] = cprev;
	cprev = tmp;
    }
    cindex[nc] = mc;

    EID lprev = 0;
    EID rprev = 0;
    for( VID v=0; v < n; ++v ) {
	EID tmp = lindex[v];
	lindex[v] = lprev;
	lprev = tmp;
	tmp = rindex[v];
	rindex[v] = rprev;
	rprev = tmp;
    }
    assert( lprev == ml );
    assert( rprev == mr );
    lindex[n] = lprev;
    rindex[n] = rprev;
}


int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    bool symmetric = P.getOptionValue("-s");
    bool binary = P.getOptionValue("-b");
    VID min_size = P.getOptionLongValue( "-minsize", 4 );

    const char * ifile = P.getOptionValue( "-i" );
    const char * odir = P.getOptionValue( "-o" );

    GraphCSx G( ifile, -1, symmetric );

    std::cerr << "Read graph.\n";

    VID n = G.numVertices();
    EID m = G.numEdges();

    assert( G.isSymmetric() );
    std::cerr << "Undirected graph: n=" << n << " m=" << m << std::endl;

    std::cerr << "Calculating coreness...\n";
    GraphCSRAdaptor GA( G, 256 );
    KCv<GraphCSRAdaptor> kcore( GA, P );
    kcore.run();

    const partitioner & part = GA.get_partitioner();
    auto & coreness = kcore.getCoreness();
    api::edgeprop<VID,EID,var_covered> covered( part, "edge covered" );
    std::fill( &covered.get_ptr()[0], &covered.get_ptr()[m], 0 );

    api::vertexprop<VID,VID,var_counts> counts( part, "clique counts" );
    std::fill( &counts.get_ptr()[0], &counts.get_ptr()[n], 0 );

    // TODO: before anything else, could try to find a large clique
    // using a hypothesis of low clique gap width.

    // Find cliques of specific size
    auto cliques
	 = find_cliques( GA, kcore.getLargestCore(),
			 coreness, covered, counts );

    std::cerr << "Number of cliques found: " << cliques.size() << "\n";

    // Get statistics on cliques and covered edges
    VID num_cliques = 0;
    EID num_members = 0;
    EID num_edges = 0;

    for( auto && c : cliques ) {
	++num_cliques;
	EID sz = c.size();
	num_members += sz;
	num_edges += ( sz - 1 ) * sz;
    }

    float pct_edges = float(num_edges)/float(m) * 100.0;

    std::cerr << "cliques: " << num_cliques
	      << "\nmembers: " << num_members
	      << "\nedges: " << num_edges
	      << "\npercentage edges in cliques: " << pct_edges
	      << "%\n";

    GraphPDG<VID,EID,VID> PDG( n, m, num_cliques, num_members, num_edges );

    write_cliques( GA, cliques, PDG.get_cliques(), PDG.get_edges(),
		   covered, min_size );

    return 0;
}
