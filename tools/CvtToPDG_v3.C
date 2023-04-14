// -*- c++ -*-

// TODO:
// * Cut out induced neighbourhood before deciding dense - done
// * Buss kernel
// * DenseMatrix at multiple widths (512, 64, 32)
// * time-out
// * online machine learning
// * parallelisation - 3-hop mis

// A Pattern Decomposed Graph
#include <mutex>

#include "graptor/graptor.h"
#include "graptor/api.h"

#include "graptor/graph/contract/vertex_set.h"
#include "graptor/graph/GraphCSx.h"
#include "graptor/graph/GraphPDG.h"
#include "graptor/graph/partitioning.h"

#include "graptor/graph/simple/csx.h"
#include "graptor/graph/simple/dicsx.h"

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
    var_priority = var_kc_num + 3
};

class CliqueLister {
public:
    // Estimate max number of cliques as m/12 as we are accepting only
    // cliques of size at least 4 (12 edges per clique) and any edge
    // can be a member of at most one clique.
    CliqueLister( EID m )
	: m_list( m/12, m, numa_allocation_interleaved() ),
	  m_max_elm( m ),
	  m_next_id( 0 ),
	  m_next_pos( 0 ),
	  m_edges( 0 ) { }

    std::pair<VID,VID *> allocate_clique( VID size ) {
	std::lock_guard<std::mutex> guard( m_lock );
	EID * idx = m_list.get_index();
	VID id = m_next_id++;
	idx[id] = m_next_pos;
	VID * lst = &m_list.get_members()[m_next_pos];
	m_next_pos += size;
	m_edges += EID( size ) * EID( size - 1 );
	return std::make_pair( id, lst );
    }

    void finalize() {
	// Terminate index array with n+1th element
	m_list.get_index()[m_next_id] = m_next_pos;
	// Trim to actual numbers required
	m_list.resize( m_next_id, m_next_pos );
    }
    
    EID * get_index() { return m_list.get_index(); }
    VID * get_members() { return m_list.get_members(); }
    VID get_num_cliques() const { return m_next_id; }
    EID get_num_members() const { return m_next_pos; }
    EID get_num_clique_edges() const { return m_edges; }

    const CompressedList<VID,EID> & get_list() const { return m_list; }
    
private:
    CompressedList<VID,EID> m_list;
    EID m_max_elm;
    VID m_next_id;
    EID m_next_pos;
    EID m_edges;
    std::mutex m_lock;
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

/*
    void intersect_candidates( Cliqueling * cl,
			       contract::vertex_set<VID> & C,
			       contract::vertex_set<VID> & Cur,
			       contract::vertex_set<VID> & Cvr ) {
	m_candidates.intersect( cl->m_candidates, C, Cur, Cvr );
    }
*/

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

	// assert( G.isSymmetric() );

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
	new ( &S ) graptor::graph::GraphCSx<VID,EID>( ns, ms ); //, -1, true, false );
	EID * index = S.getIndex();
	VID * edges = S.getEdges();

	// Copy - TODO: avoid
	std::copy( &idx[0], &idx[ns+1], index );

	if( ms == 0 ) {
	    // S.build_degree();
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

	// S.build_degree();
	idx.del();
	fs.del();
    }
    GraphBuilder( const graptor::graph::GraphCSx<VID,EID> & G, const bool * const f, VID ns )
	: g2s( G.numVertices(), numa_allocation_interleaved() ),
	  s2g( ns, numa_allocation_interleaved() ) {
	VID n = G.numVertices();
	EID m = G.numEdges();
	const EID * const gindex = G.getIndex();
	const VID * const gedges = G.getEdges();

	// assert( G.isSymmetric() );

	// Initialize
	std::fill( &g2s[0], &g2s[n], (VID)~0 );
	
	// create array with vertex IDs, will be sorted
	mm::buffer<EID> idx( ns+1, numa_allocation_interleaved() );
	for( VID i=0, ni=0; i < n; ++i ) {
	    if( f[i] ) {
		s2g[ni] = i; // copy
		g2s[i] = ni; // reverse map
		VID deg = 0;
		for( EID e=gindex[i], ee=gindex[i+1]; e != ee; ++e ) {
		    VID u = gedges[e];
		    if( f[u] )
			++deg;
		}
		idx[ni] = deg;
		++ni;
	    }
	}

	// Prefix sum of degrees becomes index array
	EID ms = sequence::plusScan( idx.get(), idx.get(), ns );
	idx[ns] = ms;

	// Construct selected graph
	new ( &S ) graptor::graph::GraphCSx<VID,EID>( ns, ms ); // , -1, true, false );
	EID * index = S.getIndex();
	VID * edges = S.getEdges();

	// Copy - TODO: avoid
	std::copy( &idx[0], &idx[ns+1], index );

	if( ms == 0 ) {
	    // S.build_degree();
	    idx.del();
	    return;
	}

	// Place neighbours
	for( VID vs=0; vs < ns; ++vs ) {
	    VID vg = s2g[vs];
	    EID se = idx[vs];
	    for( EID e=gindex[vg], ee=gindex[vg+1]; e != ee; ++e ) {
		VID u = gedges[e];
		if( f[u] )
		    edges[se++] = g2s[u];
	    }
	    assert( se == index[vs+1] && "error in neighbour count" );
	}

	// S.build_degree();
	idx.del();
    }

    ~GraphBuilder() {
	// S.del();
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
    const graptor::graph::GraphCSx<VID,EID> & get_graph() const { return S; }

private:
    graptor::graph::GraphCSx<VID,EID> S;
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

void
mc_iterate( const graptor::graph::GraphCSx<VID,EID> & G,
	    contract::vertex_set<VID> & R,
	    contract::vertex_set<VID> & P,
	    int depth,
	    contract::vertex_set<VID> & mc ) {
    if( P.size() == 0 ) {
	if( R.size() > mc.size() )
	    mc = R;
	return;
    }
    if( R.size() + P.size() < mc.size() ) // fail to improve
	return;

    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();

    for( auto I=P.begin(), E=P.end(); I != E; ++I ) {
	VID u = *I;

	VID deg = index[u+1] - index[u];
	VID sz = std::min( deg, P.size() );
	contract::vertex_set<VID> Pv( sz );
	Pv.resize( sz );
	Pv.resize( contract::detail::intersect(
		       I, (VID)std::distance( I, E ), // prune visited elm of P
		       &edges[index[u]], deg,
		       Pv.begin() ) );
	// contract::vertex_set<VID> Pv
	// = P.intersect( &edges[index[u]], index[u+1] - index[u] );
	R.push( u );
	mc_iterate( G, R, Pv, depth+1, mc );
	R.pop();
    }
}

void
mc_iterate( const graptor::graph::GraphDoubleIndexCSx<VID,EID> & G,
	    contract::vertex_set<VID> & R,
	    contract::vertex_set<VID> & P,
	    int depth,
	    contract::vertex_set<VID> & mc ) {
    if( P.size() == 0 ) {
	if( R.size() > mc.size() )
	    mc = R;
	return;
    }
    if( R.size() + P.size() < mc.size() ) // fail to improve
	return;

    const EID * const bindex = G.getBeginIndex();
    const EID * const eindex = G.getEndIndex();
    const VID * const edges = G.getEdges();

    for( auto I=P.begin(), E=P.end(); I != E; ++I ) {
	VID u = *I;

	VID deg = eindex[u] - bindex[u];
	VID sz = std::min( deg, P.size() );
	contract::vertex_set<VID> Pv( sz );
	Pv.resize( sz );
	Pv.resize( contract::detail::intersect(
		       I, (VID)std::distance( I, E ), // prune visited elm of P
		       &edges[bindex[u]], deg,
		       Pv.begin() ) );
	// contract::vertex_set<VID> Pv
	// = P.intersect( &edges[index[u]], index[u+1] - index[u] );
	R.push( u );
	mc_iterate( G, R, Pv, depth+1, mc );
	R.pop();
    }
}


class GraphBuilderInduced {
public:
    GraphBuilderInduced( const GraphCSx & G,
			 VID v, const VID * const assigned_clique )
	: GraphBuilderInduced( G, v, assigned_clique,
			       [=]( VID v ) { return true; } ) { }

    template<typename Fn>
    GraphBuilderInduced( const GraphCSx & G,
			 VID v, const VID * const assigned_clique,
			 Fn && is_enabled )
	: g2s( G.numVertices(), numa_allocation_interleaved() ) {
	VID n = G.numVertices();
	EID m = G.numEdges();
	const EID * const gindex = G.getIndex();
	const VID * const gedges = G.getEdges();

	// Set of eligible neighbours
	contract::vertex_set<VID> ngh;
	std::fill( &g2s.get()[0], &g2s.get()[n], ~(VID)0 );
	g2s[v] = 0;
	VID ns = 1;
	for( EID e=gindex[v], ee=gindex[v+1]; e != ee; ++e ) {
	    VID u = gedges[e];
	    if( ~assigned_clique[e] == 0 && is_enabled( u ) && v != u ) {
		ngh.push( u );
		g2s[u] = ns;
		++ns;
	    }
	}

	EID * tmp = new EID[ns+1];
	std::fill( &tmp[0], &tmp[ns], 0 );

	new ( &s2g ) mm::buffer<VID>( ns, numa_allocation_interleaved() );
	
	// Count vertices and edges
	EID ms = ngh.size();
	tmp[0] = ngh.size();
	s2g[0] = v;
	for( auto && u : ngh ) {
	    VID su = g2s[u];
	    assert( 0 < su && su < ns );
	    s2g[su] = u;
	    ms++;
	    tmp[su]++;
	    contract::detail::intersect_tmpl(
		(const VID*)&*ngh.begin(), (const VID*)&*ngh.end(),
		&gedges[gindex[u]], &gedges[gindex[u+1]],
		[&]( VID w ) {
		    const VID * const pos = std::lower_bound(
			&gedges[gindex[u]], &gedges[gindex[u+1]], w );
		    if( pos != &gedges[gindex[u+1]] && *pos == w && u != w ) {
			EID ew = pos - gedges;
			if( ~assigned_clique[ew] == 0 ) {
			    ++ms;
			    ++tmp[su];
			}
		    }
		    return true;
		} );
	}

	// ms *= 2; // ???

	// std::cerr << "ns=" << ns << "\n";
	// std::cerr << "ms=" << ms << "\n";

	// Construct selected graph
	new ( &S ) graptor::graph::GraphDoubleIndexCSx( ns, ms ); // , -1, true, false );
	EID * sindex = S.getBeginIndex();
	EID * eindex = S.getEndIndex();
	VID * edges = S.getEdges();

	EID mms = sequence::plusScan( tmp, tmp, ns );
	assert( ms == mms );
	std::copy( &tmp[0], &tmp[ns], sindex );
	std::copy( &tmp[0], &tmp[ns], eindex );
	sindex[ns] = ms;
	eindex[ns] = ms;

	assert( ms % 2 == 0 );
	
	for( auto && u : ngh ) {
	    VID su = g2s[u];
	    edges[tmp[0]++] = su;
	    edges[tmp[su]++] = 0;
	    contract::detail::intersect_tmpl(
		(const VID *)&*ngh.begin(), (const VID *)&*ngh.end(),
		&gedges[gindex[u]], &gedges[gindex[u+1]],
		[&]( VID w ) {
		    const VID * const pos = std::lower_bound(
			&gedges[gindex[u]], &gedges[gindex[u+1]], w );
		    if( pos != &gedges[gindex[u+1]] && *pos == w && u != w ) {
			EID ew = pos - gedges;
			if( ~assigned_clique[ew] == 0 ) {
			    edges[tmp[su]++] = g2s[w];
			}
		    }
		    return true;
		} );
	    assert( tmp[su] == sindex[su+1] );
	    eindex[su] = tmp[su];
	}

	delete[] tmp;

	// S.build_degree();
    }
    GraphBuilderInduced( const GraphCSx & G,
			 VID v, const VID * const assigned_clique,
			 VID num_neighbours, const VID * neighbours )
	: g2s( G.numVertices(), numa_allocation_interleaved() ) {
	VID n = G.numVertices();
	EID m = G.numEdges();
	const EID * const gindex = G.getIndex();
	const VID * const gedges = G.getEdges();

	// Set of eligible neighbours
	std::fill( &g2s.get()[0], &g2s.get()[n], ~(VID)0 );
	VID ns = num_neighbours;
	for( VID i=0; i < num_neighbours; ++i ) {
	    VID u = neighbours[i];
	    g2s[u] = i;
	}

	EID * tmp = new EID[ns+1];
	std::fill( &tmp[0], &tmp[ns], 0 );

	new ( &s2g ) mm::buffer<VID>( ns, numa_allocation_interleaved() );
	
	// Count vertices and edges
	EID ms = 0;
	for( VID su=0; su < ns; ++su ) {
	    VID u = neighbours[su];
	    s2g[su] = u;
	    contract::detail::intersect_tmpl(
		&neighbours[0], &neighbours[num_neighbours],
		&gedges[gindex[u]], &gedges[gindex[u+1]],
		[&]( VID w ) {
		    const VID * const pos = std::lower_bound(
			&gedges[gindex[u]], &gedges[gindex[u+1]], w );
		    if( pos != &gedges[gindex[u+1]] && *pos == w && u != w ) {
			EID ew = pos - gedges;
			if( ~assigned_clique[ew] == 0 ) {
			    ++ms;
			    ++tmp[su];
			}
		    }
		    return true;
		} );
	}

	// ms *= 2; // ???

	// std::cerr << "ns=" << ns << "\n";
	// std::cerr << "ms=" << ms << "\n";

	// Construct selected graph
	new ( &S ) graptor::graph::GraphDoubleIndexCSx( ns, ms ); // , -1, true, false );
	EID * sindex = S.getBeginIndex();
	EID * eindex = S.getEndIndex();
	VID * edges = S.getEdges();

	EID mms = sequence::plusScan( tmp, tmp, ns );
	assert( ms == mms );
	std::copy( &tmp[0], &tmp[ns], sindex );
	sindex[ns] = ms;
	eindex[ns] = ms;

	assert( ms % 2 == 0 );
	
	for( VID su=0; su < ns; ++su ) {
	    VID u = neighbours[su];
	    contract::detail::intersect_tmpl(
		&neighbours[0], &neighbours[num_neighbours],
		&gedges[gindex[u]], &gedges[gindex[u+1]],
		[&]( VID w ) {
		    const VID * const pos = std::lower_bound(
			&gedges[gindex[u]], &gedges[gindex[u+1]], w );
		    if( pos != &gedges[gindex[u+1]] && *pos == w && u != w ) {
			EID ew = pos - gedges;
			if( ~assigned_clique[ew] == 0 ) {
			    edges[tmp[su]++] = g2s[w];
			}
		    }
		    return true;
		} );
	    assert( tmp[su] == sindex[su+1] );
	    eindex[su] = tmp[su];
	}

	delete[] tmp;

	// S.build_degree();
    }


    ~GraphBuilderInduced() {
	// S.del();
	g2s.del();
	s2g.del();
    }

    const VID * get_g2s() const { return g2s.get(); }
    const VID * get_s2g() const { return s2g.get(); }
    auto & get_graph() { return S; }

private:
    graptor::graph::GraphDoubleIndexCSx<VID,EID> S;
    mm::buffer<VID> g2s;
    mm::buffer<VID> s2g;
};

class GraphBuilderComplement {
public:
    GraphBuilderComplement( const graptor::graph::GraphCSx<VID,EID> & G ) {
	VID n = G.numVertices();
	EID m = G.numEdges();
	const EID * const gindex = G.getIndex();
	const VID * const gedges = G.getEdges();

	// Count number of self-edges
	EID mself = 0;
	for( VID v=0; v < n; ++v ) {
	    const VID * p = std::lower_bound(
		&gedges[gindex[v]],
		&gedges[gindex[v+1]],
		v );
	    if( p != &gedges[gindex[v+1]] && *p == v )
		++mself;
	}

	// Construct selected graph
	// Edges: complement, not including diagonal
	EID ms = EID(n) * EID(n) - EID(n) - ( m - mself );
	new ( &S ) graptor::graph::GraphCSx( n, ms ); // , -1, true, false );
	EID * index = S.getIndex();
	VID * edges = S.getEdges();

	// Set up index array
	parallel_loop( (VID)0, n, [&]( VID v ) {
	    // Degree of vertex
	    EID deg = gindex[v+1] - gindex[v];

	    // Account for self-edge, if any
	    const VID * p = std::lower_bound(
		&gedges[gindex[v]],
		&gedges[gindex[v+1]],
		v );
	    if( p != &gedges[gindex[v+1]] && *p == v )
		deg--; // deg is one too large

	    index[v] = EID(n) - 1 - deg;
	} );
	EID mms = sequence::plusScan( index, index, n );
	assert( mms == ms && "edge count error" );
	index[n] = ms;

	parallel_loop( (VID)0, n, [&]( VID v ) {
	    EID ge = gindex[v], gee = gindex[v+1];
	    EID e = index[v];
	    for( VID u=0; u < n; ++u ) {
		if( ge != gee && u == gedges[ge] )
		    ++ge;
		else if( u != v ) // no self-edges
		    edges[e++] = u;
	    }
	    assert( ge == gee );
	    assert( e == index[v+1] );
	} );
	
	// S.build_degree();
    }
    ~GraphBuilderComplement() {
	// S.del();
    }

    const graptor::graph::GraphCSx<VID,EID> & get_graph() const { return S; }

private:
    graptor::graph::GraphCSx<VID,EID> S;
};

class GraphBuilderComplementAndPrune {
public:
    GraphBuilderComplementAndPrune(
	const graptor::graph::GraphDoubleIndexCSx<VID,EID> & G )
	: m_g2s( G.numVertices(), numa_allocation_interleaved() ),
	  m_s2g( G.numVertices(), numa_allocation_interleaved() ) { // pessimistic
	VID n = G.numVertices();
	EID m = G.numEdges();
	const EID * const gsindex = G.getBeginIndex();
	const EID * const geindex = G.getEndIndex();
	const VID * const gedges = G.getEdges();

	// Count number of self-edges
	EID mself = 0;
	VID new_n = 0;
	for( VID v=0; v < n; ++v ) {
	    m_g2s[v] = ~(VID)0;
	    const VID * p = std::lower_bound(
		&gedges[gsindex[v]],
		&gedges[geindex[v]],
		v );
	    if( p != &gedges[geindex[v]] && *p == v ) {
		if( geindex[v] - gsindex[v] > 1 ) {
		    ++mself;
		    m_g2s[v] = new_n;
		    m_s2g[new_n] = v;
		    ++new_n;
		}
	    } else {
		if( geindex[v] - gsindex[v] > 0 ) {
		    m_g2s[v] = new_n;
		    m_s2g[new_n] = v;
		    ++new_n;
		}
	    }
	}

	// Construct selected graph
	// Edges: complement, not including diagonal
	EID ms = EID(new_n) * EID(new_n) - EID(new_n) - ( m - mself );
	new ( &S ) graptor::graph::GraphDoubleIndexCSx( new_n, ms ); // , -1, true, false );
	EID * sindex = S.getBeginIndex();
	EID * eindex = S.getEndIndex();
	VID * edges = S.getEdges();

	// Set up index array
	for( VID v=0; v < n; ++v ) {
	    // Degree of vertex
	    EID deg = geindex[v] - gsindex[v];

	    if( ~m_g2s[v] == 0 )
		continue;

	    // if( deg == 0 || ( deg == 1 && gedges[gsindex[v]] == v ) )
	    // continue;

	    // Account for self-edge, if any
	    const VID * p = std::lower_bound(
		&gedges[gsindex[v]],
		&gedges[geindex[v]],
		v );
	    // if( p != &gedges[geindex[v]] && *p == v )
	    // deg--; // deg is one too large
	    if( p != &gedges[geindex[v]] && *p == v ) {
		if( geindex[v] - gsindex[v] > 1 ) // should be futile
		    deg--;
	    }

	    sindex[m_g2s[v]] = EID(new_n) - 1 - deg;
	}
	// seq -> use std::exclusive_scan
	EID mms = sequence::plusScan( sindex, sindex, new_n );
	assert( mms == ms && "edge count error" );
	sindex[new_n] = ms;
	std::copy( sindex+1, sindex+new_n, eindex );
	eindex[new_n-1] = ms;
	eindex[new_n] = ms;

	for( VID vs=0; vs < new_n; ++vs ) {
	    VID v = m_s2g[vs];
	    EID ge = gsindex[v], gee = geindex[v];
	    EID e = sindex[vs];

	    for( VID us=0; us < new_n; ++us ) {
		VID u = m_s2g[us];
		while( ge != gee && gedges[ge] < u )
		    ++ge;
		assert( ge == gee || gedges[ge] >= u );
		if( ( ge == gee || u != gedges[ge] )
		    && us != vs ) // no self-edges
		    edges[e++] = us;
	    }
	    assert( ge == gee || ge == gee-1 );
	    assert( e == sindex[vs+1] );
	    assert( e == eindex[vs] );
	}
	
	// S.build_degree();
    }
    ~GraphBuilderComplementAndPrune() {
	// S.del();
	m_g2s.del();
	m_s2g.del();
    }

    const VID * get_g2s() const { return m_g2s.get(); }
    const VID * get_s2g() const { return m_s2g.get(); }
    auto & get_graph() { return S; }

private:
    graptor::graph::GraphDoubleIndexCSx<VID,EID> S;
    mm::buffer<VID> m_g2s;
    mm::buffer<VID> m_s2g;
};

	 
/*
contract::vertex_set<VID>
induced_max_clique_kb( const graptor::graph::GraphCSx<VID,EID> & G,
		       VID v,
		       const VID * assigned_clique ) {
    VID n = G.numVertices();
    EID m = G.numEdges();
    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();

    contract::vertex_set<VID> N, mc;
    
    for( EID e=index[v], ee=index[v+1]; e != ee; ++e ) {
	if( ~assigned_clique[e] == 0 )
	    N.push( edges[e] );
    }

    for( EID e=index[v], ee=index[v+1]; e != ee; ++e ) {
	if( ~assigned_clique[e] != 0 )
	    continue;
	
	VID u = edges[e];
	contract::vertex_set<VID> R, P;

	R.push( v );
	R.push( u );

	// Consider as candidates only those neighbours of u that are larger than u
	// to avoid revisiting the vertices unnecessarily.
	const VID * const start = std::upper_bound(
	    &edges[index[u]], &edges[index[u+1]], u );
	P = N.intersect( start, &edges[index[u+1]] - start );

	mc_iterate( G, R, P, 1, mc );
    }

    return mc;
}
*/

void
mark( VID & best_size, VID * best_cover, VID v ) {
    best_cover[best_size++] = v;
}

// For path or cycle
void
trace_path( VID n, 
	    const EID * const index,
	    const VID * const edges,
	    bool * visited,
	    VID & best_size,
	    VID * best_cover,
	    VID cur,
	    VID nxt,
	    bool incl ) {
    if( visited[nxt] )
	return;

    visited[nxt] = true;

    if( incl )
	mark( best_size, best_cover, nxt );

    // Done if nxt is degree-1 vertex
    if( index[nxt+1] - index[nxt] == 2 ) {
	VID ngh1 = edges[index[nxt]];
	VID ngh2 = edges[index[nxt]+1];

	VID ngh = ngh1 == cur ? ngh2 : ngh1;

	trace_path( n, index, edges, visited, best_size, best_cover, nxt, ngh,
		    !incl );
    }
}

// For path or cycle
void
trace_path( const graptor::graph::GraphDoubleIndexCSx<VID,EID> & G,
	    bool * visited,
	    VID & best_size,
	    VID * best_cover,
	    VID cur,
	    VID nxt,
	    bool incl ) {
    if( visited[nxt] )
	return;

    visited[nxt] = true;

    if( incl )
	mark( best_size, best_cover, nxt );

    const EID * const index = G.getBeginIndex();
    const VID * const edges = G.getEdges();

    // Done if nxt is degree-1 vertex
    if( G.getDegree( nxt ) == 2 ) {
	VID ngh1 = edges[index[nxt]];
	VID ngh2 = edges[index[nxt]+1];

	VID ngh = ngh1 == cur ? ngh2 : ngh1;

	trace_path( G, visited, best_size, best_cover, nxt, ngh, !incl );
    }
}


bool
vertex_cover_poly( VID n, 
		   VID k,
		   const EID * const index,
		   const VID * const edges,
		   VID & best_size,
		   VID * best_cover ) {
    bool * visited = new bool[n];
    std::fill( visited, &visited[n], false );

    VID old_best_size = best_size;

    // Find paths
    for( VID v=0; v < n; ++v ) {
	assert( index[v+1] - index[v] <= 2 );
	if( index[v+1] - index[v] == 1 && !visited[v] ) {
	    visited[v] = true;
	    trace_path( n, index, edges, visited, best_size, best_cover,
			v, edges[index[v]], true );
	}
    }
    
    // Find cycles (uses same auxiliary as paths)
    for( VID v=0; v < n; ++v ) {
	assert( index[v+1] - index[v] <= 2 );
	if( index[v+1] - index[v] == 2 && !visited[v] ) {
	    visited[v] = true;
	    mark( best_size, best_cover, v );
	    trace_path( n, index, edges, visited, best_size, best_cover,
			v, edges[index[v]], false );
	}
    }

    delete[] visited;

    return best_size - old_best_size <= k;
}

bool
vertex_cover_poly( const graptor::graph::GraphDoubleIndexCSx<VID,EID> & G,
		   VID k,
		   VID & best_size,
		   VID * best_cover ) {
    VID n = G.numVertices();

    bool * visited = new bool[n];
    std::fill( visited, &visited[n], false );

    VID old_best_size = best_size;

    // Find paths
    for( VID v=0; v < n; ++v ) {
	VID deg = G.getDegree( v );
	assert( deg <= 2 );
	if( deg == 1 && !visited[v] ) {
	    visited[v] = true;
	    trace_path( G, visited, best_size, best_cover,
			v, *G.nbegin( v ), true );
	}
    }
    
    // Find cycles (uses same auxiliary as paths)
    for( VID v=0; v < n; ++v ) {
	VID deg = G.getDegree( v );
	assert( deg <= 2 );
	if( deg == 2 && !visited[v] ) {
	    visited[v] = true;
	    mark( best_size, best_cover, v );
	    trace_path( G, visited, best_size, best_cover,
			v, *G.nbegin( v ), false );
	}
    }

    delete[] visited;

    return best_size - old_best_size <= k;
}


void
check_cover( VID n, 
	     const EID * const index,
	     const VID * const edges,
	     VID size,
	     VID * cover ) {
    std::sort( cover, cover+size );
    EID e = index[0];
    for( VID v=0; v < n; ++v ) {
	const VID * pos = std::lower_bound( cover, cover+size, v );
	if( pos != cover+size && *pos == v ) {
	    e = index[v+1];
	} else {
	    for( EID ee = index[v+1]; e != ee; ++e ) {
		VID u = edges[e];
		const VID * pos = std::lower_bound( cover, cover+size, u );
		if( pos != cover+size && *pos == u ) {
		    ; // ok
		} else {
		    assert( 0 && "not a cover" );
		}
	    }
	}
    }
}

void
check_cover( const graptor::graph::GraphDoubleIndexCSx<VID,EID> & G,
	     VID size,
	     VID * cover ) {
    std::sort( cover, cover+size );
    VID n = G.numVertices();
    const EID * const bindex = G.getBeginIndex();
    const EID * const eindex = G.getEndIndex();
    const VID * const edges = G.getEdges();
    for( VID v=0; v < n; ++v ) {
	const VID * pos = std::lower_bound( cover, cover+size, v );
	if( pos != cover+size && *pos == v ) {
	    continue;
	} else {
	    for( EID e=bindex[v], ee=eindex[v]; e != ee; ++e ) {
		VID u = edges[e];
		const VID * pos = std::lower_bound( cover, cover+size, u );
		if( pos != cover+size && *pos == u ) {
		    ; // ok
		} else {
		    assert( 0 && "not a cover" );
		}
	    }
	}
    }
}


bool
vertex_cover_vc3( VID n, 
		  VID k,
		  VID c,
		  const EID * const index,
		  const VID * const edges,
		  VID & best_size,
		  VID * best_cover );

bool
vertex_cover_vc3_buss( VID n, 
		       VID k,
		       VID c,
		       const EID * const index,
		       const VID * const edges,
		       VID & best_size,
		       VID * best_cover ) {
    // Set U of vertices of degree higher than k
    VID u_size = 0;
    for( VID v=0; v < n; ++v )
	if( index[v+1] - index[v] > k )
	    ++u_size;

    // If |U| > k, then there exists no cover of size k
    if( u_size > k )
	return false;

    assert( u_size > 0 );
    
    // Construct G'
    EID * gp_index = new EID[n - u_size + 1];
    VID * gp_xlat = new VID[n - u_size];
    VID * gp_rlat = new VID[n];
    EID m = 0;
    VID x = 0;
    gp_index[0] = 0;
    for( VID v=0; v < n; ++v ) {
	if( index[v+1] - index[v] > k ) // v excluded
	    continue;

	// Map new vertex IDs to old ones
	gp_xlat[x] = v;
	gp_rlat[v] = x;

	for( EID e=index[v], ee=index[v+1]; e != ee; ++e ) {
	    VID u = edges[e];
	    if( index[u+1] - index[u] <= k )
		m++;
	}
	gp_index[x+1] = m;

	// Next new vertex ID
	++x;
    }

    assert( x == n - u_size );

    VID * gp_edges = new VID[m];
    EID pos = 0;
    for( VID v=0; v < n; ++v ) {
	if( index[v+1] - index[v] > k ) // v excluded
	    continue;

	for( EID e=index[v], ee=index[v+1]; e != ee; ++e ) {
	    VID u = edges[e];
	    if( index[u+1] - index[u] <= k )
		gp_edges[pos++] = gp_rlat[u];
	}
    }

    assert( pos == m );
    

    // If G' has more than k(k-|U|) edges, reject
    if( m > k * ( k - u_size ) ) {
	delete[] gp_index;
	delete[] gp_edges;
	delete[] gp_xlat;
	delete[] gp_rlat;
	return false;
    }

    // Find a cover for the remaining vertices
    VID gp_best_size = 0;
    bool rec = vertex_cover_vc3(
	n - u_size, k - u_size, c, gp_index, gp_edges,
	gp_best_size, &best_cover[best_size] );

    if( rec ) {
	// Debug
	// check_cover( n - u_size, gp_index, gp_edges, gp_best_size, &best_cover[best_size] );

	for( VID i=0; i < gp_best_size; ++i )
	    best_cover[best_size+i] = gp_xlat[best_cover[best_size+i]];
	best_size += gp_best_size;

	// All vertices with degree > k must be included in the cover
	for( VID v=0; v < n; ++v )
	    if( index[v+1] - index[v] > k )
		best_cover[best_size++] = v;

	// check_cover( n, index, edges, best_size, best_cover );
    }

    delete[] gp_edges;
    delete[] gp_index;
    delete[] gp_xlat;
    delete[] gp_rlat;

    return rec;
}


bool
vertex_cover_vc3( VID n, 
		  VID k,
		  VID c,
		  const EID * const index,
		  const VID * const edges,
		  VID & best_size,
		  VID * best_cover ) {
    EID m = index[n];

    VID max_deg = 0;
    VID max_v = 0;
    for( VID v=0; v < n; ++v ) {
	VID deg = index[v+1] - index[v];
	if( deg > max_deg ) {
	    max_deg = deg;
	    max_v = v;
	}
    }
    if( max_deg <= 2 )
	return vertex_cover_poly( n, k, index, edges, best_size, best_cover );

    if( m/2 > c * k * k && max_deg > k ) {
	// replace by Buss kernel
	return vertex_cover_vc3_buss(
	    n, k, c, index, edges, best_size, best_cover );
    }

    // Any vertex with degree > 3
    assert( index[max_v+1] - index[max_v] >= 3 );

    // Create two subproblems
    EID * i_index = new EID[n-1+1];
    EID * x_index = new EID[n-max_deg-1+1];
    VID * i_xlat = new VID[n-1];
    VID * x_xlat = new VID[n-max_deg-1];
    VID * i_rlat = new VID[n];
    VID * x_rlat = new VID[n];

    VID i_n = 0, x_n = 0;
    EID i_m = 0, x_m = 0;
    i_index[0] = 0;
    x_index[0] = 0;
    for( VID v=0; v < n; ++v ) {
	if( v == max_v ) // removed
	    continue;

	const VID * vpos = std::lower_bound(
	    &edges[index[max_v]], &edges[index[max_v+1]], v );
	// v is a neighbour of max_v, hence removed
	bool x_rm = vpos != &edges[index[max_v+1]] && *vpos == v;
	
	for( EID e=index[v], ee=index[v+1]; e != ee; ++e ) {
	    VID u = edges[e];
	    if( u == max_v ) // u is removed
		continue;

	    i_m++;

	    if( !x_rm ) {
		const VID * pos = std::lower_bound(
		    &edges[index[max_v]], &edges[index[max_v+1]], u );
		// u is not a neighbour of max_v, hence not removed
		if( pos == &edges[index[max_v+1]] || *pos != u )
		    x_m++;
	    }
	}
	i_xlat[i_n] = v;
	i_rlat[v] = i_n;
	i_index[++i_n] = i_m;
	if( !x_rm ) {
	    x_xlat[x_n] = v;
	    x_rlat[v] = x_n;
	    x_index[++x_n] = x_m;
	}
    }

    assert( i_n == n - 1 );
    assert( x_n == n - 1 - max_deg );

    VID * i_edges = new VID[i_m];
    VID * x_edges = new VID[x_m];
    EID i_pos = 0, x_pos = 0;
    for( VID v=0; v < n; ++v ) {
	if( v == max_v ) // removed
	    continue;

	const VID * vpos = std::lower_bound(
	    &edges[index[max_v]], &edges[index[max_v+1]], v );
	// v is a neighbour of max_v, hence removed
	bool x_rm = vpos != &edges[index[max_v+1]] && *vpos == v;
	
	for( EID e=index[v], ee=index[v+1]; e != ee; ++e ) {
	    VID u = edges[e];
	    if( u == max_v ) // u is removed
		continue;

	    i_edges[i_pos++] = i_rlat[u];

	    if( !x_rm ) {
		const VID * pos = std::lower_bound(
		    &edges[index[max_v]], &edges[index[max_v+1]], u );
		// u is not a neighbour of max_v, hence not removed
		if( pos == &edges[index[max_v+1]] || *pos != u )
		    x_edges[x_pos++] = x_rlat[u];
	    }
	}
    }

    assert( i_pos == i_m );
    assert( x_pos == x_m );

    
    // Branch on max_v
    VID i_best_size = 0;
    VID * i_best_cover = new VID[n-1];
    VID x_best_size = 0;
    VID * x_best_cover = new VID[n-1-max_deg];

    VID x_k = std::min( n-1-max_deg, k-max_deg );
    bool x_ok = vertex_cover_vc3( n-1-max_deg, x_k, c,
				  x_index, x_edges,
				  x_best_size, x_best_cover );
    VID i_k = x_ok ? std::min( max_deg+x_best_size, k-1 ) : k-1;
    bool i_ok = vertex_cover_vc3( n-1, i_k, c, i_index, i_edges,
				  i_best_size, i_best_cover );

    if( i_ok && ( !x_ok || i_best_size+1 < x_best_size+max_deg ) ) {
	best_cover[best_size++] = max_v;
	for( VID i=0; i < i_best_size; ++i )
	    best_cover[best_size++] = i_xlat[i_best_cover[i]];
    } else if( x_ok ) {
	for( EID e=index[max_v], ee=index[max_v+1]; e != ee; ++e )
	    best_cover[best_size++] = edges[e];
	for( VID i=0; i < x_best_size; ++i )
	    best_cover[best_size++] = x_xlat[x_best_cover[i]];
    }

    delete[] i_index;
    delete[] i_edges;
    delete[] i_xlat;
    delete[] i_rlat;
    delete[] i_best_cover;
    delete[] x_index;
    delete[] x_edges;
    delete[] x_xlat;
    delete[] x_rlat;
    delete[] x_best_cover;

    return i_ok || x_ok;
}

bool
vertex_cover_vc3( graptor::graph::GraphDoubleIndexCSx<VID,EID> & G,
		  VID k,
		  VID c,
		  VID & best_size,
		  VID * best_cover );

bool
vertex_cover_vc3_buss( graptor::graph::GraphDoubleIndexCSx<VID,EID> & G,
		       VID k,
		       VID c,
		       VID & best_size,
		       VID * best_cover ) {
    // Set U of vertices of degree higher than k
    VID u_size = std::count_if( G.dbegin(), G.dend(),
				[&]( VID deg ) { return deg > k; } );

    // If |U| > k, then there exists no cover of size k
    if( u_size > k )
	return false;

    assert( u_size > 0 );
    
    // Construct G'
    VID n = G.numVertices();
    auto chkpt = G.checkpoint();
    G.disable_incident_edges( [&]( VID v ) {
	return chkpt.get_degree( v ) > k;
    } );
    EID m = G.numEdges();
    
    // If G' has more than k(k-|U|) edges, reject
    if( m > k * ( k - u_size ) ) {
	G.restore_checkpoint( chkpt );
	return false;
    }

    // Find a cover for the remaining vertices
    VID gp_best_size = 0;
    bool rec = vertex_cover_vc3(
	G, k - u_size, c,
	gp_best_size, &best_cover[best_size] );

    if( rec ) {
	// Debug
	// check_cover( G, gp_best_size, &best_cover[best_size] );

	// for( VID i=0; i < gp_best_size; ++i )
	// best_cover[best_size+i] = gp_xlat[best_cover[best_size+i]];
	best_size += gp_best_size;

	// All vertices with degree > k must be included in the cover
	for( VID v=0; v < n; ++v )
	    if( G.getDegree( v ) > k )
		best_cover[best_size++] = v;

	// check_cover( G, best_size, best_cover );
    }

    G.restore_checkpoint( chkpt );

    return rec;
}


bool
vertex_cover_vc3( graptor::graph::GraphDoubleIndexCSx<VID,EID> & G,
		  VID k,
		  VID c,
		  VID & best_size,
		  VID * best_cover ) {
    VID n = G.numVertices();
    // VID n = std::count_if(
    // G.dbegin(), G.dend(), [=]( EID deg ) { return deg > 0; } );
    EID m = G.numEdges();

    VID max_v, max_deg;
    std::tie( max_v, max_deg ) = G.max_degree();
    if( max_deg <= 2 ) {
	bool ret = vertex_cover_poly( G, k, best_size, best_cover );
	// if( ret )
	// check_cover( G, best_size, best_cover );
	return ret;
    }

    if( k == 0 )
	return m == 0;

/*
    if( m/2 > c * k * k && max_deg > k ) {
	// replace by Buss kernel
	return vertex_cover_vc3_buss( G, k, c, best_size, best_cover );
    }
*/

    // Any vertex with degree > 3
    assert( G.getDegree( max_v ) >= 3 );

    // Create two subproblems ; branch on max_v
    VID i_best_size = 0;
    VID * i_best_cover = new VID[n-1];
    VID x_best_size = 0;
    VID * x_best_cover = new VID[n-1-max_deg];

    // In case v is included, erase only v
    auto chkpt = G.checkpoint();
    G.disable_incident_edges( [=]( VID v ) { return v == max_v; } );

    // In case v is excluded, erase v (previous step) and all its neighbours
    // Make sure our neighbours are sorted. Iterators remain valid after
    // erasing incident edges.
    G.sort_neighbours( max_v );
    auto NI = G.nbegin( max_v );
    auto NE = G.nend( max_v );
    auto chkpti = G.checkpoint();
    G.disable_incident_edges( [=]( VID v ) {
	auto pos = std::lower_bound( NI, NE, v );
	return pos != NE && *pos == v;
    } );

    VID x_k = std::min( n-1-max_deg, k-max_deg );
    bool x_ok = false;
    if( k >= max_deg )
	x_ok = vertex_cover_vc3( G, x_k, c, x_best_size, x_best_cover );

    G.restore_checkpoint( chkpti );
    VID i_k = x_ok ? std::min( max_deg+x_best_size, k-1 ) : k-1;
    bool i_ok = vertex_cover_vc3( G, i_k, c, i_best_size, i_best_cover );

    if( i_ok && ( !x_ok || i_best_size+1 < x_best_size+max_deg ) ) {
	best_cover[best_size++] = max_v;
	for( VID i=0; i < i_best_size; ++i )
	    best_cover[best_size++] = i_best_cover[i]; // i_xlat[i_best_cover[i]];
    } else if( x_ok ) {
	for( auto I=NI; I != NE; ++I )
	    best_cover[best_size++] = *I;
	for( VID i=0; i < x_best_size; ++i )
	    best_cover[best_size++] = x_best_cover[i]; // x_xlat[x_best_cover[i]];
    }

    G.restore_checkpoint( chkpt );

    // if( i_ok || x_ok )
    // check_cover( G, best_size, best_cover );

    return i_ok || x_ok;
}


bool
vertex_cover_buss( VID n, 
		   VID k,
		   const EID * const index,
		   const VID * const edges,
		   VID & best_size,
		   VID * best_cover ) {
    // Set U of vertices of degree higher than k
    VID u_size = 0;
    for( VID v=0; v < n; ++v )
	if( index[v+1] - index[v] > k )
	    ++u_size;

    // If |U| > k, then there exists no cover of size k
    if( u_size > k )
	return false;

    assert( u_size > 0 );
    
    // Construct G'
    EID * gp_index = new EID[n - u_size + 1];
    VID * gp_xlat = new VID[n - u_size];
    EID m = 0;
    VID x = 0;
    gp_index[0] = 0;
    for( VID v=0; v < n; ++v ) {
	if( index[v+1] - index[v] > k ) // v excluded
	    continue;

	// Map new vertex IDs to old ones
	gp_xlat[x] = v;

	VID nn = 0;
	for( EID e=index[v], ee=index[v+1]; e != ee; ++e ) {
	    VID u = edges[e];
	    if( index[u+1] - index[u] <= k )
		++nn;
	}
	m += nn;
	gp_index[x+1] = nn + gp_index[x];

	// Next new vertex ID
	++x;
    }

    assert( x == n - u_size );
    gp_index[n - u_size] = m;

    // If G' has more than k(k-|U|) edges, reject
    if( m > k * ( k - u_size ) ) {
	delete[] gp_index;
	delete[] gp_xlat;
	return false;
    }

    VID * gp_edges = new VID[m];

    x = 0;
    EID mm = 0;
    for( VID v=0; v < n; ++v ) {
	if( index[v+1] - index[v] > k ) // v excluded
	    continue;

	VID nn = 0;
	for( EID e=index[v], ee=index[v+1]; e != ee; ++e ) {
	    VID u = edges[e];
	    if( index[u+1] - index[u] <= k )
		gp_edges[mm++] = u;
	}
	assert( mm == gp_index[x+1] );

	// Next new vertex ID
	++x;
    }
    
    // Find a cover for the remaining vertices
    VID gp_best_size = 0;
    bool rec = vertex_cover_buss( n - u_size, k - u_size, gp_index, gp_edges,
				  gp_best_size, &best_cover[best_size] );

    if( rec ) {
	// Debug
	// check_cover( n, index, edges, gp_best_size, &best_cover[best_size] );

	for( VID i=0; i < gp_best_size; ++i )
	    best_cover[best_size+i] = gp_xlat[best_cover[best_size+i]];
	best_size += gp_best_size;
    }

    delete[] gp_edges;
    delete[] gp_index;
    delete[] gp_xlat;

    return rec;
}

void
vertex_cover_buss( VID n, 
		   const EID * const index,
		   const VID * const edges,
		   VID maxdeg,
		   VID & best_size,
		   VID * best_cover ) {
    VID k_min = 0, k_max = maxdeg;

    VID * cover = new VID[n];
    VID size = 0;
    bool rec = false;

    do {
	VID k = ( k_min + k_max ) / 2;
	std::cerr << "  k=" << k << " min=" << k_min << " max=" << k_max << "\n";

	rec = vertex_cover_buss( n, k, index, edges, size, cover );

	if( rec )
	    k_max = k;
	else
	    k_min = k;
	
    } while( !rec && k_min < k_max-1 );

    std::copy( &cover[0], &cover[size], best_cover );
    best_size = size;

    // Search for a smaller cover
    k_max -= 1;

    do {
	VID k = ( k_min + k_max ) / 2;

	rec = vertex_cover_buss( n, k, index, edges, size, cover );

	if( rec ) {
	    k_max = k - 1;
	    std::copy( &cover[0], &cover[size], best_cover );
	    best_size = size;
	} else
	    k_min = k;
	
    } while( rec && k_min < k_max-1 );
}

VID
vertex_cover_iterate( VID n, 
		      const EID * const index,
		      const VID * const edges,
		      VID v,
		      short * state,
		      VID vcsize,
		      VID & best_size,
		      short * best_state ) {
    // We have tried all vertices
    if( v == n ) {
	if( vcsize < best_size ) {
	    std::copy( &state[0], &state[n], best_state );
	    best_size = vcsize;
	}
	return vcsize;
    }

    // If no incident edges, then don't want vertex
    VID deg = index[v+1] - index[v];
    if( deg == 0 ) {
	state[v] = -1;
	VID sz = vertex_cover_iterate( n, index, edges, v+1, state, vcsize,
				       best_size, best_state );
	state[v] = 0;
	return sz;
    }
// todo: if degree > best_size -> skip vertex

    // First check if all neighbours of v are covered
    VID num_covered = 0;
    VID num_uncovered = 0;
    for( EID e=index[v], ee=index[v+1]; e != ee; ++e ) {
	VID u = edges[e];
	if( state[u] == 1 )
	    num_covered++;
	else if( state[u] == -1 )
	    num_uncovered++;
    }

    // In case we don't have choice: including all neighbours would result in
    // a vertex cover larger than the one of interest. In that case, include the
    // vertex and not the (remaining) neighbours
    if( vcsize + deg - num_covered >= best_size ) {
	state[v] = 1;
	VID sz = vertex_cover_iterate( n, index, edges, v+1, state, vcsize+1,
				       best_size, best_state );
	state[v] = 0;
	return sz;
    }

    // All neighbours included, so this vertex is not needed
    // If any neighbour is undecided, then try both ways.
    if( num_covered == deg ) {
	state[v] = -1;
	VID sz = vertex_cover_iterate( n, index, edges, v+1, state, vcsize,
				       best_size, best_state );
	state[v] = 0;
	return sz;
    } else if( num_uncovered > 0 ) {
	state[v] = 1;
	VID sz = vertex_cover_iterate( n, index, edges, v+1, state, vcsize+1,
				       best_size, best_state );
	state[v] = 0;
	return sz;
    }

    state[v] = -1;
    VID vco = vertex_cover_iterate( n, index, edges, v+1, state, vcsize,
				    best_size, best_state );

    state[v] = 1;
    VID vci = vertex_cover_iterate( n, index, edges, v+1, state, vcsize+1,
				    best_size, best_state );
    state[v] = 0;

    if( vci < vco )
	return vci;
    else
	return vco;
}

contract::vertex_set<VID>
vertex_cover_exact( const graptor::graph::GraphCSx<VID,EID> & G ) {
    VID n = G.numVertices();
    EID m = G.numEdges();
    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();

    // State: 0: undecided, 1: include, -1: include neighbours
    short * state = new short[n];
    short * best_state = new short[n];
    std::fill( &state[0], &state[n], 0 );
    std::fill( &best_state[0], &best_state[n], 1 );
    
    VID best_size = n;
    vertex_cover_iterate( n, index, edges, 0, state, 0, best_size, best_state );

    contract::vertex_set<VID> vc;
    for( VID v=0; v < n; ++v )
	if( best_state[v] == -1 ) // return the MIS/clique, not the cover
	    vc.push( v );

    delete[] state;
    delete[] best_state;
    
    return vc;
}

// Gavril and Yannakakis
contract::vertex_set<VID>
vertex_cover_approximate( const GraphCSx & G ) {
    VID n = G.numVertices();
    EID m = G.numEdges();
    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();

    mm::buffer<bool> selected( n, numa_allocation_interleaved() );
    std::fill( &selected[0], &selected[n], false );

    contract::vertex_set<VID> vc;

    EID e = 0;
    for( VID v=0; v < n; ++v ) {
	EID ee=index[v+1];

	if( selected[v] ) {
	    e = ee;
	    continue;
	}
	    
	for( ; e != ee; ++e ) {
	    VID u = edges[e];
	    if( !selected[u] ) {
		selected[v] = selected[u] = true;
		vc.add( v );
		vc.add( u );
		e = ee;
		break;
	    }
	}
    }

    return vc;
}


/*
contract::vertex_set<VID>
induced_max_clique_vc( const graptor::graph::GraphCSx<VID,EID> & G,
		       VID v,
		       const VID * assigned_clique ) {
    GraphBuilderInduced ibuilder( G, v, assigned_clique );
    const auto & IG = ibuilder.get_graph();

    GraphBuilderComplement cbuilder( IG );
    const auto & CIG = cbuilder.get_graph();

    auto c = vertex_cover_exact( CIG );
    std::transform( c.begin(), c.end(), c.begin(),
		    [&]( VID v ) { return ibuilder.get_s2g()[v]; } );
    return c;
}

// Could choose to solve this using MIS or VC if graph sufficiently dense.
contract::vertex_set<VID>
induced_max_clique( const graptor::graph::GraphCSx<VID,EID> & G,
		    VID v,
		    const VID * assigned_clique ) {
    // return induced_max_clique_kb( G, v, assigned_clique );
    return induced_max_clique_vc( G, v, assigned_clique );
}
*/

contract::vertex_set<VID>
bron_kerbosch( const graptor::graph::GraphCSx<VID,EID> & G ) {
    VID n = G.numVertices();
    EID m = G.numEdges();
    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();

    contract::vertex_set<VID> mc;
    
    for( VID v=0; v < n; ++v ) {
	if( index[v] == index[v+1] )
	    continue;
	
	contract::vertex_set<VID> R, P;

	R.push( v );

	// Consider as candidates only those neighbours of u that are larger
	// than u to avoid revisiting the vertices unnecessarily.
	const VID * const start = std::upper_bound(
	    &edges[index[v]], &edges[index[v+1]], v );
	P.push( start, &edges[index[v+1]] );

	mc_iterate( G, R, P, 1, mc );
    }

/*
    best_size = n - mc.size();
    std::fill( &best_state[0], &best_state[n], 1 );
    for( auto && v : mc )
	best_state[v] = -1;
*/
    return mc;
}

contract::vertex_set<VID>
bron_kerbosch( const graptor::graph::GraphDoubleIndexCSx<VID,EID> & G ) {
    VID n = G.numVertices();
    EID m = G.numEdges();
    const EID * const bindex = G.getBeginIndex();
    const EID * const eindex = G.getEndIndex();
    const VID * const edges = G.getEdges();

    contract::vertex_set<VID> mc;
    
    for( VID v=0; v < n; ++v ) {
	if( bindex[v] == eindex[v] )
	    continue;
	
	contract::vertex_set<VID> R, P;

	R.push( v );

	// Consider as candidates only those neighbours of u that are larger
	// than u to avoid revisiting the vertices unnecessarily.
	const VID * const start = std::upper_bound(
	    &edges[bindex[v]], &edges[eindex[v]], v );
	P.push( start, &edges[eindex[v]] );

	mc_iterate( G, R, P, 1, mc );
    }

/*
    best_size = n - mc.size();
    std::fill( &best_state[0], &best_state[n], 1 );
    for( auto && v : mc )
	best_state[v] = -1;
*/
    return mc;
}


// VID *
template<typename Fn>
void
clique_partition( const GraphCSx & GG,
		  graptor::graph::GraphDoubleIndexCSx<VID,EID> & G,
		  VID v_reference,
		  Fn && map_i2g,
		  VID * assigned_clique,
		  CliqueLister & clist ) {
    // GraphCSRAdaptor GA( G, 1 );

    VID n = G.numVertices();
    EID m = G.numEdges();
    unsigned npart = 1;

    VID to_assign = n;

    // State: 0: undecided, 1: include, -1: include neighbours
    // short * state = new short[n];
    short * best_state = new short[n];

    // VID * clique_number = new VID[n];
    // std::fill( &clique_number[0], &clique_number[n], ~(VID)0 );

    uint64_t tid = gettid();

    while( to_assign > 3 ) {
	// Cut out sub-graph of all vertices not yet placed.
	// (Each vertex (neighbour/edge) can only be placed once because the main
	//  vertex v is part of every clique and each edge may be covered once.)
	timer tm;
	tm.start();
	auto & IG = G;
	float density = float(IG.numEdges())
	    / ( float(IG.numVertices()) * float(IG.numVertices()) );
	VID max_deg = IG.max_degree().second;

	double tb = tm.stop();

	// TODO:
	// * Instances with m=O(n): prune low-degree vertices?
	// * Instance with m << n: ???
	// * Create version of BK that works without rebuilding graph
	// * Iteratively rebuild graph, i.e., reduce down from previous
	//   previous as opposed to reduce from original
	// * Encode neighbour lists in BK (R, P) as bit masks to accelerate
	//   operations. Speed of popcount? Store graph in dense format?

	VID in = IG.numVertices();
	EID im = IG.numEdges();
	VID best_size = in;
	VID cliqno;

	std::cerr << tid << "  subset of unassigned neighbours: "
		  << " n=" << IG.numVertices()
		  << " m=" << IG.numEdges()
		  << " density=" << density
		  << " maxdeg=" << max_deg
		  << " build time=" << tb
		  << " launched\n";

	std::string variant;

	tm.start();

	if( EID(in) * EID( in - 1 ) == im ) {
	    // A clique. Because to_assign > 3, so is in.
	    // Hence, the clique is sufficiently large.
	    
	    std::fill( &best_state[0], &best_state[in], -1 );
	    variant = "clique";
	    best_size = in;

	    // Get a unique ID for the clique
	    VID * members;
	    std::tie( cliqno, members ) = clist.allocate_clique( in+1 );
	    *members++ = v_reference;

	    // std::cerr << "     clique:";
	    for( VID v=0; v < in; ++v ) {
		VID vg = v; // ibuilder.get_s2g()[v];
		// assert( ~clique_number[vg] == 0 );
		// clique_number[vg] = cliqno;
		*members++ = map_i2g( vg );
	    }

	    mark_edges( GG, v_reference, map_i2g,
			graptor::graph::range_iterator( VID(0) ),
			graptor::graph::range_iterator( in ),
			assigned_clique, cliqno );

	    break; // we are done
	} else if( ( im < 8000 && density < 0.1 ) || density < 0.01 ) {
	    std::fill( &best_state[0], &best_state[in], 1 );
	    IG.sort_neighbour_lists();
	    contract::vertex_set<VID> c = bron_kerbosch( IG );
	    variant = "BK";

	    best_size = c.size();
	    
	    if( best_size < 3 )
		break;

	    // Get a unique ID for the clique
	    VID * members;
	    std::tie( cliqno, members ) = clist.allocate_clique( best_size+1 );
	    *members++ = v_reference;

	    for( auto && v : c ) {
		VID vg = v; // ibuilder.get_s2g()[v];
		// assert( ~clique_number[vg] == 0 );
		// clique_number[vg] = cliqno;
		*members++ = map_i2g( vg );
	    }

	    mark_edges( GG, v_reference, map_i2g,
			c.begin(), c.end(),
			assigned_clique, cliqno );

	    std::sort( c.begin(), c.end() ); // necessary?
	    G.disable_incident_edges( c.begin(), c.end() );
	} else {
	    IG.sort_neighbour_lists();
	    GraphBuilderComplementAndPrune cbuilder( IG );
	    auto & CIG = cbuilder.get_graph();
	    VID cn = CIG.numVertices();
	    EID cm = CIG.numEdges();

	    best_size = 0;
	    VID * best_cover = new VID[cn];
	    bool ok = vertex_cover_vc3( CIG, cn, 1, best_size, best_cover );
	    assert( ok );
	    variant = "VC-buss";

	    // check_cover( CIG, best_size, best_cover );

	    assert( best_size > 0 );

	    // Compute complement of set
	    std::fill( &best_state[0], &best_state[cn], -1 );
	    std::for_each( best_cover, &best_cover[best_size],
			   [&]( VID v ) {
			       assert( best_state[v] == -1 ); // check duplic
			       best_state[v] = 1;
			   } );

	    best_size = in - best_size;

	    if( best_size < 3 )
		break;

	    // Get a unique ID for the clique
	    VID * members;
	    std::tie( cliqno, members ) = clist.allocate_clique( best_size+1 );
	    *members++ = v_reference;


	    // determine the MIS/clique, not the cover
	    std::cerr << "          content:";
	    std::vector<VID> c; c.reserve( best_size );
	    for( VID v=0; v < cn; ++v ) {
		if( best_state[v] == -1 ) { // determine the MIS/clique, not the cover
		    VID vg = cbuilder.get_s2g()[v];
		    c.push_back( vg );
		    // assert( ~clique_number[vg] == 0 );
		    // clique_number[vg] = cliqno;
		    *members++ = vg;
		    std::cerr << ' ' << v;
		}
	    }
	    std::cerr << "\n";
	    delete[] best_cover;

	    mark_edges( GG, v_reference, map_i2g,
			c.begin(), c.end(),
			assigned_clique, cliqno );

	    G.disable_incident_edges( [&]( VID v ) {
		VID vs = cbuilder.get_g2s()[v];
		return ~vs != 0 && best_state[vs] == -1;
	    } );
	}

	double t = tm.next();
	std::cerr << tid
		  << ' ' << v_reference
		  << "  subset of unassigned neighbours: "
		  << " n=" << IG.numVertices()
		  << " m=" << IG.numEdges()
		  << " density=" << density
		  << " size=" << best_size
		  << " cliqno=" << cliqno
		  << " time(" << variant << "): " << t
		  << "\n";

	to_assign -= best_size;
    }

    // delete[] state;
    delete[] best_state;

    // return clique_number;
}

template<unsigned Bits>
class DenseMatrix {
    using type = uint64_t;
    static constexpr unsigned bits_per_lane = sizeof(type)*8;
    static constexpr unsigned short VL = Bits / bits_per_lane;
    using tr = vector_type_traits_vl<type,VL>;
    using row_type = typename tr::type;

    static_assert( VL * bits_per_lane == Bits );

public:
    static constexpr size_t MAX_VERTICES = bits_per_lane * VL;

public:
    template<typename Fn>
    DenseMatrix( const GraphCSx & G, VID v, const VID * assigned_clique,
		 Fn && is_enabled ) {
	VID n = G.numVertices();
	EID m = G.numEdges();
	const EID * const gindex = G.getIndex();
	const VID * const gedges = G.getEdges();

	// Set of eligible neighbours
	contract::vertex_set<VID> ngh;
	m_g2s = new VID[n];
	std::fill( &m_g2s[0], &m_g2s[n], ~(VID)0 );
	VID ns = 0;
	for( EID e=gindex[v], ee=gindex[v+1]; e != ee; ++e ) {
	    VID u = gedges[e];
	    if( ~assigned_clique[e] == 0 && is_enabled( u ) && v != u ) {
		ngh.push( u );
		m_g2s[u] = ns;
		++ns;
	    }
	}

	assert( ns <= MAX_VERTICES );

	// m_words = ( ns + bits_per_lane - 1 ) / bits_per_lane;
	m_words = 4; // VL
	assert( ( ns + bits_per_lane - 1 ) / bits_per_lane <= m_words );
	m_words = std::min( (unsigned)VL, m_words );
	m_matrix = m_matrix_alc = new type[m_words * ns + 4];
	intptr_t p = reinterpret_cast<intptr_t>( m_matrix );
	if( p & 31 ) // 31 = 256 bits / 8 bits per byte - 1
	    m_matrix = &m_matrix[(p&31)/sizeof(type)];
	std::fill( &m_matrix[0], &m_matrix[m_words * ns], 0 );

	m_s2g = new VID[ns];

	// Place edges
	VID ni = 0;
	m_m = 0;
	for( auto && u : ngh ) { // can loop over su=0...ns, then u=ngh[su]
	    VID su = m_g2s[u];
	    assert( 0 <= su && su < ns );
	    m_s2g[su] = u;

	    row_type row_u = tr::setzero();

	    contract::detail::intersect_tmpl(
		(const VID*)&*ngh.begin(), (const VID*)&*ngh.end(),
		&gedges[gindex[u]], &gedges[gindex[u+1]],
		[&]( VID w ) {
		    const VID * const pos = std::lower_bound(
			&gedges[gindex[u]], &gedges[gindex[u+1]], w );
		    if( pos != &gedges[gindex[u+1]] && *pos == w && u != w ) {
			EID ew = pos - gedges;
			if( ~assigned_clique[ew] == 0 ) {
			    VID sw = m_g2s[w];
			    row_u = tr::bitwise_or( row_u, create_row( sw ) );
			    // set( su, sw ); // TODO perform on register before storing to memory : row=bitwise_or( row, create_row( sw ) )
			    ++m_m;
			}
		    }
		    return true;
		} );

	    tr::store( &m_matrix[VL * su], row_u );
	}

	m_n = ns;
    }
    DenseMatrix( const GraphCSx & G, VID v, const VID * assigned_clique,
		 VID num_neighbours, const VID * neighbours ) {
	VID n = G.numVertices();
	EID m = G.numEdges();
	const EID * const gindex = G.getIndex();
	const VID * const gedges = G.getEdges();

	// Set of eligible neighbours
	m_g2s = new VID[n];
	std::fill( &m_g2s[0], &m_g2s[n], ~(VID)0 );
	VID ns = num_neighbours;
	assert( ns <= MAX_VERTICES );

	for( VID i=0; i < ns; ++i ) {
	    VID u = neighbours[i];
	    m_g2s[u] = i;
	}


	// m_words = ( ns + bits_per_lane - 1 ) / bits_per_lane;
	m_words = 4; // VL
	assert( ( ns + bits_per_lane - 1 ) / bits_per_lane <= m_words );
	m_words = std::min( (unsigned)VL, m_words );
	m_matrix = m_matrix_alc = new type[m_words * ns + 4];
	intptr_t p = reinterpret_cast<intptr_t>( m_matrix );
	if( p & 31 ) // 31 = 256 bits / 8 bits per byte - 1
	    m_matrix = &m_matrix[(p&31)/sizeof(type)];
	std::fill( &m_matrix[0], &m_matrix[m_words * ns], 0 );

	m_s2g = new VID[ns];

	// Place edges
	VID ni = 0;
	m_m = 0;
	for( VID su=0; su < ns; ++su ) {
	    VID u = neighbours[su];
	    m_s2g[su] = u;

	    row_type row_u = tr::setzero();

	    contract::detail::intersect_tmpl(
		&neighbours[0], &neighbours[num_neighbours],
		&gedges[gindex[u]], &gedges[gindex[u+1]],
		[&]( VID w ) {
		    const VID * const pos = std::lower_bound(
			&gedges[gindex[u]], &gedges[gindex[u+1]], w );
		    if( pos != &gedges[gindex[u+1]] && *pos == w && u != w ) {
			EID ew = pos - gedges;
			if( ~assigned_clique[ew] == 0 ) {
			    VID sw = m_g2s[w];
			    row_u = tr::bitwise_or( row_u, create_row( sw ) );
			    // set( su, sw ); // TODO perform on register before storing to memory : row=bitwise_or( row, create_row( sw ) )
			    ++m_m;
			}
		    }
		    return true;
		} );

	    tr::store( &m_matrix[VL * su], row_u );
	}

	m_n = ns;
    }


    ~DenseMatrix() {
	delete[] m_g2s;
	delete[] m_s2g;
	delete[] m_matrix_alc;
    }

    
    contract::vertex_set<VID>
    bron_kerbosch() {
	m_mc = tr::setzero();
	m_mc_size = 0;

	for( VID v=0; v < m_n; ++v ) {
	    if( tr::is_zero( get_row( v ) ) )
		continue;

	    row_type vrow = create_row( v );
	    row_type R = vrow;

	    // Consider as candidates only those neighbours of u that are larger
	    // than u to avoid revisiting the vertices unnecessarily.
	    row_type P = tr::bitwise_and( get_row( v ), get_himask( v ) );

	    bk_iterate( R, P, 1 );
	}

	// Erase columns
	for( VID v=0; v < m_n; ++v )
	    tr::store( &m_matrix[m_words * v],
		       tr::bitwise_andnot(
			   m_mc, tr::load( &m_matrix[m_words * v] ) ) );
	
	contract::vertex_set<VID> vset;
	while( !tr::is_zero( m_mc ) ) {
	    VID v;
	    row_type mc_new;
	    std::tie( v, mc_new ) = remove_element( m_mc );
	    assert( v < m_n );
	    m_mc = mc_new;
	    vset.push( v );

	    // Erase rows
	    tr::store( &m_matrix[m_words * v], tr::setzero() );
	}

	return vset;
    }

    contract::vertex_set<VID>
    vertex_cover() {
	m_mc = tr::setone();
	m_mc_size = m_n;

	row_type z = tr::setzero();
	vc_iterate( 0, z, z, 0 );

	contract::vertex_set<VID> vset;
	m_mc = tr::bitwise_invert( m_mc ); // cover vs clique on complement

	// Erase columns
	for( VID v=0; v < m_n; ++v )
	    tr::store( &m_matrix[m_words * v],
		       tr::bitwise_andnot(
			   m_mc, tr::load( &m_matrix[m_words * v] ) ) );
	
	while( !tr::is_zero( m_mc ) ) {
	    VID v;
	    row_type mc_new;
	    std::tie( v, mc_new ) = remove_element( m_mc );
	    m_mc = mc_new;

	    if( v >= m_n )
		break;
	    
	    vset.push( v );

	    // Erase rows
	    tr::store( &m_matrix[m_words * v], tr::setzero() );
	}

	return vset;
    }

    VID numVertices() const { return m_n; }
    EID numEdges() const { return m_m; }

    const VID * get_g2s() const { return m_g2s; }
    const VID * get_s2g() const { return m_s2g; }

private:
    void bk_iterate( row_type R, row_type P, int depth ) {
	// depth == get_size( R )
	if( tr::is_zero( P ) ) {
	    if( depth > m_mc_size ) {
		m_mc = R;
		m_mc_size = depth;

		// Correctness check
		row_type r = m_mc;
		while( !tr::is_zero( r ) ) {
		    VID v;
		    row_type r_new;
		    std::tie( v, r_new ) = remove_element( r );
		    r = r_new;

		    if( v >= m_n )
			break;
	    
		    row_type vv = create_row( v );
		    row_type inc = tr::bitwise_or( get_row( v ), vv );
		    row_type ins = tr::bitwise_and( m_mc, inc );
		    row_type eq = tr::cmpeq( m_mc, ins, target::mt_vmask() );
		    row_type eqi = tr::bitwise_invert( eq );
		    assert( tr::is_zero( eqi ) );
		    assert( tr::cmpeq( m_mc, ins, target::mt_bool() ) );
		}
	    }
	    return;
	}
	if( depth + get_size( P ) < m_mc_size )
	    return;

	row_type x = P;
	while( !tr::is_zero( x ) ) {
	    VID u;
	    row_type x_new;
	    std::tie( u, x_new ) = remove_element( x );
	    row_type u_row = tr::bitwise_andnot( x_new, x );
	    x = x_new;
	    // assert( tr::is_zero( tr::bitwise_and( get_row( u ), create_row( u ) ) ) );
	    row_type Pv = tr::bitwise_and( x, get_row( u ) ); // x vs P?
	    row_type Rv = tr::bitwise_or( R, u_row );
	    bk_iterate( Rv, Pv, depth+1 );
	}
    }

#if 0
    // 1-byte labels work for up to 256 vertices
    VID cc_1b( uint8_t * components ) {
	using trb = vector_type_traits_vl<uint8_t,256>;
	using lvec = typename trb::type;
	
	// Setup initial labels
	for( VID v=0; v < m_n; ++v )
	    components[v] = v;

	// Do SpMV iteration using min,* semiring until convergence
	lvec labels = trb::loadu( components );
	bool changed;
	do {
	    changed = false;
	    lvec upd = trb::setone(); // max value (unsigned)
	    for( VID v=0; v < m_n; ++v ) {
		row_type r = get_row( v );
		lvec c = trb::set1( trb::lane( labels, v ) );
		lvec unc = trb::min( labels, c );
		lvec upd = trb::blend( r, labels, unc );
		if( !changed && trb::cmpne( upd, labels, target::mt_bool() ) )
		    changed = true;
		labels = upd;
	    }
	} while( changed );
	trb::storeu( components, labels );

	// Pick up components
	VID num_components = 0;
	for( VID v=0; v < m_n; ++v ) {
	    if( components[v] == v )
		++num_components;
	}

	return num_components;
    }
#endif

    // cin is a bitmask indicating which vertices are in the cover.
    // It is filled up only up to vertex v. Remaining bits are zero.
    // cout indicates the vertices excluded.
    void vc_iterate( VID v, row_type cin, row_type cout, VID cin_sz ) {
	// Leaf node
	if( v == m_n ) {
	    if( cin_sz < m_mc_size ) {
		m_mc = cin;
		m_mc_size = cin_sz;
	    }
	    return;
	}

	// isolated vertex
	row_type v_set = create_row( v );
	row_type v_row = tr::bitwise_andnot(
	    get_himask( m_n ), tr::bitwise_xnor( v_set, get_row( v ) ) );
	VID deg = get_size( v_row );
	if( deg == 0 ) {
	    vc_iterate( v+1, cin, tr::bitwise_or( cout, v_set ), cin_sz );
	    return;
	}

	// Count number of covered neighbours
	VID num_covered = get_size( tr::bitwise_and( v_row, cin ) );

	// In case we don't have choice: including all neighbours would result
	// in a vertex cover larger than the one of interest. In that case,
	// include the vertex and not the (remaining) neighbours
	if( cin_sz + deg - num_covered >= m_mc_size ) {
	    vc_iterate( v+1, tr::bitwise_or( cin, v_set ), cout, cin_sz+1 );
	    return;
	}

	// All neighbours included, so this vertex is not needed
	// Any neighbour not included, then this vertex must be included
	if( num_covered == deg ) {
	    vc_iterate( v+1, cin, tr::bitwise_or( cout, v_set ), cin_sz );
	    return;
	}

	// Count number of uncovered neighbours; only chance we have any
	// if cout_sz is non-zero
	VID cout_sz = v - cin_sz;
	if( cout_sz > 0 ) {
	    VID num_uncovered = get_size( tr::bitwise_and( v_row, cout ) );
	    if( num_uncovered > 0 ) {
		vc_iterate( v+1, tr::bitwise_or( cin, v_set ), cout, cin_sz+1 );
		return;
	    }
	}

	// Otherwise, try both ways.
	vc_iterate( v+1, cin, tr::bitwise_or( cout, v_set ), cin_sz );

	vc_iterate( v+1, tr::bitwise_or( cin, v_set ), cout, cin_sz+1 );
    }
    
    static std::pair<VID,row_type> remove_element( row_type s ) {
	// find_first includes tzcnt; can be avoided because lane includes
	// a switch, so can check on mask & 1, mask & 2, etc instead of
	// tzcnt == 0, tzcnt == 1, etc
	auto mask = tr::cmpne( s, tr::setzero(), target::mt_mask() );

	type xtr;
	unsigned lane;
	
	if constexpr ( VL == 4 ) {
	    __m128i half;
	    if( ( mask & 0x3 ) == 0 ) {
		half = tr::upper_half( s );
		lane = 2;
		mask >>= 2;
	    } else {
		half = tr::lower_half( s );
		lane = 0;
	    }
	    if( ( mask & 0x1 ) == 0 ) {
		lane += 1;
		xtr = _mm_extract_epi64( half, 1 );
	    } else {
		xtr = _mm_extract_epi64( half, 0 );
	    }
	} else if constexpr ( VL == 2 ) {
	    if( ( mask & 0x1 ) == 0 ) {
		xtr = tr::upper_half( s );
		lane = 1;
	    } else {
		xtr = tr::lower_half( s );
		lane = 0;
	    }
	} else if constexpr ( VL == 1 ) {
	    lane = 0;
	    xtr = s;
	} else
	    assert( 0 && "Oops" );

	assert( xtr != 0 );
	unsigned off = _tzcnt_u64( xtr );
	assert( off != bits_per_lane );
	row_type s_upd = tr::bitwise_and( s, tr::sub( s, tr::setoneval() ) );
	row_type new_s = tr::blend( 1 << lane, s, s_upd );
	return std::make_pair( lane * bits_per_lane + off, new_s );
    }

    row_type get_row( VID v ) {
	return tr::load( &m_matrix[m_words * v] );
    }

    row_type create_row( VID v ) { // TODO: lookup table with 0x1 in precisely one lane
	row_type z = tr::setzero();
	row_type o = tr::setoneval();
	row_type p = tr::sll( o, v % bits_per_lane );
	VID lane = v / bits_per_lane;
	row_type r = tr::blend( 1 << lane, z, p );
	return r;
	
	// return tr::setlane(
	// tr::setzero(), type(1) << ( v % bits_per_lane ), v / bits_per_lane );
    }

    row_type get_himask( VID v ) {
#if 1
	row_type z = tr::setzero();
	row_type s = tr::setone();
	row_type o = tr::setoneval();
	// row_type p = tr::bitwise_invert(
	// tr::sub( tr::sll( o, v % bits_per_lane ), o ) );
	row_type p = tr::sll( s, v % bits_per_lane );
	VID lane = v / bits_per_lane;
	VID mask = ( VID(1) << VL ) - ( VID(1) << lane );
	row_type a = tr::blend( mask, z, s );
	row_type r = tr::blend( 1 << lane, a, p );
	return r;
#else
	VID lane = v / bits_per_lane;
	row_type a = tr::load( &himask_starter[lane * VL] );
	row_type b = tr::slli( a, v % bits_per_lane );
	row_type c = tr::sub( b, a );
	row_type d = tr::srli( a, 1 );
	row_type e = tr::bitwise_or( c, d );
	row_type f = tr::bitwise_invert( e );
	return f;
#endif
    }

    void set( VID u, VID v ) {
	assert( u != v );
	VID word, off;
	std::tie( word, off ) = slocate( u, v );
	type w = type(1) << off;
	m_matrix[word] |= w;
	// assert( tr::is_zero( tr::bitwise_and( get_row( u ), create_row( u ) ) ) );
	// assert( tr::is_zero( tr::bitwise_and( get_row( v ), create_row( v ) ) ) );
    }

    static VID get_size( row_type r ) {
	return target::allpopcnt<VID,type,VL>::compute( r );
    }

    std::pair<VID,VID> slocate( VID u, VID v ) const {
	VID col = v / bits_per_lane;
	VID word = u * VL + col;
	return std::make_pair( word, v % bits_per_lane );
    }
		    
private:
    VID m_n;
    unsigned m_words;
    EID m_m;
    VID * m_g2s;
    VID * m_s2g;
    type * m_matrix;
    type * m_matrix_alc;

    VID m_mc_size;
    row_type m_mc;

    // assumes VL == 4
    alignas(64) static constexpr uint64_t himask_starter[16] = {
	0x1, 0x0, 0x0, 0x0,
	0xffffffffffffffff, 0x1, 0x0, 0x0,
	0xffffffffffffffff, 0xffffffffffffffff, 0x1, 0x0,
	0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0x1
	// 0x1, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
	// 0x0, 0x1, 0xffffffffffffffff, 0xffffffffffffffff,
	// 0x0, 0x0, 0x1, 0xffffffffffffffff,
	// 0x0, 0x0, 0x0, 0x1,
    };
};

template<unsigned Bits, typename Fn>
// VID *
void
clique_partition( const GraphCSx & GG,
		  DenseMatrix<Bits> & G,
		  VID v_reference,
		  Fn && map_i2g,
		  VID * assigned_clique,
		  CliqueLister & clist ) {
    VID n = G.numVertices();
    EID m = G.numEdges();

    // bool * unplaced = new bool[n];
    // std::fill( &unplaced[0], &unplaced[n], true );

    VID to_assign = n;

    // VID * clique_number = new VID[n];
    // std::fill( &clique_number[0], &clique_number[n], ~(VID)0 );

    uint64_t tid = gettid();

    // use clique size 3 as v_reference is included by default
    while( to_assign > 3 ) {
	// Cut out sub-graph of all vertices not yet placed.
	// (Each vertex (neighbour/edge) can only be placed once because the main
	//  vertex v is part of every clique and each edge may be covered once.)
	timer tm;
	tm.start();
	float density = float(m) / ( float(n) * float(n) );

	// TODO:
	// * Instances with m=O(n): prune low-degree vertices?
	// * Instance with m << n: ???
	// * Create version of BK that works without rebuilding graph
	// * Iteratively rebuild graph, i.e., reduce down from previous
	//   previous as opposed to reduce from original
	// * Encode neighbour lists in BK (R, P) as bit masks to accelerate
	//   operations. Speed of popcount? Store graph in dense format?

	VID best_size = n;
	VID cliqno;

	std::cerr << tid << "  subset of unassigned neighbours: "
		  << " n=" << n
		  << " m=" << m
		  << " density=" << density
		  << " launched (dense)\n";

	std::string variant;

	tm.start();

	if( EID(n) * EID( n - 1 ) == m ) {
	    // A clique. Because to_assign > 4, so is in.
	    // Hence, the clique is sufficiently large.
	    
	    variant = "clique";
	    best_size = n;

	    // Get a unique ID for the clique
	    VID * members;
	    std::tie( cliqno, members ) = clist.allocate_clique( n+1 );

	    *members++ = v_reference; 

	    for( VID i=0; i < n; ++i ) {
		VID vg = i; // G.get_s2g()[i];
		// assert( ~clique_number[i] == 0 );
		// clique_number[i] = cliqno;
		*members++ = map_i2g( vg );
	    }

	    // for( VID i=0; i < n; ++i )
	    // assert( clique_number[i] != 0 );

	    mark_edges( GG, v_reference, map_i2g,
			graptor::graph::range_iterator( VID(0) ),
			graptor::graph::range_iterator( n ),
			assigned_clique, cliqno );

	    break; // we are done
	} else if( true ) { // density < 0.1 ) {
	    // for( VID i=0; i < n; ++i )
	    // assert( clique_number[i] != 0 );

	    contract::vertex_set<VID> c = G.bron_kerbosch();
	    variant = "BK-dense";

	    // for( VID i=0; i < n; ++i )
	    // assert( clique_number[i] != 0 );

	    best_size = c.size();
	    
	    if( best_size < 3 )
		break;

	    // Get a unique ID for the clique
	    VID * members;
	    std::tie( cliqno, members )
		= clist.allocate_clique( best_size+1 );

	    *members++ = v_reference; 

	    std::cerr << "          content:";
	    for( auto && v : c ) {
		VID vg = v; // G.get_s2g()[v];
		// assert( ~clique_number[v] == 0 );
		// clique_number[v] = cliqno;
		*members++ = map_i2g( vg );
		// unplaced[v] = false; // no longer consider this vertex
		std::cerr << ' ' << v;
	    }
	    std::cerr << "\n";

	    mark_edges( GG, v_reference, map_i2g,
			c.begin(), c.end(),
			assigned_clique, cliqno );
	} else {
	    contract::vertex_set<VID> c = G.vertex_cover();
	    variant = "VC-dense";

	    best_size = c.size();
	    
	    if( best_size < 3 )
		break;

	    // Get a unique ID for the clique
	    VID * members;
	    std::tie( cliqno, members )
		= clist.allocate_clique( best_size+1 );

	    // clique_number[v_reference] = cliqno;
	    *members++ = v_reference; 

	    std::cerr << "     clique: #" << c.size();
	    for( auto && v : c ) {
		VID vg = v; // G.get_s2g()[v];
		// assert( ~clique_number[v] == 0 );
		// clique_number[v] = cliqno;
		*members++ = map_i2g( vg );
		// unplaced[v] = false; // no longer consider this vertex
		std::cerr << ' ' << v;
	    }
	    std::cerr << "\n";

	    mark_edges( GG, v_reference, map_i2g,
			c.begin(), c.end(),
			assigned_clique, cliqno );

	    // for( VID i=0; i < n; ++i )
	    // assert( clique_number[i] != 0 );
	}

	double t = tm.next();
	std::cerr << tid
		  << ' ' << v_reference
		  << "  subset of unassigned neighbours: "
		  << " n=" << n
		  << " m=" << m
		  << " density=" << density
		  << " size=" << best_size
		  << " cliqno=" << cliqno
		  << " time(" << variant << "): " << t
		  << "\n";

	to_assign -= best_size;
    }

    // delete[] unplaced;

    // return clique_number;
}

// For each vertex in iteration range as well as v_ref, mark edges among them
// The iteration range ibegin...iend need not be sorted
template<typename I2G, typename Iter>
void mark_edges( const GraphCSx & G,
		 VID v_ref,
		 I2G && map_i2g,
		 Iter ibegin, Iter iend,
		 VID * assigned_clique,
		 VID cliqno ) {
    if( ibegin == iend )
	return;

    const EID * const gindex = G.getIndex();
    const VID * const gedges = G.getEdges();

    EID num_edges = 0;

    auto find_and_mark = [&]( const VID * const nb, const VID * const ne,
			      VID v ) {
	const VID * const pos = std::lower_bound( nb, ne, v );
	assert( pos != ne && *pos == v && "must be neighbour" );

	EID e = pos - gedges;
	assert( ~assigned_clique[e] == 0 );
	assigned_clique[e] = cliqno;
	++num_edges;
    };
    
    // Mark edges v_ref -> map_i2g(i in iteration range)
    // Do binary search in neighbour list of v_ref, assuming that size of
    // clique is substantially smaller than degree of v_ref
    const VID * const nb = &gedges[gindex[v_ref]];
    const VID * const ne = &gedges[gindex[v_ref+1]];
    for( Iter i=ibegin; i != iend; ++i ) {
	VID ig = map_i2g( *i );
	find_and_mark( nb, ne, ig );
    }

    for( Iter j=ibegin; j != iend; ++j ) {
	VID jg = map_i2g( *j );

	const VID * const njb = &gedges[gindex[jg]];
	const VID * const nje = &gedges[gindex[jg+1]];

	find_and_mark( njb, nje, v_ref );

	for( Iter i=ibegin; i != iend; ++i ) {
	    if( i != j ) {
		VID ig = map_i2g( *i );
		find_and_mark( njb, nje, ig );
	    }
	}
    }

    EID n = std::distance( ibegin, iend );
    assert( num_edges == n * (n+1) );
}

template<unsigned Bits>
void
clique_partition_neighbours_dense(
    const GraphCSx & G,
    VID v,
    VID * assigned_clique,
    const VID * const coreness,
    CliqueLister & clist,
    VID num_neighbours,
    const VID * neighbours ) {

    // Cut out the vertex and its neighbours, ensure minimum coreness of 4.
    // Remove the vertex v because v will be a member of every clique.
    DenseMatrix<Bits> IG( G, v, assigned_clique, num_neighbours, neighbours );

    // Partition this small graph in cliques
    clique_partition( G, IG, v, [&]( VID vi ) { return IG.get_s2g()[vi]; },
		      assigned_clique, clist );

#if 0
    // for( VID i=0; i < num_neighbours; ++i )
    // assert( clique_number[i] != 0 );

    VID n = G.numVertices();
    EID m = G.numEdges();
    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();

    // Any edges not touched here are undecided
    for( EID e=index[v], ee=index[v+1]; e != ee; ++e ) {
	VID u = edges[e];
	VID iu = IG.get_g2s()[u];

	if( ~iu == 0 )
	    continue;
	if( ~clique_number[iu] == 0 )
	    continue;

	// Edge v->u

	VID cl = clique_number[iu];
	assert( ~assigned_clique[e] == 0 );
	assigned_clique[e] = cl;

	for( EID f=index[u], ff=index[u+1]; f != ff; ++f ) {
	    VID w = edges[f];
	    VID iw = IG.get_g2s()[w];

	    // Edge u->w
	    if( ~iw != 0 && clique_number[iw] == clique_number[iu] ) {
		assert( ~assigned_clique[f] == 0 );
		assigned_clique[f] = cl;
	    } else if( w == v ) { // Edge u->v
		assert( ~assigned_clique[f] == 0 );
		assigned_clique[f] = cl;
	    }
	}
    }

    delete[] clique_number;
#endif
}

void
clique_partition_neighbours_base(
    const GraphCSx & G,
    VID v,
    VID * assigned_clique,
    const VID * const coreness,
    CliqueLister & clist,
    VID num_neighbours,
    const VID * neighbours ) {

    // Cut out the vertex and its neighbours, ensure minimum coreness of 4.
    // Remove the vertex v because v will be a member of every clique.
    GraphBuilderInduced ibuilder(
	G, v, assigned_clique, num_neighbours, neighbours );
	// [=]( VID u ) { return coreness[u] >= 4 && u != v; } );
    auto & IG = ibuilder.get_graph();

    // std::cerr << "neighbourhood of vertex " << v << ": n="
    // << IG.numVertices() << " m=" << IG.numEdges() << "\n";

    // Partition this small graph in cliques
    // VID * clique_number =
    clique_partition( G, IG, v,
		      [&]( VID vi ) { return ibuilder.get_s2g()[vi]; },
		      assigned_clique, clist );

#if 0
    VID n = G.numVertices();
    EID m = G.numEdges();
    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();

    // Any edges not touched here are undecided
    for( EID e=index[v], ee=index[v+1]; e != ee; ++e ) {
	VID u = edges[e];
	VID iu = ibuilder.get_g2s()[u];

	if( ~iu == 0 )
	    continue;
	if( ~clique_number[iu] == 0 )
	    continue;

	// Edge v->u

	VID cl = clique_number[iu];
	assert( ~assigned_clique[e] == 0 );
	assigned_clique[e] = cl;

	for( EID f=index[u], ff=index[u+1]; f != ff; ++f ) {
	    VID w = edges[f];
	    VID iw = ibuilder.get_g2s()[w];

	    // Edge u->w
	    if( ~iw != 0 && clique_number[iw] == clique_number[iu] ) {
		assert( ~assigned_clique[f] == 0 );
		assigned_clique[f] = cl;
	    } else if( w == v ) { // Edge u->v
		assert( ~assigned_clique[f] == 0 );
		assigned_clique[f] = cl;
	    }
	}
    }

    delete[] clique_number;
#endif
}

std::tuple<VID,VID *,VID>
get_induced_set(
    const GraphCSx & G,
    VID v,
    VID * assigned_clique,
    const VID * const coreness ) {

    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();
    VID deg = index[v+1] - index[v];

    // Would be reasonable to use vertex_set here
    VID * iset = new VID[deg];
    VID * degrees = new VID[deg];
    EID tot_degree = 0;
    VID n_iset = 0;

    VID maxdeg = 0;

    for( EID e=index[v], ee=index[v+1]; e != ee; ++e ) {
	if( ~assigned_clique[e] != 0 ) // edge (v,u) already assigned
	    continue;
	VID u = edges[e];
	if( coreness[u] < 4 ) // no interesting cliques
	    continue;
	VID udeg = 0;

	// TODO: keep a counter for number of assigned neighbours for each
	//       vertex so we can avoid the loop
	for( EID f=index[u], ff=index[u+1]; f != ff; ++f ) {
	    if( ~assigned_clique[e] != 0 ) // edge (u,w) already assigned
		continue;
	    VID w = edges[f];
	    const VID * pos
		= std::lower_bound( &edges[index[v]], &edges[index[v+1]], w);
	    if( pos != &edges[index[v+1]] && *pos == w )
		++udeg;
	}
	// TODO: remember degree of each vertex for later use (in sparse case)
	if( udeg >= 4 ) {
	    iset[n_iset] = u;
	    degrees[n_iset] = deg;
	    tot_degree += deg;
	    ++n_iset;
	    if( udeg > maxdeg )
		maxdeg = udeg;
	}
    }

    delete[]degrees; // TODO
    return std::make_tuple( n_iset, iset, maxdeg ); // degrees, tot_degree
}

void
clique_partition_neighbours(
    const GraphCSx & G,
    VID v,
    VID * assigned_clique,
    const VID * const coreness,
    CliqueLister & clist ) {
    // TODO: determine set of vertices first, deducting degrees for
    //       neighbours that have been assigned to cliques in order to
    //       reduce the set of vertices. Then decide whether this problem
    //       can be treated as dense.
    VID * induced_set;
    VID num;
    VID maxdeg;
    std::tie( num, induced_set, maxdeg )
	= get_induced_set( G, v, assigned_clique, coreness );

    // Need degree 3 or higher to get 4-clique after inclusion of v
    if( maxdeg < 3 )
	return;
    
    if( num <= 64 ) {
	clique_partition_neighbours_dense<64>(
	    G, v, assigned_clique, coreness, clist,
	    num, induced_set );
    } else if( num <= 128 ) {
	clique_partition_neighbours_dense<128>(
	    G, v, assigned_clique, coreness, clist,
	    num, induced_set );
    } else if( num <= 256 ) {
	clique_partition_neighbours_dense<256>(
	    G, v, assigned_clique, coreness, clist,
	    num, induced_set );
    } else {
	clique_partition_neighbours_base(
	    G, v, assigned_clique, coreness, clist,
	    num, induced_set );
    }

    delete[] induced_set;
}


void checkIS( const GraphCSx & G, frontier & f ) {
    VID * vlist = f.getSparse();
    VID k = f.nActiveVertices();

    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();

    for( VID i=0; i < k; ++i ) {
	VID v = vlist[i];

	for( EID e=index[v], ee=index[v+1]; e != ee; ++e ) {
	    VID u = edges[e];
	    const VID * pos = std::lower_bound( &vlist[0], &vlist[k], u );
	    assert( pos == &vlist[k] || *pos != u );
	}
    }
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
    auto & coreness = kcore.getCoreness();

#if 1
    std::cerr << "Calculating sort order...\n";
    mm::buffer<VID> order( n, numa_allocation_interleaved() );
    mm::buffer<VID> rev_order( n, numa_allocation_interleaved() );

    // Alternative sort order: determine for each vertex the largest clique that
    // contains it, then sort by decreasing clique size. This should guarantee
    // finding the maximum clique. Will take as long as finding the max clique.
    sort_order( order.get(), rev_order.get(),
		coreness.get_ptr(), n, kcore.getLargestCore() );
#endif

    const partitioner & part = GA.get_partitioner();
#if 0
    api::vertexprop<VID,VID,var_priority>
	priority( GA.get_partitioner(), "priority" );
    expr::array_ro<VID, VID, var_degrees_ro> degree(
	const_cast<VID *>( GA.getCSR().getDegree() ) );

    make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) { return priority[v] = _0; } )
	    .materialize();

    // Evaluates to true if s has priority over d
    auto prio_fn = [&]( auto s, auto d ) {
#if LLF
	return expr::make_unop_lzcnt<VID>( degree[s] )
	    < expr::make_unop_lzcnt<VID>( degree[d] )
	    || ( expr::make_unop_lzcnt<VID>( degree[s] )
		 == expr::make_unop_lzcnt<VID>( degree[d] )
		 && s < d );
#else
	return degree[s] > degree[d]
	    || ( degree[s] == degree[d] && s < d );
#endif
    };
    
    frontier roots;
    api::edgemap(
	GA,
	api::relax( [&]( auto s, auto d, auto e ) {
	    return priority[d] += _p( _1(priority[d]), prio_fn( s, d ) );
	} ),
	api::record( roots, [&]( auto d ) { return priority[d] == _0; },
		     api::strong )
	)
	.materialize();
#endif

    mm::buffer<VID> assigned_clique( m, numa_allocation_interleaved() );
    std::fill( &assigned_clique[0], &assigned_clique[m], ~(VID)0 );

    CliqueLister clist( m );

    // Could partition this loop for parallel processing by computing a MIS,
    // then performing this operation for all vertices in independent set;
    // apply iteratively.

#if 0
    VID vdone = 0;

    while( !roots.isEmpty() ) {
	frontier new_roots;

	// Requires active vertices to be three hops apart such that
	// their neighbourhoods do not intersect.
	// As an approximation, could peel the root set apart such that
	// vertices two hops apart are not treated in parallel.
        // This will have limited parallelism.
	// A better version would use a more dynamic approach with a ready
	// queue per thread and waking up vertices asynchronously. Then we
	// only need a limited root set to start with. However - how do the
	// wakeup? Can't just look at the neighbours.
	roots.toSparse( part );

	checkIS( G, roots );

	VID k = roots.nActiveVertices();
	const VID * vlist = roots.getSparse();

	std::cerr << "Iteration with " << k << " roots; completed " << vdone
		  << "/" << n << "\n";
	
	parallel_loop( VID(0), k, 1, [&]( VID i ) {
	    VID v = vlist[i];
	    // std::cerr << "i=" << i << " v=" << v << " c(v)=" << coreness[v] << "\n";

	    clique_partition_neighbours( G, v, assigned_clique.get(),
					 coreness.get_ptr(), clist );

	    std::cerr << "  vertex " << v << " complete\n";
	} );

	vdone += k;

	api::edgemap(
	    GA,
	    api::filter( api::src, api::strong, roots ),
	    api::relax( [&]( auto s, auto d, auto e ) {
		return priority[d].count_down( _0(priority[d]) );
	    } ),
	    api::record( new_roots, api::reduction, api::strong )
	    )
	    .materialize();

	roots.del();
	roots = new_roots;
    }
#else
    for( VID i=0; i < n; ++i ) {
	VID v = order[i];

	std::cerr << "Iteration " << i << " with v=" << v
		  << " deg=" << G.getDegree( v )
		  << "\n";
	
	clique_partition_neighbours( G, v, assigned_clique.get(),
				     coreness.get_ptr(), clist );

	std::cerr << "  vertex " << v << " complete\n";
    }
#endif

    // Enables core methods on the CompressedList such as copy
    clist.finalize();

    // Write graph
    VID num_cliques = clist.get_num_cliques();
    EID num_members = clist.get_num_members();
    EID num_clique_edges = clist.get_num_clique_edges();

    std::cerr << "Found n=" << n
	      << " m=" << m
	      << " num_cliques=" << num_cliques
	      << " num_members=" << num_members
	      << " num_clique_edges=" << num_clique_edges
	      << " num_remaining_edges=" << ( m - num_clique_edges )
	      << " edges-covered=" << float(num_clique_edges)/float(m)
	      << "\n";

    GraphPDG<VID,EID,VID>
	PDG( G.numVertices(), G.numEdges(), num_cliques, num_members,
	     num_clique_edges );

    // Set the membership for the cliques
    auto & cliques = PDG.get_cliques();
    cliques.get_corpus().copy( clist.get_list() );

    // Link vertices to their cliques
    EID * const cclindx = cliques.get_corpus().get_index();
    VID * const cmembers = cliques.get_corpus().get_members();
    EID * const cindex = cliques.get_links().get_index();
    VID * const cedges = cliques.get_links().get_members();

    // This could make use of SAPCo sort ideas
    // Set counts of incident cliques to zero
    parallel_loop( VID(0), n, [=]( VID v ) { cindex[v] = 0; } );

    // Count occurence of each vertex - once per clique
    parallel_loop( EID(0), num_members, [=]( EID e ) {
	__sync_fetch_and_add( &cindex[cmembers[e]], 1 );
    } );

    // Prefix scan
    EID mms = sequence::plusScan( cindex, cindex, n );
    assert( mms == num_members && "clique count error" );
    cindex[n] = num_members;

    // Map vertices to their cliques
    parallel_loop( VID(0), num_cliques, [&]( VID c ) {
	for( EID e=cclindx[c], ee=cclindx[c+1]; e != ee; ++e ) {
	    VID v = cmembers[e];
	    EID f = __sync_fetch_and_add( &cindex[v], 1 );
	    cedges[f] = c;
	}
    } );

    // Restore index - could parallelize
    EID idx = 0;
    for( VID v=0; v < n; ++v ) {
	EID tmp = cindex[v];
	cindex[v] = idx;
	idx = tmp;
    }

    std::cerr << "Sorting cliques...\n";

    // Sort clique membership lists
    parallel_loop( VID(0), num_cliques, [&]( VID c ) {
	std::sort( &cmembers[cclindx[c]], &cmembers[cclindx[c+1]] );
    } );
    
    // Sort vertex-to-clique lists
    parallel_loop( VID(0), n, [&]( VID v ) {
	std::sort( &cedges[cindex[v]], &cedges[cindex[v+1]] );
    } );

    std::cerr << "Remainder graph...\n";

    auto & elist = PDG.get_edges();
    EID * const index = elist.get_index();
    VID * const edges = elist.get_members();
    const EID * const gindex = G.getIndex();
    const VID * const gedges = G.getEdges();

    // Set counts of remaining edges per vertex to zero
    parallel_loop( VID(0), n, [=]( VID v ) { index[v] = 0; } );

    // Count incident edges
    	// parallel_loop( EID(0), m, [=]( EID e ) {
    for( EID e=0; e < m; ++e ) {
	if( ~assigned_clique[e] == 0 ) {
		VID u = gedges[e];
		++index[u];
		// __sync_fetch_and_add( &index[u], 1 );
	}
    } // );

    // Prefix scan
    mms = sequence::plusScan( index, index, n );
    assert( mms == m - num_clique_edges && "edge count error" );
    index[n] = mms;

    // Place edges
    parallel_loop( VID(0), n, [&]( VID v ) {
	for( EID e=gindex[v], ee=gindex[v+1]; e != ee; ++e ) {
	    if( ~assigned_clique[e] == 0 ) {
		VID u = gedges[e];
		EID f = __sync_fetch_and_add( &index[u], 1 );
		edges[f] = v;
	    }
	}
    } );

    // Restore index - could parallelize
    idx = 0;
    for( VID v=0; v < n; ++v ) {
	EID tmp = index[v];
	index[v] = idx;
	idx = tmp;
    }

    // Sort edge lists
    parallel_loop( VID(0), n, [&]( VID v ) {
	std::sort( &edges[index[v]], &edges[index[v+1]] );
    } );

    // Write to files
    std::cerr << "Writing to directory " << odir << "\n";
    PDG.write_file( odir );

    return 0;
}
