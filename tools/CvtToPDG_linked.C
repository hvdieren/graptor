// -*- c++ -*-
// A Pattern Decomposed Graph
#include "graptor/graptor.h"
#include "graptor/api.h"

#include "graptor/graph/contract/vertex_set.h"
#include "graptor/graph/GraphCSx.h"
#include "graptor/graph/GraphPDG.h"

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
    EdgeCliquePartition( const GraphCSx & G )
	: m_G( G ),
	  m_vertex_cliques( G.numEdges(), numa_allocation_interleaved() ),
	  m_sets( G.numEdges(), numa_allocation_interleaved() ),
	  m_singleton( G.numVertices(), numa_allocation_interleaved() ),
	  m_num_2cliques( 0 ) {
	m_cliquelings.resize( G.numEdges() );
	parallel_loop( EID(0), G.numEdges(), [&]( EID i ) {
	    m_vertex_cliques[i] = i;
	    m_sets[i] = 0;
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
	for( VID v=0; v < n; ++v ) {
	    const EID * const index = m_G.getIndex();
	    const VID * const edges = m_G.getEdges();

	    EID e = index[v];
	    EID ee = index[v+1];
	    for( ; e != ee; ++e ) {
		VID u = edges[e];
		// Only visit each edge once. Ignore self-edges.
		// Assumes neighbour list is sorted in increasing order.
		if( u >= v )
		    break;

		// Use an ordering rule to keep trees flat
		VID a = v;
		VID b = u;
		if( order( a ) > order( b ) ) {
		    std::swap( a, b );
		}

		// m_locks[a].lock();
		// m_locks[b].lock();

		process_edge( a, b );

		// m_locks[b].unlock();
		// m_locks[a].unlock();
	    }

	    if( v % 10000 == 0 ) {
		std::cerr << "@" << tm.next() << " v=" << v << ": ";
		sizes();
	    } else if( v % 1000 == 0 ) {
		std::cerr << " v=" << v << "\n";
	    }
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

	for( VID v=0; v < n; ++v ) {
	    lindex[v] = lindex[v+1];
	    rindex[v] = rindex[v+1];
	}
	lindex[n] = ml;
	rindex[n] = mr;
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
    VID order( VID v ) const {
	// return m_G.numVertices() - v;
	return v;
    }

    /**
     * With u < v, u and v are neighbours
     */
    void process_edge( VID u, VID v ) {
	assert( order( u ) < order( v ) && "assumption / requirement" );

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
		// std::cerr << "left-over Cu: @" << get_id( c ) << " " << c << '\n';
	    }

	    M.swap( Cv.get_vertices() );
	    Cv.clear(); // Cv no longer necessary

	    // Create left-over cliqueling for Cv, if any
	    if( !Cvr.empty() ) {
		distribute_candidates( M, Cvr );
		// std::cerr << "left-over Cv: @" << get_id( c ) << " " << c << '\n';
	    }
	} else {
	    std::cerr << "edge " << u << ',' << v << " is 2-clique\n";
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
	for( ; e != ee; ++e ) {
	    intptr_t p = tree_find( m_vertex_cliques[e] );
	    Cliqueling & c = m_cliquelings[p];
	    if( c.empty() )
		break;
	    assert( c.get_vertices().contains( v ) );
	    cnt += c.get_vertices().size() - 1 + c.get_candidates().size();
	}
	assert( cnt == index[v+1] - index[v] );
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
		std::cerr << "NOTE: member lists have common element: "
			  << *bu << "\n";
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
    std::tuple<EID,EID,EID>
    three_way_split( EID Cu,
		     EID Mv,
		     EID Cv ) {
	const VID * const edges = m_G.getEdges();
	constexpr EID eol = ~EID(0);
	EID M_head = eol, M_tail = eol;
	EID C_head = eol, C_tail = eol;
	EID Cr_head = eol, Cr_tail = eol;

	while( Cu != eol ) {
	    // Scan past elements of Mv lower than *Cu
	    while( Mv != eol && edges[Mv] > edges[Cu] )
		Mv = m_sets[Mv];
	    // Scan past elements of Cv lower than *Cu
	    while( Cv != eol && edges[Cv] > edges[Cu] )
		Cv = m_sets[Cv];
	    
	    if( edges[Cu] == edges[Mv] ) {
		// Element in Cu is a member of Mv: exclude
	    } else if( edges[Cu] == edges[Cv] ) {
		// Element in Cu is a member of Cv: move to C
		if( C_tail == eol )
		    C_head = C_tail = Cu;
		else {
		    m_sets[C_tail] = Cu;
		    C_tail = Cu;
		}
	    } else {
		if( Cr_tail == eol )
		    Cr_head = Cr_tail = Cu;
		else {
		    m_sets[Cr_tail] = Cu;
		    Cr_tail = Cu;
		}
	    }

	    Cu = m_sets[Cu];
	}
    }
    
    void
    swizzle_sets( VID u,
		  VID v,
		  EID Mu,
		  EID Cu,
		  EID Mv,
		  EID Cv,
		  EID su, // singleton set location
		  EID sv ) {
	const VID * const edges = m_G.getEdges();
	constexpr EID eol = ~EID(0);
	EID M_head = eol, M_tail = eol;
	EID C_head = eol, C_tail = eol;
	EID Cur_head = eol, Cur_tail = eol;
	EID Cvr_head = eol, Cvr_tail = eol;

	while( Cu != eol && Cv != eol ) {
	    // Advance Cu or Cv pointer?
	    if( edges[Cu] <= edges[Cv] ) {
		// Scan past elements of Mv lower than *Cu and move
		// them onto M
		while( Mv != eol && edges[Mv] > edges[Cu] ) {
		    if( M_tail == eol )
			M_head = M_tail = Mv;
		    else {
			m_sets[M_tail] = Mv;
			M_tail = Mv;
		    }
		    Mv = m_sets[Mv];
		}
	    
		if( edges[Cu] == edges[Mv] ) {
		    // Element in Cu is a member of Mv: exclude from C
		} else if( edges[Cu] == edges[Cv] ) {
		    // Element in Cu is a member of Cv: move to C
		    if( C_tail == eol )
			C_head = C_tail = Cu;
		    else {
			m_sets[C_tail] = Cu;
			C_tail = Cu;
		    }
		} else {
		    if( Cur_tail == eol )
			Cur_head = Cur_tail = Cu;
		    else {
			m_sets[Cur_tail] = Cu;
			Cur_tail = Cu;
		    }
		}
		Cu = m_sets[Cu];
	    } else {
		// Scan past elements of Mu lower than *Cv and move
		// them onto M
		while( Mu != eol && edges[Mu] > edges[Cv] ) {
		    if( M_tail == eol )
			M_head = M_tail = Mu;
		    else {
			m_sets[M_tail] = Mu;
			M_tail = Mu;
		    }
		    Mu = m_sets[Mu];
		}
	    
		if( edges[Cv] == edges[Mv] ) {
		    // Element in Cv is a member of Mv: exclude from C
		} else if( edges[Cu] == edges[Cv] ) { // impossible ...
		    // Element in Cu is a member of Cv: move to C
		    if( C_tail == eol )
			C_head = C_tail = Cv;
		    else {
			m_sets[C_tail] = Cv;
			C_tail = Cv;
		    }
		} else {
		    if( Cvr_tail == eol )
			Cvr_head = Cvr_tail = Cv;
		    else {
			m_sets[Cvr_tail] = Cv;
			Cvr_tail = Cv;
		    }
		}
		Cv = m_sets[Cv];
	    }
	}
	// Process remaining Cu, Cv, Mu, Mv
    }
#if 0
	{
	const EID * const index = m_G.getIndex();
	const VID * const edges = m_G.getEdges();

	// Every element on Mu, Cu, Mv, Cv needs to land somewhere.
	// M: those in Mu or Mv (union)
	// C: those in Cu and Cv, but not in M (intersect)
	// Cur: those in Cu that do not land in C
	// Cvr: those in Cv that do not land in C
	// 
	// Split Cu in three parts: Mv | Cu and Cv | remainder
	//
	// Memory layout: sets are embedded in the neighbour lists of u and v.
	// The sets Mu, Cu, Mv, Cv will not sit in the space of the neighbour
	// lists of u respectively v. The merged sets may remain where they
	// are. The remainder sets must be merged with the |M|=1 set of u/v.
	intptr_t C, Cur, Cvr;

	// We will try to keep M within the neighbour list of whatever
	// vertex it is currently sitting in. All elements of Mu and Mv are
	// neighbours of this vertex. Those elements will be in Cv (for Mu)
	// and in Cu (for Mv).
	// We will build M in the place of Mu. Hence, the starting point for
	// M is the head of Mu and Mv, whichever is smaller.
	VID M_first = std::min( edges[Mu], edges[Mv] );
	EID M_head = find_e( u, M_first, Mu );
	EID M_tail = M_head;
	
	// We have alread taken one element from Mu/Mv. Set pointers accordingly
	EID Mu_next = edges[Mu] < edges[Mv] ? m_sets[Mu] : Mu;
	EID Mv_next = edges[Mu] < edges[Mv] ? Mv : m_sets[Mv];

	while( m_sets[Mu_next] != ~EID(0) && m_sets[Mv_next] != ~EID(0) ) {
	    if( edges[Mu_next] < edges[Mv_next] ) {
		m_sets[M_tail] = Mu_next;
		Mu_next = m_sets[Mu_next];
	    } else {
		m_sets[M_tail] = Mv_next;
		Mv_next = m_sets[Mv_next];
	    }
	}
	while( m_sets[Mu_next] != ~EID(0) ) {
	    m_sets[M_tail] = Mu_next;
	    Mu_next = m_sets[Mu_next];
	}
	while( m_sets[Mv_next] != ~EID(0) ) {
	    m_sets[M_tail] = Mv_next;
	    Mv_next = m_sets[Mv_next];
	}
	m_sets[M_tail] = ~EID(0);
    }
#endif

    EID find_e( VID v, VID ngh, EID hint ) {
	const EID * const index = m_G.getIndex();
	const VID * const edges = m_G.getEdges();
	if( edges[hint] == ngh
	    && index[v] <= hint && hint < index[v+1] )
	    return hint;

	const VID * p
	    = std::lower_bound( &edges[index[v]], &edges[index[v+1]], ngh );
	assert( p != &edges[index[v+1]] );
	assert( *p == ngh );
	return p - &edges[index[v]];
    }
	
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
    mm::buffer<intptr_t> m_vertex_cliques;
    mm::buffer<VID> m_sets;
    mm::buffer<intptr_t> m_singleton;
    std::vector<Cliqueling> m_cliquelings;
    size_t m_num_2cliques;
};

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

    EdgeCliquePartition ECP( G );
    ECP.find_cliques();

    VID num_cliques;
    EID num_members, num_edges;
    std::tie( num_cliques, num_members, num_edges )
	= ECP.count_cliques( min_size );

    std::cerr << "cliques: " << num_cliques
	      << "\nmembers: " << num_members
	      << "\nedges: " << num_edges
	      << "\n";

    GraphPDG<VID,EID,VID>
	PDG( G.numVertices(), G.numEdges(), num_cliques, num_members,
	     num_edges );

    ECP.write_cliques( PDG.get_cliques(), PDG.get_edges(), min_size );

    std::cerr << "Writing to directory " << odir << "\n";
    PDG.write_file( odir );

    return 0;
}
