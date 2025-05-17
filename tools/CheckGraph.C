#include <cstdlib>
#include <unistd.h>
#include <cstdio>
#include <string>

#include "graptor/graptor.h"
#include "graptor/graph/cgraph.h"
#include "graptor/cmdline.h"

int main( int argc, char *argv[] ) {
    CommandLine P(
	argc, argv,
	"\t-s\t\tinput graph is symmetric (only required for specific formats)\n"
	"\t-i {file}\tinput file containing graph\n"
	"\t-weights {file}\tinput file for weights of graph\n"
	"\t--max {number}\tmaximum number of errors of each type to report [default: 10]\n"
	);

    const bool symmetric = P.get_bool_option( "-s" );
    const char * ifile = P.get_string_option( "-i" );
    const char * weights = P.get_string_option( "-weights" );
    const EID max_err = P.get_long_option( "--max", 100 );

    GraphCSx G( ifile, -1, symmetric, weights );

    std::cout << "Vertices: " << G.numVertices()
	      << "\nEdges: " << G.numEdges() << "\n";
    
    EID err_sort = 0, err_dup = 0, err_range = 0;
    EID m_self = 0;
    VID n_isolated = 0;

    // Validate if all neighbour lists are sorted, and unique
    const EID * index = G.getIndex();
    const VID * edges = G.getEdges();
    VID n = G.numVertices();
    parallel_for( VID v=0; v < n; ++v ) {
	EID es = index[v];
	EID ee = index[v+1];

	if( es == ee )
	    ++n_isolated;
	
	for( EID e=es+1; e < ee; ++e ) {
	    if( edges[e] == v )
		++m_self;
	    
	    if( edges[e] >= n ) {
		if( err_range++ < max_err )
		    std::cerr << "edge " << e << " connecting to vertex " << v
			      << " which is out of range\n";
	    }
	    if( edges[e-1] > edges[e] ) {
		if( err_sort++ < max_err )
		    std::cerr << "violating sort order v=" << v
			      << " e=" << (e-1) << " to " << edges[e-1]
			      << "and e=" << e << " to " << edges[e]
			      << "\n";
	    }
	    else if( edges[e-1] == edges[e] ) {
		if( err_dup++ < max_err )
		    std::cerr << "duplicate subsequent edges v=" << v
			      << " e=" << (e-1) << "," << e
			      << " to " << edges[e] << "\n";
	    }
	}
    }

    bool any_error = err_sort > 0 || err_dup > 0 || err_range > 0;

    EID m = G.numEdges();
    if( index[n] != m ) {
	std::cerr << "End of edge list marker " << index[n]
		  << " does not match number of edges " << m << "\n";
	any_error = true;
    }

    std::cerr << "checked " << n << " vertices\n"
	      << "sort order errors: " << err_sort
	      << "\nduplicate edges errors: " << err_dup
	      << "\nvertex out of range errors: " << err_range
	      << "\nisolated vertices: " << n_isolated
	      << "\nself-edges: " << m_self
	      << "\n";

    return any_error ? 1 : 0;
}
