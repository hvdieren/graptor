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
	"\t-weights {file}\toutput file for weights of graph\n"
	);

    const bool symmetric = P.get_bool_option("-s");
    const char * ifile = P.get_string_option( "-i" );
    const char * weights = P.get_string_option("-weights"); // file with weights

    std::cerr << "Reading graph " << ifile << "...\n";
    GraphCSx G( ifile, -1, symmetric, weights );

    // Validate if all neighbour lists are sorted, and unique
    const EID * index = G.getIndex();
    const VID * edges = G.getEdges();
    VID n = G.numVertices();
    parallel_for( VID v=0; v < n; ++v ) {
	EID es = index[v];
	EID ee = index[v+1];
	for( EID e=es+1; e < ee; ++e ) {
	    if( edges[e] >= n )
		std::cerr << "edge " << e << " connecting to vertex " << v
			  << " which is out of range\n";
	    if( edges[e-1] > edges[e] )
		std::cerr << "violating sort order v=" << v
			  << " e=" << (e-1) << " to " << edges[e-1]
			  << "and e=" << e << " to " << edges[e]
			  << "\n";
	    else if( edges[e-1] == edges[e] )
		std::cerr << "duplicate subsequent edges v=" << v
			  << " e=" << (e-1) << "," << e
			  << " to " << edges[e] << "\n";
	}
    }

    std::cerr << "checked " << n << " vertices\n";

    return 0;
}
