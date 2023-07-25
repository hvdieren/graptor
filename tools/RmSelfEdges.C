#include <cstdlib>
#include <unistd.h>
#include <cstdio>
#include <string>

#include "graptor/graptor.h"
#include "graptor/graph/cgraph.h"

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    bool symmetric = P.getOptionValue( "-s" );

    const char * ifile = P.getOptionValue( "-i" );
    const char * wfile = P.getOptionValue("-weights"); // file with weights
    const char * ofile = P.getOptionValue( "-o" );

    std::cerr << "Reading graph " << ifile << "...\n";

    GraphCSx G( ifile, -1, symmetric, wfile );

    VID n = G.numVertices();
    EID m = G.numEdges();
    EID * index = G.getIndex();
    VID * edges = G.getEdges();
    float * weights = G.getWeights() ? G.getWeights()->get() : nullptr;

    std::cerr << "  |V|=" << n << "\n";
    std::cerr << "  |E|=" << m << "\n";

    mmap_ptr<EID> new_index( n );

    std::cerr << "Finding self-edges\n";
    parallel_for( VID u=0; u < n; ++u ) {
	EID es = index[u];
	EID ee = index[u+1];
	EID n_self = 0;
	for( EID e=es; e < ee; ++e ) {
	    VID v = edges[e];
	    if( v == u )
		++n_self;
	}
	new_index[u] = n_self;
    }

    std::cerr << "Rebuilding index\n";
    EID tmp = 0;
    for( VID u=0; u < n; ++u ) {
	EID nxt = index[u+1] - index[u] - new_index[u];
	new_index[u] = tmp;
	tmp += nxt;
    }
    new_index[n] = tmp;

    std::cerr << "Creating new graph object\n";
    GraphCSx UG( n, tmp, -1 );
    EID * uidx = UG.getIndex();
    VID * uedge = UG.getEdges();

    std::cerr << "Rebuilding edge list (" << tmp << " edges)\n";

    parallel_for( VID u=0; u < n; ++u ) {
	EID es = index[u];
	EID ee = index[u+1];
	EID j = new_index[u];
	uidx[u] = j;
	for( EID e=es; e < ee; ++e ) {
	    VID v = edges[e];
	    if( v != u )
		uedge[j++] = v;
	}
    }
    uidx[n] = new_index[n];

    std::cerr << "Writing graph\n";
    UG.writeToBinaryFile( ofile );

    std::cerr << "Done\n";

    return 0;
}
