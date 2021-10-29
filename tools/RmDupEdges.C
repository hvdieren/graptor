#include <cstdlib>
#include <unistd.h>
#include <cstdio>
#include <string>

#include "graptor/graptor.h"
#include "graptor/graph/cgraph.h"

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    char* iFile = P.getArgument(0);
    bool symmetric = P.getOptionValue("-s");
    bool binary = P.getOptionValue("-b");             //Galois binary format

    const char * ofile = P.getOptionValue( "-o" );

    GraphCSx G( iFile, -1 );

    std::cerr << "Read graph.\n";
    
    EID * idx = G.getIndex();
    VID * edge = G.getEdges();
    VID n = G.numVertices();
    EID m = G.numEdges();

    EID um = 0;

    // Let's assume the graph is sorted.
    // Modify it in place, then write back.
    for( VID s=0; s < n; ++s ) {
	EID from = idx[s];
	EID to = idx[s+1];

	// Updated info for this vertex
	idx[s] = um;

	// First edge is kept always (um may equal from initially...)
	edge[um] = edge[from];
	++um;

	for( EID i=from+1; i < to; ++i ) {
	    VID v = edge[i];
	    if( v != edge[um-1] ) { // keep it
		edge[um] = v;
		++um;
	    }
	}
    }
    idx[n] = um;

    std::cerr << "Original: m=" << m << ", updated: um=" << um << "\n";

    G.setNumEdges( um );
    G.writeToBinaryFile( ofile );

    std::cerr << "Done writing file.\n";

    return 0;
}
