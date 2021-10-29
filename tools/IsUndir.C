#include <cstdlib>
#include <unistd.h>
#include <cstdio>
#include <string>

#include "graptor/graptor.h"
#include "graptor/graph/cgraph.h"

bool hasEdge( GraphCSx & G, VID src, VID dst ) {
    const EID * idx = G.getIndex();
    const VID * edge = G.getEdges();

    for( EID i=idx[src]; i < idx[src+1]; ++i )
	if( edge[i] == dst )
	    return true;
    return false;
}

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    char* iFile = P.getArgument(0);
    bool symmetric = P.getOptionValue("-s");
    bool binary = P.getOptionValue("-b");             //Galois binary format

    const char * ofile = P.getOptionValue( "-o" );

    GraphCSx G( iFile, -1 );

    std::cerr << "Read graph.\n";
    
    const EID * idx = G.getIndex();
    const VID * edge = G.getEdges();
    VID n = G.numVertices();
    EID m = G.numEdges();

    std::cerr << "Graph: n=" << n << " m=" << m << std::endl;

    bool ok = true;
    parallel_for( VID s=0; s < n; ++s ) {
	EID i = idx[s];
	EID j = idx[s+1];
	for( ; i < j; ++i ) {
	    if( !hasEdge( G, edge[i], s ) ) {
		std::cerr << "Edge s=" << s << " d=" << edge[i] << " not reversed\n";
		ok = false;
	    }
	}
    }

    if( ok )
	std::cerr << "Graph is undirected\n";
    else
	std::cerr << "Graph is directed\n";

    G.del();

    return 0;
}
