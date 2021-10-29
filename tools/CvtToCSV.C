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
    
    const EID * idx = G.getIndex();
    const VID * edge = G.getEdges();
    VID n = G.numVertices();
    EID m = G.numEdges();

    for( VID s=0; s < n; ++s ) {
	EID from = idx[s];
	EID to = idx[s+1];
	for( EID i=from; i < to; ++i ) {
	    VID d = edge[i];
	    std::cout << s << ' ' << d << '\n';
	}
    }

    return 0;
}
