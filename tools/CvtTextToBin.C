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
    bool binary = P.getOptionValue("-b"); // Galois

    const char * ofile = P.getOptionValue( "-o" );

    if( symmetric ) {
	wholeGraph<symmetricVertex> WG =
	    readGraph<symmetricVertex>( iFile, false, binary );
	std::cerr << "Read graph (symmetric).\n";
    
	GraphCSx G( WG, -1 );
	std::cerr << "Converted graph.\n";

	G.writeToBinaryFile( ofile );
	std::cerr << "Wrote graph.\n";
    } else {
	wholeGraph<asymmetricVertex> WG =
	    readGraph<asymmetricVertex>( iFile, false, binary );
	std::cerr << "Read graph (asymmetric).\n";
    
	GraphCSx G( WG, -1 );
	std::cerr << "Converted graph.\n";

	G.writeToBinaryFile( ofile );
	std::cerr << "Wrote graph.\n";
    }

    return 0;
}
