#include <cstdlib>
#include <unistd.h>
#include <cstdio>
#include <string>

#include "graptor/graptor.h"

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    const char * ofile = P.getOptionValue( "-o" );
    const char * ifile = P.getOptionValue( "-i" );
    const char * wfile = P.getOptionValue("-weights"); // file with weights

    std::cerr << "Reading graph " << ifile << "...\n";

    GraphCSx G( ifile, -1, false,
		wfile != nullptr ? "vertex_weights" : nullptr );

    std::cerr << "Vertices: " << G.numVertices()
	      << "\nEdges: " << G.numEdges()
	      << "\nSymmetric: " << G.isSymmetric()
    	      << "\nCreating output file " << ofile << "...\n";

    G.writeToBinaryFile( ofile );
    if( wfile != nullptr )
	G.writeWeightsToBinaryFile( wfile );

    return 0;
}
