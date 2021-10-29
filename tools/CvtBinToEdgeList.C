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

/*
    std::cerr << "Reading Ligra graph " << iFile << "\n";
    wholeGraph<asymmetricVertex> WG =
	readGraph<asymmetricVertex>((char *)iFile,false,binary);
    std::cerr << "Convert to GraphCSx\n";
    GraphCSx G( WG, -1 );
*/

    GraphCSx G( iFile, -1 );

    std::cerr << "Read graph.\n";

    {
	std::cerr << "Create COO.\n";
	GraphCOO GCSR( G, -1 );

	std::cerr << "Rejig data.\n";
	uint64_t n = GCSR.numVertices();
	uint64_t m = GCSR.numEdges();

	std::cerr << "Write to file.\n";
	std::string csrfile = std::string( ofile );
	ofstream file( csrfile, ios::out | ios::trunc );

	for( EID i=0; i < m; ++i )
	    file << GCSR.getSrc()[i] << ' ' << GCSR.getDst()[i] << '\n';

	GCSR.del();

	file.close();
	std::cerr << "Wrote edge list.\n";
    }

    return 0;
}
