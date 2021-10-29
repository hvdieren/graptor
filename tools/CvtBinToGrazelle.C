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
	uint64_t * data = new uint64_t[m*2];

	for( EID i=0; i < m; ++i ) {
	    data[2*i] = GCSR.getSrc()[i];
	    data[2*i+1] = GCSR.getDst()[i];
	}
	GCSR.del();

	std::cerr << "Write to file.\n";
	std::string csrfile = std::string( ofile ) + "-push";
	ofstream file( csrfile, ios::out | ios::trunc | ios::binary );

	file.write( (const char *)&n, sizeof(n) );
	file.write( (const char *)&m, sizeof(m) );
	file.write( (const char *)data, sizeof(*data) * m * 2 );
	
	file.close();
	delete[] data;
	std::cerr << "Push version created.\n";
    }

    {
	std::cerr << "Transpose.\n";
	uint64_t n = G.numVertices();
	uint64_t m = G.numEdges();
	GraphCSx Gt( n, m, -1 );
	Gt.import_transpose( G );
	G.del();

	std::cerr << "Create COO of transpose.\n";
	GraphCOO GCSC( Gt, -1 );
	Gt.del();
	
	std::cerr << "Rejig data.\n";
	uint64_t * data = new uint64_t[m*2];
	for( EID i=0; i < m; ++i ) {
	    // Still in src - dst order
	    data[2*i] = GCSC.getDst()[i];
	    data[2*i+1] = GCSC.getSrc()[i];
	}
	GCSC.del();
	
	std::cerr << "Write to file.\n";
	std::string cscfile = std::string( ofile ) + "-pull";
	ofstream file( cscfile, ios::out | ios::trunc | ios::binary );

	file.write( (const char *)&n, sizeof(n) );
	file.write( (const char *)&m, sizeof(m) );
	file.write( (const char *)data, sizeof(*data) * m * 2 );
	
	file.close();
	delete[] data;
	std::cerr << "Pull version created.\n";
    }

    return 0;
}
