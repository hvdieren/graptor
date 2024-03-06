#include <cstdlib>
#include <unistd.h>
#include <cstdio>
#include <string>
#include <cassert>
#include <ostream>

#include "ligra-numa.h"
#include "vecop.C"
#include "IO.h"

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );

    size_t n = P.getOptionLongValue( "-n", 1 );
    size_t m = P.getOptionLongValue( "-m", 1 );
    bool v2 = P.getOption( "-v2", false );
    const char * ofile = P.getOptionValue( "-o" );
    const char * ifile = P.getOptionValue( "-i" );
    bool binary = P.getOptionValue("-b");             //Galois binary format

    std::cerr << "Reading Ligra graph " << ifile << "\n";
    wholeGraph<asymmetricVertex> WG =
	readGraph<asymmetricVertex>((char *)ifile,false,binary);

    std::cerr << "Convert to GraphCSx\n";
    GraphCSx G( WG, -1 );

    // Write graph to file.
    std::cerr << "Write graph to file " << ofile << "\n";
    if( v2 )
	G.writeToBinaryFile( ofile );
    else
	G.writeToGraptorV4File( ofile );
}
