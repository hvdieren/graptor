#include <cstdlib>
#include <unistd.h>
#include <cstdio>
#include <string>

#include "graptor/graptor.h"
#include "graptor/graph/cgraph.h"

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    const bool symmetric = P.getOptionValue("-s");
    const char * ofile = P.getOptionValue( "-o" );
    const char * ifile = P.getOptionValue( "-i" );
    const char * weights = P.getOptionValue("-weights"); // file with weights

    std::cerr << "Reading graph " << ifile << "...\n";

    GraphCSx G( ifile, -1, symmetric, weights );

    std::cerr << "Writing graph " << ofile << "...\n";
    
    G.writeToTextFile( ofile );

    return 0;
}
