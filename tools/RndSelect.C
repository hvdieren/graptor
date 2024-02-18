#include <cstdlib>
#include <unistd.h>
#include <cstdio>
#include <string>
#include <random>

#include "graptor/graptor.h"
#include "graptor/graph/cgraph.h"

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    bool symmetric = P.getOptionValue( "-s" );

    const char * ifile = P.getOptionValue( "-i" );
    const VID nsamples = P.getOptionLongValue( "--nsamples", 10 );

    std::cerr << "Reading graph " << ifile << "...\n";

    GraphCSx G( ifile, -1, symmetric );

    VID n = G.numVertices();
    EID m = G.numEdges();

    std::cerr << "  |V|=" << n << "\n";
    std::cerr << "  |E|=" << m << "\n";
    std::cerr << "  #samples=" << nsamples << "\n";

    mmap_ptr<bool> selected( n );
    std::mt19937_64 generator;
    std::uniform_int_distribution<std::mt19937_64::result_type> dist( 0, n-1 );

    parallel_loop( VID(0), n, [&]( VID v ) { selected[v] = false; } );

    for( VID s=0; s < nsamples; ++s ) {
	while( true ) {
	    VID v = dist( generator );
	    VID d = G.getDegree( v );
	    if( !selected[v] && d != 0 ) {
		selected[v] = true;
		std::cout << v << ' ' << d << '\n';
		break;
	    }
	}
    }

    return 0;
}
