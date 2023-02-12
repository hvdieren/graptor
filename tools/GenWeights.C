#include <cstdlib>
#include <unistd.h>
#include <cstdio>
#include <string>
#include <random>

#include "graptor/graptor.h"
#include "graptor/graph/GraphCSx.h"

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );

    const char * ifile = P.getOptionValue( "-i" );
    const char * ofile = P.getOptionValue( "-o" );
    const bool symmetric = P.getOptionValue("-s");

    timer tm;
    tm.start();

    std::cerr.precision( 5 );
    std::cerr << std::fixed;
    std::cerr << tm.next() << " Reading graph: " << ifile << "\n";

    GraphCSx G( ifile, -1, symmetric );
    VID n = G.numVertices();
    EID m = G.numEdges();
    const EID * index = G.getIndex();
    const VID * edges = G.getEdges();

    std::cerr << "Number of edges: " << m << "\n";

    std::cerr << tm.next() << " Creating weights file: " << ofile << "\n";
    
    int fd;

    if( (fd = open( ofile, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600 )) < 0 ) {
	std::cerr << "Cannot open+create file '" << ofile << "': "
		  << strerror( errno ) << "\n";
	exit( 1 );
    }

    using weight_t = float;
    size_t wsize = sizeof(weight_t) * m;
    off_t len = lseek( fd, wsize-1, SEEK_SET );
    if( len == off_t(-1) ) {
	std::cerr << "Cannot lseek to desired end of file '" << ofile << "': "
		  << strerror( errno ) << "\n";
	exit( 1 );
    }
    write( fd, "", 1 ); // Create area
    
    std::cerr << tm.next() << " Map file to memory\n";
    const char * data = (const char *)mmap( 0, wsize, PROT_READ | PROT_WRITE,
					    MAP_SHARED, fd, 0 );
    if( data == (const char *)-1 ) {
	std::cerr << "Cannot mmap file '" << ofile << "' read-only: "
		  << strerror( errno ) << "\n";
	exit( 1 );
    }

    std::cerr << tm.next() << " Generating weights\n";

    weight_t * w = (weight_t *)data;

#if 0
    std::mt19937_64 generator;
    std::uniform_int_distribution<std::mt19937_64::result_type>
	dist( 1, 100000 );
	// dist( 1, 1<<5 );

    for( EID e=0; e < m; ++e ) {
	w[e] = (weight_t) dist( generator );
    }

    std::cerr << tm.next() << " Building inverse edge map\n";

    mm::buffer<EID> inverse_map = G.buildInverseEdgeMap();
    const EID * invert = inverse_map.get();

    std::cerr << tm.next() << " Matching weights on inverse edges\n";

    parallel_for( EID e=0; e < m; ++e ) {
	assert( !symmetric || invert[e] != ~(EID)0 );
	assert( !symmetric || invert[invert[e]] == e );
	if( invert[e] > e )
	    w[invert[e]] = w[e];
    }

    std::cerr << tm.next() << " Cleaning up\n";

    inverse_map.del();

#else
    parallel_for( VID v=0; v < n; ++v ) {
	EID es = index[v];
	EID ee = index[v+1];
	for( EID e=es; e < ee; ++e ) {
	    VID u = edges[e];
	    VID cR = 127;
	    float cO = 64;
	    float cW = 64 * 16;
	    float val = ( u + v ) & cR;
	    // Generates integer weights (compatibility ligra)
	    // For float weights, divide by cW
	    w[e] = ( cW / 8.0f ) + ( val - cO );
	}
    }
#endif

    std::cerr << tm.next() << " Syncing data to file\n";
    msync( (void *)data, wsize, MS_SYNC );
    munmap( (void *)data, wsize );
    close( fd );

    std::cerr << tm.next() << " Done\n";

    return 0;
}
