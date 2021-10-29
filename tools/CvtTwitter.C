#include <cstdlib>
#include <unistd.h>
#include <cstdio>
#include <string>
#include <algorithm>

#include "graptor/graptor.h"
#include "graptor/graph/cgraph.h"

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );

    size_t n = P.getOptionLongValue( "-n", 1 );
    size_t m = P.getOptionLongValue( "-m", 1 );
    size_t vdown = P.getOptionLongValue( "-vdown", 0 );
    const char * ifile = P.getOptionValue( "-i" );
    const char * ofile = P.getOptionValue( "-o" );

    GraphCOO G( n, m, -1 );

    VID * src = G.getSrc();
    VID * dst = G.getDst();
    EID enxt = 0;

    std::string infile = ifile;

    printf( "Reading file '%s'\n", infile.c_str() );

    // Parent
    const size_t SIZE = 1024*1024;
    char * buf = new char[SIZE];

    FILE *fp = fopen( ifile, "r" );

    char * p = buf, * q;
    VID vmax = 0;
    while( (q = fgets( buf, SIZE, fp )) != nullptr ) {
	if( p[0] != '#' ) {
	    src[enxt] = strtoull( p, &q, 10 ) - vdown;
	    // assert( src[enxt] < n );
	    if( src[enxt] > vmax )
		vmax = src[enxt];
	    assert( p != q );
	    assert( *q == ' ' || *q == '\t' );
	    ++q;
	    dst[enxt] = strtoull( q, &q, 10 ) - vdown;
	    // assert( dst[enxt] < n );
	    if( dst[enxt] > vmax )
		vmax = dst[enxt];
	    if( *q == '\r' )
		++q;
	    assert( *q == '\n' );

	    ++enxt;
	    if( enxt == m )
		break;
	}
    }

    ++vmax; // high end of range is one larger than highest observed value
    if( vmax > n ) {
	std::cerr << "Warning: vmax=" << vmax << " but n=" << n << "\n";
	n = vmax;
    }

    delete[] buf;
    fclose( fp );

    std::cerr << "n=" << n << "\n";
    std::cerr << "enxt=" << enxt << "\n";
    std::cerr << "m=" << m << "\n";

    assert( enxt == m );

    // Convert COO to CSR
    std::cerr << "Convert to CSR\n";
    GraphCSx GG( n, m, -1 );
    EID * idx = GG.getIndex();
    VID * edge = GG.getEdges();
    mmap_ptr<EID> pos( n+1 );

    for( VID v=0; v < n+1; ++v )
	idx[v] = 0;

    for( EID e=0; e < m; ++e )
	idx[src[e]]++;

    EID val = 0;
    for( VID v=0; v < n; ++v ) {
	EID x = val;
	val += idx[v];
	idx[v] = x;
	pos[v] = idx[v];
    }
    idx[n] = val;
    assert( idx[0] == 0 );
    assert( idx[n] == m );

    for( EID e=0; e < m; ++e )
	edge[pos[src[e]]++] = dst[e];

    std::cerr << "Validate and sort\n";
    for( VID v=0; v < n; ++v ) {
	// Check we have all edges / all values initialized
	assert( pos[v] == idx[v+1] );
	std::sort( &edge[idx[v]], &edge[idx[v+1]] );

	// Check all vertex IDs within range.
	assert( idx[v] == idx[v+1] || edge[idx[v+1]-1] < n );
    }

    // Write graph to file.
    std::cerr << "Write graph to file " << ofile << "\n";
    GG.writeToBinaryFile( ofile );
}
