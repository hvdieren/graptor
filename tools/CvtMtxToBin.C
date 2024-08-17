#include <string.h>

#include <cstdlib>
#include <unistd.h>
#include <cstdio>
#include <string>

#include "graptor/graptor.h"
#include "graptor/graph/cgraph.h"
#include "graptor/cmdline.h"

int main( int argc, char *argv[] ) {
    CommandLine P(
	argc, argv,
	"\t-i {file}\tinput file containing graph\n"
	"\t-o {file}\toutput file for converted graph\n"
	"\t-weights {file}\toutput file for weights of graph\n"
	"\t-vdown {shift}\tdownward shift to vertex IDs in output file\n"
	);
    bool symmetric = false;
    bool weighted = false;
    size_t vdown = P.get_long_option( "-vdown", 1 );
    const char * ofile = P.get_string_option( "-o" );
    const char * ifile = P.get_string_option( "-i" );
    const char * wfile = P.get_string_option( "-w" ); // file for weights

    std::cerr << "Reading Matrix Market file " << ifile << "...\n";

    const size_t SIZE = 1024*1024;
    char * buf = new char[SIZE];

    FILE *fp = fopen( ifile, "r" );
    if( fp == nullptr ) {
	std::cerr << "Error opening file '" << ifile << "'\n";
	return 1;
    }

    if( fgets( buf, SIZE, fp ) == nullptr ) {
	std::cerr << "Error reading header line\n";
	return 1;
    }
    if( strncmp( buf, "%%MatrixMarket ", 15 ) ) {
	std::cerr << "Error finding %%MatrixMarket\n";
	return 1;
    }
    char * p = buf + 15;
    char * token = strtok( p, " " );
    while( token ) {
	if( !strncmp( token, "real", 4 ) ) {
	    std::cerr << "option: edge weights are real\n";
	    weighted = true;
	} else if( !strncmp( token, "matrix", 6 ) ) {
	    // nothing to do
	} else if( !strncmp( token, "pattern", 6 ) ) {
	    std::cerr << "option: pattern matrix\n";
	    weighted = false;
	} else if( !strncmp( token, "symmetric", 9 ) ) {
	    symmetric = true;
	    std::cerr << "option: symmetric\n";
	} else if( !strncmp( token, "general", 7 ) ) {
	    symmetric = false;
	    std::cerr << "option: general\n";
	} else if( !strncmp( token, "coordinate", 10 ) ) {
	    std::cerr << "option: sparse matrix\n";
	} else {
	    std::cerr << "option: not supported: '" << token << "'\n";
	    return 1;
	}
	token = strtok( NULL, " " );
    }

    size_t nr = 0, nc = 0, m = 0;

    // Skip comments
    do {
	if( fgets( buf, SIZE, fp ) == nullptr ) {
	    std::cerr << "Error reading file\n";
	    return 1;
	}
    } while( buf[0] == '%' );

    // Read dimensions
    if( sscanf( buf, "%ld %ld %ld\n", &nr, &nc, &m ) != 3 ) {
	std::cerr << "error reading dimensions\n";
	std::cerr << buf << "\n";
	return 1;
    }

    // This is assuming no self-edges. If not, number will be less.
    if( symmetric )
	m *= 2;

    assert( nr != 0 && nc != 0 );

    if( nr != nc )
	std::cerr << "warning: rows " << nr << " differs from columns "
		  << nc << ", using largest\n";
    size_t n = std::max( nr, nc );

    std::cerr << "Number of vertices: " << n << "\n";
    std::cerr << "Number of edges: " << m << "\n";

    GraphCOO G( n, m, -1, weighted );

    VID * src = G.getSrc();
    VID * dst = G.getDst();
    float * wght = weighted ? G.getWeights() : nullptr;
    EID enxt = 0;

    // Skip over dimensions line
    if( fgets( buf, SIZE, fp ) == nullptr ) {
	std::cerr << "Error reading file\n";
	return 1;
    }

    // buf currently holds first line
    char * q, * qq, * qqq;
    do {
	p = buf;
	src[enxt] = strtoull( p, &q, 10 ) - vdown;
	assert( src[enxt] < n );
	assert( p != q );
	assert( *q == ' ' || *q == '\t' );
	++q;
	dst[enxt] = strtoull( q, &qq, 10 ) - vdown;
	assert( dst[enxt] < n );
	assert( q != qq );

	assert( src[enxt] < n );
	assert( dst[enxt] < n );

	if( weighted ) {
	    assert( *qq == ' ' || *qq == '\t' );
	    ++qq;
	    wght[enxt] = strtof( qq, &qqq );
	} else
	    qqq = qq;
	
	if( *qqq == '\r' )
	    ++qqq;
	assert( *qqq == '\n' );

	++enxt;

	// Need to mirror edges
	// Note: self-edges are duplicated in this process!
	if( symmetric ) {
	    if( src[enxt-1] != dst[enxt-1] ) {
		src[enxt] = dst[enxt-1];
		dst[enxt] = src[enxt-1];
		if( weighted )
		    wght[enxt] = wght[enxt-1];
		++enxt;
	    } else {
		std::cout << "Note: self-edge: " << src[enxt-1]
			  << ' ' << dst[enxt-1] << " not flipped\n";
	    }
	}

	if( enxt == m )
	    break;
    } while( (q = fgets( buf, SIZE, fp )) != nullptr );

    fclose( fp );
    delete[] buf;

    // Discrepancy is due to self-edges that aren't flipped in symmetric matrix
    if( m > enxt ) {
	m = enxt;
	G.shrink_num_edges( enxt );
    }

    // Convert COO to CSR
    std::cerr << "Convert to CSR\n";
    GraphCSx GG( n, m, -1, false, weighted ); // create one-sided only
    EID * idx = GG.getIndex();
    VID * edge = GG.getEdges();
    float * wght2 = weighted ? GG.getWeights()->get() : nullptr;
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

    for( EID e=0; e < m; ++e ) {
	EID p = pos[src[e]]++;
	edge[p] = dst[e];
	if( weighted )
	    wght2[p] = wght[e];
    }

    std::cerr << "Validate and sort\n";
    for( VID v=0; v < n; ++v ) {
	// Check we have all edges / all values initialized
	assert( pos[v] == idx[v+1] );
	if( weighted )
	    paired_sort( &edge[idx[v]], &edge[idx[v+1]], &wght2[idx[v]] );
	else
	    std::sort( &edge[idx[v]], &edge[idx[v+1]] );

	// Check all vertex IDs within range.
	assert( idx[v] == idx[v+1] || edge[idx[v+1]-1] < n );
    }

    // Write graph to file.
    std::cerr << "Write graph to file " << ofile << "\n";
    GG.writeToBinaryFile( ofile );
    if( wfile != nullptr )
	GG.writeWeightsToBinaryFile( wfile );
    else
	std::cerr << "Ignoring weights from file\n";

    return 0;
}
