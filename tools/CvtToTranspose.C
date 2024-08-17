#include <cstdlib>
#include <unistd.h>
#include <cstdio>
#include <string>

#include "graptor/graptor.h"
#include "graptor/cmdline.h"

int main( int argc, char *argv[] ) {
    CommandLine P(
	argc, argv,
	"\t-i {file}\tinput file containing graph\n"
	"\t-o {file}\toutput file for transposed graph\n"
	);

    const char * ifile = P.get_string_option( "-i" );
    const char * ofile = P.get_string_option( "-o" );

    GraphCSx G( ifile, -1, false, nullptr );

    const EID * idx = G.getIndex();
    const VID * edge = G.getEdges();
    VID n = G.numVertices();
    EID m = G.numEdges();

    std::cerr << "Read directed graph: " << ifile
	      << " n=" << n << " m=" << m << std::endl;

    mmap_ptr<VID> new_deg( n );

    parallel_for( VID s=0; s < n; ++s )
	new_deg[s] = 0;

    parallel_for( EID e=0; e < m; ++e ) {
	assert( 0 <= edge[e] && edge[e] < n );
	__sync_fetch_and_add( &new_deg[edge[e]], 1 );
    }

    std::cerr << "Counted reverse edges\n";

    GraphCSx TG( n, m, -1 );
    EID * tidx = TG.getIndex();
    VID * tedge = TG.getEdges();

    tidx[0] = 0;
    for( VID s=0; s < n; ++s )
	tidx[s+1] = tidx[s] + EID(new_deg[s]);
    assert( tidx[n] == m );

    new_deg.del();
    mmap_ptr<EID> tmp( n );
    
    parallel_for( VID s=0; s < n; ++s )
	tmp[s] = tidx[s];
    parallel_for( EID e=0; e < m; ++e )
	tedge[e] = ~VID(0);

    parallel_for( VID s=0; s < n; ++s ) {
	EID i = idx[s];
	EID j = idx[s+1];
	for( ; i < j; ++i ) {
	    EID pos = __sync_fetch_and_add( &tmp[edge[i]], 1 );
	    assert( 0 <= pos && pos < m );
	    assert( tidx[edge[i]] <= pos && pos < tidx[edge[i]+1] );
	    tedge[pos] = s;
	}
    }

    std::cerr << "Constructed transposed graph\n";

    parallel_for( EID e=0; e < m; ++e ) {
	assert( 0 <= tedge[e] && tedge[e] < n );
    }

    std::cerr << "Checked destinations\n";

    parallel_for( VID v=0; v < n; ++v ) {
	// Check we have all edges / all values initialized
	assert( tmp[v] == tidx[v+1] );
	std::sort( &tedge[tidx[v]], &tedge[tidx[v+1]] );

	// Check all vertex IDs within range.
        assert( tidx[v] == tidx[v+1] || tedge[tidx[v+1]-1] < n );
    }

    parallel_for( EID e=0; e < m; ++e ) {
	assert( 0 <= tedge[e] && tedge[e] < n );
    }

    std::cerr << "Graph sorted\n";

    TG.writeToBinaryFile( ofile );

    return 0;
}
