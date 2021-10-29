#include <cstdlib>
#include <unistd.h>
#include <cstdio>
#include <string>

#include "graptor/graptor.h"
#include "graptor/graph/cgraph.h"

bool hasEdge( GraphCSx & G, VID src, VID dst ) {
    const EID * idx = G.getIndex();
    const VID * edge = G.getEdges();

    for( EID i=idx[src]; i < idx[src+1]; ++i )
	if( edge[i] == dst )
	    return true;
    return false;
}

template<typename vertex>
bool hasEdge( graph<vertex> & G, VID src, VID dst ) {
    const vertex *V = G.V;

    for( VID i=0, d=V[dst].getInDegree(); i < d; ++i )
	if( src == V[dst].getInNeighbor(d) )
	    return true;
    return false;
}

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    char* iFile = P.getArgument(0);
    bool symmetric = P.getOptionValue("-s");
    bool binary = P.getOptionValue("-b");             //Galois binary format

    const char * ofile = P.getOptionValue( "-o" );

    GraphCSx G( iFile, -1 );

    std::cerr << "Read graph.\n";
    
    const EID * idx = G.getIndex();
    const VID * edge = G.getEdges();
    VID n = G.numVertices();
    EID m = G.numEdges();

    std::cerr << "Directed graph: n=" << n << " m=" << m << std::endl;

    mmap_ptr<VID> new_deg( n );

    for( VID s=0; s < n; ++s )
	new_deg[s] = idx[s+1] - idx[s];

    EID um = m;
    parallel_for( VID s=0; s < n; ++s ) {
	EID i = idx[s];
	EID j = idx[s+1];
	for( ; i < j; ++i ) {
	    if( !hasEdge( G, edge[i], s ) ) {
		assert( 0 <= edge[i] && edge[i] < n );
		__sync_fetch_and_add( &new_deg[edge[i]], 1 );
		__sync_fetch_and_add( &um, 1 );
	    } 
	}
	// if( (s % 1000000) == 0 )
	    // std::cerr << "progress: " << (float(s)/float(n)*100) << "%\n";
    }

    std::cerr << "Undirected graph: m=" << um << std::endl;

    GraphCSx UG( n, um, -1 );
    EID * uidx = UG.getIndex();
    VID * uedge = UG.getEdges();

    uidx[0] = 0;
    for( VID s=0; s < n; ++s )
	uidx[s+1] = uidx[s] + new_deg[s];
    assert( uidx[n] == um );

    new_deg.del();
    mmap_ptr<EID> uidx2( n );
    
    parallel_for( VID s=0; s < n; ++s )
	uidx2[s] = uidx[s];

    parallel_for( VID s=0; s < n; ++s ) {
	EID i = idx[s];
	EID j = idx[s+1];
	for( ; i < j; ++i ) {
	    EID p = __sync_fetch_and_add( &uidx2[s], 1 );
	    uedge[p] = edge[i];
	    if( !hasEdge( G, edge[i], s ) ) {
		EID p = __sync_fetch_and_add( &uidx2[edge[i]], 1 );
		uedge[p] = s;
	    }
	}
    }

    std::cerr << "Constructed undirected graph\n";

    parallel_for( VID v=0; v < n; ++v ) {
	// Check we have all edges / all values initialized
	assert( uidx2[v] == uidx[v+1] );
	std::sort( &uedge[uidx[v]], &uedge[uidx[v+1]] );

	// Check all vertex IDs within range.
        assert( uidx[v] == uidx[v+1] || uedge[uidx[v+1]-1] < n );
    }

    std::cerr << "Graph sorted\n";

    UG.writeToBinaryFile( ofile );

    return 0;
}
