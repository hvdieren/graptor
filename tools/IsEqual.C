#include <cstdlib>
#include <unistd.h>
#include <cstdio>
#include <string>

#include "ligra-numa.h"
#include "cgraph.h"

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    char* file0 = P.getArgument(0);
    char* file1 = P.getArgument(1);

    GraphCSx G0( file0, -1 );
    const EID * idx0 = G0.getIndex();
    const VID * edge0 = G0.getEdges();
    VID n0 = G0.numVertices();
    EID m0 = G0.numEdges();
    std::cerr << "Graph 0: " << file0 << " n=" << n0
	      << " m=" << m0 << std::endl;

    GraphCSx G1( file1, -1 );
    const EID * idx1 = G1.getIndex();
    const VID * edge1 = G1.getEdges();
    VID n1 = G1.numVertices();
    EID m1 = G1.numEdges();
    std::cerr << "Graph 1: " << file1 << " n=" << n1
	       << " m=" << m1 << std::endl;

    bool diff = false;
    if( n0 != n1 ) {
	std::cerr << "n differs\n";
	diff = true;
    }
    if( m0 != m1 ) {
	std::cerr << "m differs\n";
	diff = true;
    }
    
    if( !diff ) {
	cilk_for( VID s=0; s < n0; ++s ) {
	    EID i0 = idx0[s];
	    EID j0 = idx0[s+1];
	    EID i1 = idx1[s];
	    EID j1 = idx1[s+1];
	    if( i0 != i1 ) {
		std::cerr << "vertex " << s
			  << " idx0=" << i0 << " idx1=" << i1 << "\n";
		diff = true;
	    }
	    EID deg = min( j0-i0, j1-i1 );
	    if( (j0-i0) != (j1-i1) ) {
		std::cerr << "vertex " << s
			  << " deg0=" << (j0-i0) << " deg1=" << (j1-i1) << "\n";
		diff = true;
	    }
	    for( EID d=0; d < deg; ++d ) {
		VID d0 = edge0[i0+d];
		VID d1 = edge1[i1+d];
		if( d0 != d1 ) {
		    std::cerr << "vertex " << s
			      << " points to d0=" << d0 << " d1=" << d1 << "\n";
		    diff = true;
		}
		if( d > 0 ) {
		    if( edge0[i0+d-1] > edge0[i0+d] )
			std::cerr << "G0: vertex " << s
				  << " has non-sorted dests\n";
		    if( edge1[i1+d-1] > edge1[i1+d] )
			std::cerr << "G1: vertex " << s
				  << " has non-sorted dests\n";
		}
	    }
	}
    }

    if( diff )
	std::cerr << "Graphs differ\n";
    else
	std::cerr << "Graphs are equal\n";

    G1.del();
    G0.del();

    return diff ? 1 : 0;
}
