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
	"\t-s\t\tinput graph is symmetric (only required for specific formats)\n"
	"\t-i {file}\tinput file containing graph\n"
	"\t-o {file}\toutput file for converted graph\n"
	"\t--rm-self\tremove self-edges\n"
	"\t--rm-isolated\tremove isolated vertices\n"
	"\t--v4\t\tcreate Graptor v4 output file\n"
	);

    bool symmetric = P.get_bool_option( "-s" );
    bool v4 = P.get_bool_option( "--v4" );
    bool rm_self = P.get_bool_option( "--rm-self" );
    bool rm_isolated = P.get_bool_option( "--rm-isolated" );

    const char * ifile = P.get_string_option( "-i" );
    const char * ofile = P.get_string_option( "-o" );

    GraphCSx G( ifile, -1, symmetric, nullptr );
    symmetric = G.isSymmetric();

    std::cerr << "Vertices: " << G.numVertices()
	      << "\nEdges: " << G.numEdges() << "\n";

    VID n = G.numVertices();
    EID m = G.numEdges();
    EID * index = G.getIndex();
    VID * edges = G.getEdges();

    std::atomic<EID> m_self( EID(0) );
    std::atomic<VID> n_isolated( 0 );

    if( rm_self )
	std::cout << "Removing self-edges\n";
    if( rm_isolated )
	std::cout << "Removing isolated vertices\n";

    mm::buffer<VID> o2n( n, numa_allocation_interleaved(),
			 "mapping old to new vertex IDs" );

    VID un = 0;
    for( VID u=0; u < n; ++u ) {
	EID es = index[u];
	EID ee = index[u+1];

	if( es == ee ) {
	    ++n_isolated;
	    if( rm_isolated )
		o2n[u] = ~(VID)0;
	    else
		o2n[u] = un++;
	} else {
	    o2n[u] = un++;
	    for( EID e=es; e < ee; ++e ) {
		VID v = edges[e];
		if( v >= n ) {
		    std::cerr << "Fatal error: vertex " << v << " in edge " << e
			      << " is out of bounds\n";
		    exit( 1 );
		}
	    
		if( u == v )
		    ++m_self;
	    }
	}
    }

    EID um = rm_self ? m - m_self : m;

    std::cout << "Self-edges: " << m_self
	      << "\nIsolated vertices: " << n_isolated << "\n";
    std::cout << "New vertices: " << un
	      << "\nNew edges: " << um << "\n";

    if( rm_isolated && un + n_isolated != n ) {
	std::cout << "Vertex check error: remaining " << un
		  << " and isolated " << n_isolated << " do not add up to "
		  << n << "\n";
	exit( 1 );
    } else if( !rm_isolated && un !=n ) {
	std::cout << "Vertex check error: remaining " << un
		  << " differs from " << n << "\n";
	exit( 1 );
    }

    GraphCSx UG( un, um, -1, symmetric, false );
    EID * uindex = UG.getIndex();
    VID * uedges = UG.getEdges();

    // Create graph
    EID ue = 0;
    for( VID u=0; u < n; ++u ) {
	VID uu = o2n[u];

	if( uu != ~(VID)0 ) {
	    uindex[uu] = ue;

	    EID es = index[u];
	    EID ee = index[u+1];
	    for( EID e=es; e < ee; ++e ) {
		VID v = edges[e];
		VID uv = o2n[v];
		assert( uv == ~(VID)0 || uv < un );
	    
		if( u == v && rm_self ) {
		    // Remove edge
		} else if( uv != ~(VID)0 )
		    uedges[ue++] = uv;
	    }
	}
    }

    if( ue != um ) {
	std::cerr << "Fatal error: expected " << um << " edges; placed "
		  << ue << " edges\n";
	exit( 1 );
    }

    uindex[un] = um;

    std::cout << "\nWriting output file '" << ofile
	      << "' format " << ( v4 ? "v4" : "b2" ) << "...\n";
    if( v4 )
	UG.writeToGraptorV4File( ofile );
    else
	UG.writeToBinaryFile( ofile );

    return 0;
}
