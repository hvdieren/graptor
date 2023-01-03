#include <cstdlib>
#include <unistd.h>
#include <cstdio>
#include <string>

#include "graptor/graptor.h"
#include "graptor/graph/cgraph.h"

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    bool symmetric = P.getOptionValue( "-s" );
    bool mtx_mkt = P.getOptionValue( "-m" );            // Matrix market format

    const char * ifile = P.getOptionValue( "-i" );
    const char * wfile = P.getOptionValue("-weights"); // file with weights
    const char * ofile = P.getOptionValue( "-o" );

    std::cerr << "Reading graph " << ifile << "...\n";

    GraphCSx G( ifile, -1, symmetric, wfile );

    std::cerr << "Vertices: " << G.numVertices()
	      << "\nEdges: " << G.numEdges()
    	      << "\nCreating output file " << ofile << "...\n";

    std::string wel = std::string( ofile );
    ofstream file( wel, ios::out | ios::trunc );

    if( !file ) {
	std::cerr << "Could not create '" << ofile << "': "
		  << strerror( errno ) << "\n";
	return 1;
    }

    VID n = G.numVertices();
    EID m = G.numEdges();
    EID * index = G.getIndex();
    VID * edges = G.getEdges();
    float * weights = G.getWeights() ? G.getWeights()->get() : nullptr;

    constexpr size_t THRESHOLD = size_t(1) << 32; // 4 GiB
    std::stringstream buffer;

    if( mtx_mkt ) {
	buffer << "%%MatrixMarket matrix coordinate real ";
	if( !symmetric )
	    buffer << "un";
	buffer << "symmetric\n";
	buffer << n << ' ' << n << ' ' << ( symmetric ? m/2 : m ) << "\n";
    }

    for( VID u=0; u < n; ++u ) {
	EID es = index[u];
	EID ee = index[u+1];
	for( EID e=es; e < ee; ++e ) {
	    VID v = edges[e];

	    if( (size_t)buffer.tellp() >= THRESHOLD ) {
		file << buffer.rdbuf();
		std::stringstream().swap(buffer);
	    }

	    if( mtx_mkt ) {
		if( u > v ) {
		    buffer << u << ' ' << v
			   << ' ' << ( weights ? weights[e] : -1.0f ) << '\n';
		}
	    } else {
		buffer << u << ' ' << v;
		if( weights )
		    buffer << ' ' << weights[e];
		buffer << '\n';
	    }
	}
    }

    file << buffer.rdbuf();
    file.close();
    std::cerr << "Wrote edge list.\n";

    return 0;
}
