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
	"\t-m\t\tproduce MatrixMarket format\n"
	"\t-i {file}\tinput file containing graph\n"
	"\t-weights {file}\tinput file for weights of graph\n"
	"\t-o {file}\toutput file for converted graph\n"
	"\t-vup {shift}\tupward shift to vertex IDs in output file\n"
	);
    bool symmetric = P.get_bool_option( "-s" );
    bool mtx_mkt = P.get_bool_option( "-m" );            // Matrix market format

    const char * ifile = P.get_string_option( "-i" );
    const char * wfile = P.get_string_option("-weights"); // file with weights
    const char * ofile = P.get_string_option( "-o" );
    VID vup = P.get_long_option( "-vup", mtx_mkt ? 1 : 0 );

    std::cerr << "Reading graph " << ifile << "...\n";

    GraphCSx G( ifile, -1, symmetric, wfile );
    symmetric = G.isSymmetric();

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
	    buffer << "%%MatrixMarket matrix coordinate ";
	if( weights )
	    buffer << "real ";
	else
	    buffer << "pattern ";
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

	    if( mtx_mkt && symmetric ) {
		// Note: do not drop self-edges
		if( u <= v ) {
		    buffer << (u+vup) << ' ' << (v+vup);
		    if( weights )
			buffer << ' ' << weights[e];
		    buffer << '\n';
		}
	    } else {
		buffer << (u+vup) << ' ' << (v+vup);
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
