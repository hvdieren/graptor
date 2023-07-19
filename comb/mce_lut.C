// -*- c++ -*-

#include "graptor/graptor.h"
#include "graptor/graph/simple/dense.h"

/*!======================================================================*
 * Generation of lookup-table for small MCE problems
 *=======================================================================*/

int main( int argc, char * argv[] ) {
    commandLine P( argc, argv, " help" );
    size_t n = P.getOptionLongValue( "-n", 1 ); // number of vertices

    if( n > 8 ) {
	std::cerr << "maximum number of vertices supported is 8 (" << n
		  << " requested)\n";
	return -1;
    }

    size_t num_bits = n * (n-1) / 2;
    size_t num_cases = size_t(1)<<num_bits;
    size_t * num_cliques = new size_t[num_cases];
    std::fill( num_cliques, num_cliques+num_cases, size_t(0) );

    size_t max_num_cliques = 0;
    for( size_t edges=0; edges < num_cases; ++edges ) {
	graptor::graph::DenseMatrix<32,VID,EID> D( n, edges );
	D.mce_bron_kerbosch( [&]( const bitset<32> & c, size_t sz ) {
	    num_cliques[edges]++;
	} );

	if( num_cliques[edges] > max_num_cliques )
	    max_num_cliques = num_cliques[edges];

	// std::cout << "edges: " << edges << " cliques: "
	// << num_cliques[edges] << "\n";
    }

    std::cout << "#define CLIQUE_LUT_" << n << "_CASES " << num_cases << "\n";
    std::cout << "#define CLIQUE_LUT_" << n << "_MAX_CLIQUES "
	      << max_num_cliques << "\n";
    std::cout << "uint8_t clique_lut_" << n << "_table[CLIQUE_LUT_"
	      << n << "_CASES][CLIQUE_LUT_" << n << "_MAX_CLIQUES] = {\n";

    for( size_t edges=0; edges < num_cases; ++edges ) {
	std::cout << "    {";
	size_t ncl = 0;
	graptor::graph::DenseMatrix<32,VID,EID> D( n, edges );
	D.mce_bron_kerbosch( [&]( const bitset<32> & c, size_t sz ) {
	    if( ncl > 0 )
		std::cout << ',';
	    std::cout << ' ' << std::hex << (uint32_t)c;
	    ++ncl;
	} );
	while( ncl++ < max_num_cliques )
	    std::cout << ", 0x0";
	std::cout << " }";
	if( edges+1 < num_cases )
	    std::cout << ',';
	std::cout << '\n';
    }
    std::cout << "};\n";

    return 0;
}
