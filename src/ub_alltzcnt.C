// -*- c++ -*-

#include "graptor/graptor.h"

template<typename tr, typename Fn>
void ubench( uint64_t repeat, uint64_t cluster,
	     size_t pos, const char * const desc, Fn && fn ) { 
    using type = typename tr::type;
    using member_type = typename tr::member_type;

    type vec = tr::setglobaloneval( pos );

    timer tm;
    tm.start();
    for( uint64_t r=0; r < repeat; ++r ) {
	for( uint64_t c=0; c < cluster; ++c ) {
	    size_t m = fn( vec );
	    if( m != pos ) {
		std::cerr << "Error: pos=" << pos << " m=" << m
			  << " r=" << r << " c=" << c
			  << "\n";
		for( unsigned short l=0; l < tr::vlen; ++l )
		    std::cerr << "  " << std::hex << tr::lane( vec, l )
			      << std::dec << "\n";
		assert( m == pos );
	    }
	}
	tm.next();
    }
    tm.stop();

    double elapsed = tm.total();
    std::cerr << desc << ": total time: " << elapsed
	      << "\n" << desc << ": cluster time: " << elapsed/float(repeat)
	      << "\n" << desc << ": average time: "
	      << elapsed/float(repeat)/float(cluster)
	      << "\n";
}

int main( int argc, char * argv[] ) {
    commandLine P( argc, argv, " help" );
    uint64_t repeat = P.getOptionLongValue( "-r", 100000 );
    uint64_t cluster = P.getOptionLongValue( "-c", 10000 );

    constexpr unsigned short VL = 2;
    using tr = vector_type_traits_vl<uint64_t,VL>;
    using type = typename tr::type;
    constexpr size_t pos = (tr::size*8) / 2 + (tr::size*8) / 4 + 3;

    std::cerr << "micro-benchmarking lane"
	      << "\n    repeat=" << repeat
	      << "\n    cluster=" << cluster
	      << "\n    vlen=" << tr::vlen
	      << "\n    pos=" << pos
	      << "\n";

    ubench<tr>( repeat, cluster, pos, "alltzcnt", [=]( type vec ) {
	return target::alltzcnt<uint64_t,uint64_t,VL>::compute( vec );
    } );

    return 0;
}
