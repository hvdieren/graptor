// -*- c++ -*-

#include "graptor/graptor.h"

template<typename tr, typename Fn>
void ubench( uint64_t repeat, uint64_t cluster,
	     const char * const desc, Fn && fn ) { 
    using type = typename tr::type;
    using member_type = typename tr::member_type;

    type vec = tr::set1inc0();

    timer tm;
    tm.start();
    for( uint64_t r=0; r < repeat; ++r ) {
	for( uint64_t c=0; c < cluster; ++c ) {
	    uint64_t l = c % tr::vlen;
	    member_type m = fn( vec, l );
	    if( m != l ) {
		std::cerr << "Error: l=" << l << " m=" << m << "\n";
		assert( m == l );
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

    using member_type = uint64_t;
    using tr = vector_type_traits_vl<member_type,2>;
    using type = typename tr::type;

    std::cerr << "micro-benchmarking lane"
	      << "\n    repeat=" << repeat
	      << "\n    cluster=" << cluster
	      << "\n    sizeof(member_type)=" << sizeof(member_type)
	      << "\n    vlen=" << tr::vlen
	      << "\n";

    ubench<tr>( repeat, cluster, "switch", [=]( type vec, uint64_t lane ) {
	return tr::lane_switch( vec, lane );
    } );

    ubench<tr>( repeat, cluster, "memory", [=]( type vec, uint64_t lane ) {
	return tr::lane_memory( vec, lane );
    } );

    ubench<tr>( repeat, cluster, "permute", [=]( type vec, uint64_t lane ) {
	return tr::lane_permute( vec, lane );
    } );
    
    return 0;
}
