// -*- c++ -*-
#ifndef GRAPTOR_UNIQUE_H
#define GRAPTOR_UNIQUE_H

#include <iostream>

// To construct a summary histogram of component sizes
template<typename T, typename U, typename F>
struct UniqueCount
{
    const T* values;
    U* count;
    T n;
    U* others;
    F* frontier;
    UniqueCount( const T* _values, U* _count, T _n, U* _others, F* _frontier ) :
        values(_values), count(_count), n(_n), others(_others),
	frontier(_frontier) {}
    inline bool operator () ( VID i ) {
	/* if( frontier[i] ) */ { // implied that Frontier is true
	    if( values[i] < 0 || values[i] >= n )
		__sync_fetch_and_add( others, 1 );
	    else
		return __sync_fetch_and_add( &count[values[i]], 1 ) == 0;
	}
	return false;
    }
};

template<typename U>
struct AppearingValue
{
    U* count;
    AppearingValue( U* _count ) : count(_count) {}
    inline bool operator () ( VID i ) {
	return count[i] != 0;
    }
};

// This code makes the assumption that all values are within the range
// of integers 0..n-1 where n is the number of vertices in the graph
// U: type of counters in histogram
// T: type of values counted up
template<typename U, typename T, class GraphType>
std::enable_if_t<std::is_integral_v<T>,VID>
count_unique( GraphType &GA, const T * values, std::ostream &os ) {
    const partitioner &part = GA.get_partitioner();
    VID n = GA.numVertices();
    EID m = GA.numEdges();

    // Collect statistics
    // 1. initialise data
    mmap_ptr<U> count;
    count.allocate( numa_allocation_partitioned( part ) );
    expr::array_ro<U,VID,0> count_a( count );

    // map_vertex does not work - padded vertices skipped
    parallel_loop( (VID)0, n, [=]( VID v ) mutable { count_a[v] = VID(0); } );

    // 2. count number of vertices per partition
    //    set only vertex donating label as active
    //    using vertexFilter, so requires a precise frontier taking account of
    //    vertices that are padding
    using traits = gtraits<GraphType>;
    frontier Frontier = traits::template createValidFrontier<sizeof(VID)>( GA );

    // TODO: immediately use packIndex as toSparse() will do the same
    U others = 0;
    frontier unique
	= vertexFilter( GA, Frontier,
			UniqueCount<T,U,logical<sizeof(VID)>>(
			    values, count, n, &others,
			    Frontier.template getDenseL<sizeof(VID)>() ) );

    // 3. get number of unique
    _seq<VID> packed = sequence::pack( (VID*)nullptr, count.get(), (VID)0,
				       n, identityF<VID>() );
    assert( unique.nActiveVertices() == packed.n &&
	    "nactv value mismatch with array" );

    VID nunique = packed.n;

    // 4. get extreme statistics and check correctness
    VID * s = packed.A;
    T largest = s[0], smallest = s[0];
    VID most_freq = 0, least_freq = 0;
    VID sum = others;
    for( VID i=0; i < nunique; ++i ) {
	VID j = s[i];
	sum += count[j];
	if( smallest > j )
	    smallest = j;
	if( largest < j )
	    largest = j;
	if( count[j] > count[most_freq] )
	    most_freq = j;
	if( count[j] < count[least_freq] )
	    least_freq = j;
    }

    os << "Unique values: " << nunique << "\n"
       << "Check: All vertices accounted for? "
       << ( sum == GA.get_partitioner().get_num_vertices() ? "PASS" : "FAIL" )
       << " (s=" << sum << '/' << n << '/'
       << GA.get_partitioner().get_num_vertices() << ")\n"
       << "Largest value: " << (VID)largest << "\n"
       << "Smallest value: " << (VID)smallest << "\n"
       << "Least frequent value: " << (VID)least_freq
       << " count: " << count[least_freq] << "\n"
       << "Most frequent value: " << most_freq
       << " count: " << count[most_freq] << "\n";

    // Check the pigeon-hole principle
    if( largest - smallest + 1 < nunique )
	os << "FAIL: largest-smallest span (" << (VID)(largest - smallest)
	   << " less then number of unique values (" << nunique << ")\n";

    // Check no other values
    if( others > 0 )
	os << "No values out of range? " << ( others == 0 ? "PASS" : "FAIL" )
	   << " (others=" << others << ")\n";

    Frontier.del();
    unique.del();
    count.del();

    delete[] packed.A;

    return nunique;
}

#endif // GRAPTOR_UNIQUE_H
