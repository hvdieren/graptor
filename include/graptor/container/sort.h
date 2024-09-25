// -*- C++ -*-
#ifndef GRAPTOR_CONTAINER_SORT_H
#define GRAPTOR_CONTAINER_SORT_H

#include <iterator>

namespace graptor {

//! \brief SAPCo sort
//
// An optimised counting sort that uses a mixture of per-thread counters for
// highly contented elements (e.g., low vertex degrees in a skewed-degree graph
// when sorting vertices by degree) and shared counters for elements with
// low contention.
// This sorting algorithm does not provide a stable sort.
//
// M. Koohi Esfahani, P. Kilpatrick and H. Vandierendonck, "SAPCo Sort:
// Optimizing Degree-Ordering for Power-Law Graphs,"
// 2022 IEEE International Symposium on Performance Analysis of Systems
// and Software (ISPASS), Singapore, 2022, pp. 138-140,
// doi: 10.1109/ISPASS55109.2022.00015.
// 
// \param[in] from start of range to sort
// \param[in] to end of range to sort
// \param[in] key sort a,b in [from,to) with a before b iff key[a] <= key[b]
// \param[in] scratch scratch space range up to max_val+1+npriv*parts
// \param[out] out output range
// \param[in] max_val the largest value in the range [from,to)
// \param[in] npriv number of private counters
// \param[in] nparts number of partitions/threads
// \param[in] reverse reverse sort order
template<typename T, typename Iterator>
void sapco_sort( Iterator && from, Iterator && to, const T * key, T * out,
		 T * scratch,
		 const T max_val, const T npriv, const T nparts,
		 bool reverse ) {
    // Clear scratch space
    std::fill( scratch, scratch+max_val+1+npriv*nparts, T(0) );

    const T n = std::distance( from, to );
    const T chunk = ( n + nparts - 1 ) / nparts;

    // Create histogram
    parallel_loop( T(0), nparts, T(1), [=]( T p ) {
	const Iterator b = std::next( from, std::min( p * chunk, n ) );
	const Iterator e = std::next( from, std::min( (p+1) * chunk, n ) );
	for( Iterator s = b; s != e; ++s ) {
	    T v = *s;
	    T k = key[v];
	    if( k < npriv )
		++scratch[p*npriv+k];
	    else
		__sync_fetch_and_add( &scratch[npriv*nparts+k], 1 );
	}
    } );

    // Reduce private counters
    parallel_loop( T(0), npriv, T(16), [=]( T k ) {
	T sum = 0;
	for( T p=0; p < nparts; ++p ) {
	    T tmp = scratch[p*npriv+k];
	    scratch[p*npriv+k] = sum;
	    sum += tmp;
	}
	scratch[npriv*nparts+k] = sum;
    } );

    // Prefix sum
    // Note: require int variables as the code checks >= 0 which is futile
    //       with unsigned int
    T sum = sequence::scan( scratch+npriv*nparts,
			      (int)0, (int)max_val+1, addF<T>(),
			      sequence::getA<T,int>( scratch+npriv*nparts ),
			      (T)0, false, reverse );

    // Place elements
    parallel_loop( T(0), nparts, T(1), [=]( T p ) {
	const Iterator b = std::next( from, std::min( p * chunk, n ) );
	const Iterator e = std::next( from, std::min( (p+1) * chunk, n ) );
	for( Iterator s = b; s != e; ++s ) {
	    T v = *s;
	    T k = key[v];
	    if( k < npriv )
		out[scratch[npriv*nparts+k]+scratch[p*npriv+k]++] = v;
	    else
		out[__sync_fetch_and_add( &scratch[npriv*nparts+k], 1 )] = v;
	}
    } );
}

//! \brief stable counting sort
//
// requires one counter per thread and per possible value in the range
// [0,max_val]. Hence, the scratch space should of size at least
// bit_ceil(max_val+1)*nparts elements
//
// This approach is efficient if max_val is not too high, e.g., max_val
// is degeneracy of graph.
template<typename T, typename Iterator>
void
stable_counting_sort(
    Iterator && from,
    Iterator && to,
    const T * const key,
    T * scratch,
    T * histo, // array of length max_val+2
    T * out, // array of length to - from
    const T max_val,
    const T nparts,
    bool reverse = false ) {

    // Calculate chunk sizes
    const T L = std::bit_ceil( max_val+1 );
    T * pt = scratch; // at least L * nparts
    const T n = std::distance( from, to );
    const T C = ( n + nparts - 1 ) / nparts;

    // Clear histogram
    std::fill( &histo[0], &histo[max_val+1], T(0) );

    // Construct histogram
    parallel_loop( T(0), nparts, [&]( T t ) {
	// Clear per-thread/partition counters
	T * h = &pt[t*L];
	std::fill( h, h+L, T(0) );

	const Iterator b = std::next( from, std::min( n, t * C ) );
	const Iterator e = std::next( from, std::min( n, (t+1) * C ) );
	for( Iterator s=b; s != e; ++s ) {
	    T v = *s;
	    assert( v < n );
	    T c = key[v];
	    h[c]++;
	}
    } );

    // 16 counters in a cache line, so 16-entry chunks to avoid false sharing.
    // This assumes that the whole array is 16-entry aligned.
    parallel_loop( T(0), max_val+1, T(16), [&]( T c ) {
	T s = 0;
	for( T t=0; t < nparts; ++t ) {
	    T val = pt[t*L+c];
	    pt[t*L+c] = s;
	    s += val;
	}
	histo[c] = s;
    } );

    // Prefix sum
    // Note: require int variables as the code checks >= 0 which is futile
    //       with unsigned int
    T sum = sequence::scan( histo, (int)0, (int)max_val+1, addF<T>(),
			      sequence::getA<T,int>( histo ),
			      (T)0, false, reverse );

    // Place in order
    parallel_loop( T(0), nparts, [&]( T t ) {
	T * h = &pt[t*L]; // per-thread insertion points
	const Iterator b = std::next( from, std::min( n, t * C ) );
	const Iterator e = std::next( from, std::min( n, (t+1) * C ) );
	for( Iterator s=b; s != e; ++s ) {
	    T v = *s;
	    T c = key[v];
	    T pos = histo[c] + h[c]++;
	    out[pos] = v;
	}
    } );

    // Restore histo
    if( reverse ) {
	histo[0] = n;
	histo[max_val+1] = 0;
    } else {
	histo[0] = 0;
	histo[max_val+1] = n;
    }
}

} // namespace graph

#endif // GRAPTOR_CONTAINER_SORT_H

