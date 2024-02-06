// -*- c++ -*-
#ifndef GRAPTOR_PRIMITIVES_H
#define GRAPTOR_PRIMITIVES_H

#include <algorithm>
#include <string.h>

#include "graptor/backend/backend.h"
#include "graptor/partitioner.h"
#include "graptor/itraits.h"

template<typename T>
void __attribute__((noinline))
fill_by_partition( const partitioner & part, T * ptr, T && val ) {
    if constexpr ( is_logical_v<T> ) {
	// Logical types do not trigger call of memset; make it easier
	// for the compiler by digging up the underlying type.
	using U = typename T::type;
	fill_by_partition<U>( part, reinterpret_cast<U *>( ptr ), (U)val );
    } else if( val == 0 || ~val == 0 ) {
	// Specialise to help get memset, in particular for common values
	// where all bytes are equal.
	map_partitionL( part, [=,&part]( unsigned p ) {
		VID ps = part.start_of( p );
		VID pe = part.end_of( p );
		// std::fill<T *,T>( &ptr[ps], &ptr[pe], val );
		char * cps = reinterpret_cast<char *>( &ptr[ps] );
		char * cpe = reinterpret_cast<char *>( &ptr[pe] );
		char cval = (char)val;
		memset( cps, cval, cpe-cps );
	    } );
    } else {
	map_partitionL( part, [=,&part]( unsigned p ) {
		VID ps = part.start_of( p );
		VID pe = part.end_of( p );
		std::fill<T *,T>( &ptr[ps], &ptr[pe], val );
	    } );
    }
}

template<typename T>
void __attribute__((noinline))
copy_by_partition( const partitioner & part, T * to, const T * from ) {
    map_partitionL( part, [=,&part]( unsigned p ) {
	    VID ps = part.start_of( p );
	    VID pe = part.end_of( p );
	    std::copy( &from[ps], &from[pe], &to[ps] );
	} );
}

#endif // GRAPTOR_PRIMITIVES_H

