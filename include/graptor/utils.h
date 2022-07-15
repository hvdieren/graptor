// -*- C++ -*-
#ifndef GRAPTOR_UTILS_H
#define GRAPTOR_UTILS_H

// For _bit_scan_reverse
#include <x86intrin.h>
#include <immintrin.h>

// Integer types and math
#include <cstdint>
#include <cmath>

// I/O
#include <ostream>
#include <iostream>

// enable_if
#include <type_traits>

// iterators
#include <iterator>
#include <functional>

// algorithms
#include <numeric>

/***********************************************************************
 * Constants
 ***********************************************************************/
#define _BSIZE 2048
#define _SCAN_LOG_BSIZE 10
#define _SCAN_BSIZE (1 << _SCAN_LOG_BSIZE)

/***********************************************************************
 * General utilities
 ***********************************************************************/
template<typename T>
constexpr T ilog2_rec( T t, T l ) {
    return t > 0 ? ilog2_rec( t >> 1, l + 1 ) : l;
}

template<typename T>
constexpr T ilog2( T t ) {
    return t <= 0 ? -1 : ilog2_rec( T(t >> 1), T(0) );
}

template<typename T>
constexpr T next_ipow2( T t ) {
    return T(1) << (ilog2(t-1)+1);
}

template<typename T>
inline
std::enable_if_t<std::is_integral_v<T> && sizeof(T) <= 4,T> rt_ilog2( T a ) {
    return a <= T(0) ? -1 : _bit_scan_reverse( (unsigned int)a );
}

template<typename T>
constexpr T roundup_multiple_pow2( T val, T mult ) {
    // Round-up val to the next multiple of mult, which is a power of 2.
    return ( val + mult - 1 ) & ~( mult - 1 );
}

/*
template<typename T>
T ilog2( T t ) {
    if( t <= 0 )
	return -1;

    T l = 0;
    t >>= 1;
    while( t > 0 ) {
	t >>= 1;
	l++;
    }
    return l;
}
*/

// A simple approximation to least common multiple
template<typename T>
constexpr T lcm( T a, T b ) {
    return ( a % b == 0 ) ? a : ( b % a == 0 ) ? b : a * b;
}

// bit reverse
// https://stackoverflow.com/questions/746171/efficient-algorithm-for-bit-reversal-from-msb-lsb-to-lsb-msb-in-c
inline unsigned int bit_reverse( unsigned int x ) {
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return((x >> 16) | (x << 16));

}

/***********************************************************************
 * For macros
 ***********************************************************************/
#define STRINGIFY(a) #a

/***********************************************************************
 * Inlining control
 ***********************************************************************/
// #define DBG_NOINLINE __attribute__((noinline))
#define DBG_NOINLINE 
#ifndef GG_INLINE
#define GG_INLINE __attribute__((always_inline))
// #define GG_INLINE
#endif // GG_INLINE

/***********************************************************************
 * Switch statement error checking
 ***********************************************************************/
#define UNREACHABLE_CASE_STATEMENT \
    assert( 0 && "Control should not reach here" )

/***********************************************************************
 * Debugging: allows to print types of expressions during compilation to
 * aid with debugging of templated code.
 ***********************************************************************/
/*
template<template<typename> class C, typename Expr>
Expr fail_expose( Expr e ) {
    // Strangely, if the function has void return type, it is entirely
    // ignored by the compiler and the assertion is not triggered.
    static_assert( !C<Expr>::value, "error exposed" );
    return e;
}
*/

template<template<typename> class C, typename Expr>
std::enable_if_t<!C<Expr>::value> fail_expose( Expr e ) { }

template<bool Cond, template<typename> class C>
struct fail_expose_c {
    template<typename Expr>
    static Expr test( Expr e ) {
	return fail_expose<C>( e );
    }
};

template<template<typename> class C>
struct fail_expose_c<false,C> {
    template<typename Expr>
    static Expr test( Expr e ) {
	return e;
    }
};

/***********************************************************************
 * Making class non-copyable
 * Usage:
 * class CantCopy : private NonCopyable<CantCopy> {};
 ***********************************************************************/
template <class T>
class NonCopyable
{
  public: 
    NonCopyable (const NonCopyable &) = delete;
    T & operator = (const T &) = delete;

  protected:
    NonCopyable () = default;
    ~NonCopyable () = default; /// Protected non-virtual destructor
};

/***********************************************************************
 * Printing byte sizes
 ***********************************************************************/
template<typename T>
class pretty_size {
public:
    pretty_size( T size ) : m_size( size ) { }

    const T & get() const { return m_size; }
    operator T () const { return get(); }

    std::ostream & print( std::ostream & os ) const {
	auto flags = os.flags();
	auto prec = os.precision();

	os.setf( std::ios::fixed );
	os.precision( 1 );

	auto & ret = print_action( os );
	
	os.precision( prec );
	os.flags( flags );

	return ret;
    }

private:
    std::ostream & print_action( std::ostream & os ) const {
	if( !has_bits<10>() )
	    return os << m_size << " B";
	if( !has_bits<20>() )
	    return os << div<10>() << " KiB";
	if( !has_bits<30>() )
	    return os << div<20>() << " MiB";
	if( !has_bits<40>() )
	    return os << div<30>() << " GiB";
	return os << div<40>() << " TiB";
    }

    template<unsigned int bits>
    bool has_bits() const {
	if constexpr ( std::is_integral_v<T> )
	    return ( m_size >> bits ) != 0;
	else
	    return m_size > std::pow( 2.0, bits );
    }

    template<unsigned int bits>
    float div() const {
	return float(m_size) / (float)( size_t(1)<<bits );
    }

private:
    T m_size;
};

template<typename T>
std::ostream & operator << ( std::ostream & os, pretty_size<T> && ps ) {
    return ps.print( os );
}

template<typename T>
pretty_size<T> pretty( T && t ) {
    return pretty_size<T>( std::forward<T>( t ) );
}

/***********************************************************************
 * paired_sort
 * Handy, e.g., when sorting data in corresponding positions of distinct
 * arrays. Goal is to sort by T s...e and adjust U along with it.
 * T, U are RandomAccessIterators
 ***********************************************************************/
template<typename I, typename T>
struct paired_sort_cmp {
    paired_sort_cmp( T s_ ) : s( s_ ) { }
    bool operator() ( I a, I b ) const {
	return *(s+a) < *(s+b);
    }
    const T s;
};

template<typename T, typename U>
void paired_sort( T s, T e, U u ) {
    using namespace std; // for ADL of swap
    
    // Check array length
    using I = typename std::iterator_traits<T>::difference_type;
    I len = std::distance( s, e );
    if( len <= 1 ) // easy
	return;

    if( len == 2 ) {
	if( *s > *(s+1) ) {
	    swap( *s, *(s+1) );
	    swap( *u, *(u+1) );
	}
	return;
    }
    
    // Set up index array
    I idx_buf[128];
    I * idx = len > 128 ? new I[len] : &idx_buf[0];
    std::iota( &idx[0], &idx[len], 0 );

    // Sort indices. A functor argument is more efficient than a lambda
    std::sort( &idx[0], &idx[len], paired_sort_cmp<I,T>( s ) );
    
    // Apply sorted permutation to arrays
    for( I i=0; i < len-1; ++i ) {
	I prev = i;
	I pos = idx[i];
	while( pos < i ) { // add path splitting optimisation
	    idx[prev] = idx[pos];
	    prev = pos;
	    pos = idx[pos];
	}

	swap( *(s+i), *(s+pos) );
	swap( *(u+i), *(u+pos) );
    }

    // Cleanup
    if( len > 128 )
	delete[] idx;
}

#endif // GRAPTOR_UTILS_H
