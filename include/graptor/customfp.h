// -*- c++ -*-
#ifndef GRAPTOR_CUSTOMFP_H
#define GRAPTOR_CUSTOMFP_H

#include <x86intrin.h>
#include <immintrin.h>

#include "graptor/itraits.h"

/***********************************************************************
 * This class defines custom floating-point values.
 * Primary usage is in PRv and APRv.
 ***********************************************************************/

namespace detail { // private namespace 

// A floating-point value with no sign bit, E bits exponent and M bits mantissa
// This format lifts E least significant bits out of the exponent and M most
// significant bits out of the mantissa. No rounding is performed.
template<bool S, unsigned short E, unsigned short M, bool Z, int B>
class customfp_em {
public:
    static constexpr bool sign_bit = S;
    static constexpr bool maybe_zero = Z;
    static constexpr unsigned short exponent_bits = E;
    static constexpr unsigned short mantissa_bits = M;
    static constexpr int exponent_bias = B;
    static constexpr size_t bit_size = size_t(S?1:0)+size_t(E)+size_t(M);
    using int_type = int_type_of_size_t<(bit_size+7)/8>;
    using self_type = customfp_em<S,E,M,Z,B>;

    explicit customfp_em( float f ) : m_val( cvt_to_cfp( f ) ) { }
    explicit customfp_em( double d ) : m_val( cvt_to_cfp( d ) ) { }
    explicit customfp_em( int_type m ) : m_val( m ) { }
    customfp_em( const self_type & s ) : m_val( s.m_val ) { }
    customfp_em() { }

    int_type get() const { return m_val; }

    bool operator == ( self_type r ) const {
	return get() == r.get();
    }
    bool operator != ( self_type r ) const {
	return get() != r.get();
    }

    self_type & operator = ( const self_type & s ) {
	m_val = s.m_val;
	return *this;
    }
    self_type & operator = ( float f ) {
	m_val = cvt_to_cfp( f );
	return *this;
    }
    self_type & operator = ( double f ) {
	m_val = cvt_to_cfp( f );
	return *this;
    }

    operator float () const { return cvt_to_float( m_val ); }
    operator double () const { return cvt_to_double( m_val ); }
    operator int_type () const { return get(); }

    template<typename ToTy, typename FromTy>
    static constexpr ToTy type_pun( FromTy f ) {
	static_assert( sizeof(ToTy) == sizeof(FromTy), "require same width" );
	return *reinterpret_cast<ToTy*>( &f );
    }

    static self_type max() {
	// Return largest representable value
	// largest positive value:
	// sign: positive (0)
	// exponent: 1...10
	// mantissa: 1...1
	int_type v = ~( int_type(1) << M );
	if constexpr ( S ) {
	    int_type e = int_type(1) << ( M + E );
	    v &= ~e;
	}
	return self_type( v );
    }

    static self_type min() {
	// Return smallest positive representable value
	// largest positive value:
	// sign: positive (0)
	// exponent: 0...01
	// mantissa: 0...0
	int_type v = int_type(1) << M;
	return self_type( v );
    }

private:
    static int_type cvt_to_cfp( float f ) {
	if constexpr ( Z && B != 0 ) {
	    if( f == 0.0f )
		return int_type(0);
	}

	constexpr uint32_t mask = (uint32_t(1) << (E+M)) - 1;
	uint32_t tt = type_pun<uint32_t>( f );
	uint32_t t = tt;

	if constexpr ( B != 0 ) {
	    int32_t b = int32_t( B ) << 23;
	    uint32_t smask = uint32_t(1) << 31;
	    t &= ~smask;
	    t -= b;
	}

	t >>= (23-M);
	t &= mask;
	if constexpr ( S )
	    t |= (tt>>(31-M-E)) & ( uint32_t(1) << (M+E) );
	return static_cast<int_type>( t );
    }
    static int_type cvt_to_cfp( double d ) {
	if constexpr ( Z && B != 0 ) {
	    if( d == 0.0f )
		return int_type(0);
	}

	constexpr uint64_t mask = (uint64_t(1) << (E+M)) - 1;
	uint64_t tt = type_pun<uint64_t>( d );
	uint64_t t = tt;

	if constexpr ( B != 0 ) {
	    int32_t b = int64_t( B ) << 48; // check
	    uint32_t smask = uint64_t(1) << 63;
	    t &= ~smask;
	    t -= b;
	}

	t >>= (52-M);
	t &= mask;
	if constexpr ( S )
	    t |= (tt>>(63-M-E)) & ( uint64_t(1) << (M+E) );
	return static_cast<int_type>( t );
    }
    static float cvt_to_float( int_type m ) {
	if constexpr ( Z && B != 0 ) {
	    if( m == 0 )
		return 0.0f;
	}
	
	uint32_t tt = static_cast<uint32_t>( m );
	uint32_t t = tt;

	if constexpr ( S && B != 0 ) // squelch sign bit, if any
	    t &= ~( uint32_t(1) << (E+M) );

	if constexpr ( E < 8 ) {
	    constexpr uint32_t e
		= ( ( uint32_t(1) << (8-E-1) ) - 1 ) << (E + M);
	    t |= e;
	}

	t <<= (23-M);

	if constexpr ( B != 0 ) {
	    int32_t b = int32_t( B ) << 23;
	    t += b;
	}
	
	if constexpr ( S )
	    t |= ( tt & ( uint32_t(1) << (M+E) ) ) << ( 31 - (M+E) );
	return type_pun<float>( t );
    }
    static double cvt_to_double( int_type m ) {
	if constexpr ( Z && B != 0 ) {
	    if( m == 0 )
		return 0.0f;
	}
	
	uint64_t tt = static_cast<uint64_t>( m );
	uint64_t t = tt;
	assert( 0 && "NYI" );
	constexpr uint64_t e = ( ( uint64_t(1) << (11-E-1) ) - 1 ) << (E + M);
	t |= e;
	t <<= (52-M);
	if constexpr ( S )
	    t |= ( tt & ( uint64_t(1) << (M+E) ) ) << ( 63 - (M+E) );
	return type_pun<double>( t );
    }

private:
    int_type m_val;
};

template<bool S, unsigned short E, unsigned short M, bool Z, int B>
constexpr
customfp_em<S,E,M,Z,B> operator + ( const customfp_em<S,E,M,Z,B> & a,
				const customfp_em<S,E,M,Z,B> & b ) {
    return customfp_em<S,E,M,Z,B>( ((float)a) + ((float)b) );
}

template<bool S, unsigned short E, unsigned short M, bool Z, int B>
std::ostream & operator << ( std::ostream & os, customfp_em<S,E,M,Z,B> cfp ) {
    return os << "customfp<" << S << ',' << E << ',' << M << ',' << Z
	      << ',' << B
	      << ">(" << cfp.get() << ')';
}

} // anonymous namespace 

template<unsigned short E, unsigned short M>
using customfp = detail::customfp_em<false,E,M,false,0>;

template<bool S, unsigned short E, unsigned short M, bool Z, int B>
using scustomfp = detail::customfp_em<S,E,M,Z,B>;

/***********************************************************************
 * Traits
 ***********************************************************************/
// is T customfp?
template<typename T>
struct is_customfp : public std::false_type { };

// template<unsigned short E, unsigned short M>
// struct is_customfp<customfp<E,M>> : public std::true_type { };

template<bool S, unsigned short E, unsigned short M, bool Z, int B>
struct is_customfp<detail::customfp_em<S,E,M,Z,B>> : public std::true_type { };

template<typename T>
static constexpr bool is_customfp_v = is_customfp<T>::value;

// is T std::floating_point or customfp?
template<typename T>
struct is_extended_floating_point : public std::is_floating_point<T> { };

// template<unsigned short E, unsigned short M>
// struct is_extended_floating_point<customfp<E,M>> : public std::true_type { };

template<bool S, unsigned short E, unsigned short M, bool Z, int B>
struct is_extended_floating_point<detail::customfp_em<S,E,M,Z,B>> : public std::true_type { };

template<typename T>
static constexpr bool is_extended_floating_point_v
= is_extended_floating_point<T>::value;

/***********************************************************************
 * Numeric limits
 ***********************************************************************/
#if 1
namespace std {

template<bool S, unsigned short E, unsigned short M, bool Z, int B>
class numeric_limits<::detail::customfp_em<S,E,M,Z,B>> {
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = S;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false;
    static constexpr bool has_signaling_NaN = false;
    static constexpr bool has_denorm = false;
    static constexpr bool has_denorm_loss = false;
    static constexpr bool round_style = false;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = false;
    static constexpr bool is_modulo = false;
    static constexpr bool digits = false;
    
    static ::detail::customfp_em<S,E,M,Z,B> max() {
	return ::detail::customfp_em<S,E,M,Z,B>::max();
    }
};

} // namespace std
#endif

/***********************************************************************
 * Utility to calculate array length in the context of element types
 * that do not take a full number of bytes.
 * Overrides base template in itraits.h
 ***********************************************************************/
template<unsigned short E, unsigned short M>
struct mm_get_total_bytes<customfp<E,M>> {
    static constexpr size_t get( size_t num_elems ) {
	using T = customfp<E,M>;
	if constexpr ( T::bit_size == 21 )
	    return (num_elems+2)/3 * 8;
	if constexpr ( ( T::bit_size & size_t(7) ) == 0 )
	    return (T::bit_size / 8) * num_elems;
	assert( false && "non-byte-wide customfp type not handled" );
    }
};

/***********************************************************************
 * Utility to index array of elements.
 * Overrides base template in itraits.h
 ***********************************************************************/
template<unsigned short E, unsigned short M>
struct customfp_index {
    using cfp_type = customfp<E,M>;
    using stored_type = typename cfp_type::int_type;
};

#endif // GRAPTOR_CUSTOMFP_H
