// -*- c++ -*-
#ifndef GRAPTOR_VCUSTOMFP_H
#define GRAPTOR_VCUSTOMFP_H

#include <x86intrin.h>
#include <immintrin.h>

#include "graptor/itraits.h"

/***********************************************************************
 * This class defines custom floating-point values.
 * Primary usage is in PRv and APRv.
 ***********************************************************************/

namespace detail { // private namespace 

#ifndef VCUSTOMFP_INF
#define VCUSTOMFP_INF 0
#endif // VCUSTOMFP_INF

#ifndef VCUSTOMFP_BIAS
#define VCUSTOMFP_BIAS 1
#endif // VCUSTOMFP_BIAS

/***********************************************************************
 * A custom floating-point value with variably defined configuration.
 *
 * The class config contains static variables that describe the
 * configuration:
 * - number of exponent bits
 * - number of mantissa bits
 * - if sign bit present
 * - if special value zero must be retained upon conversion
 *
 * \tparam Cfg class specifying the configuration
 ***********************************************************************/
template<typename Cfg>
class vcustomfp {
public:
    using cfg = Cfg;
    static constexpr size_t bit_size = cfg::bit_size;
    using int_type = int_type_of_size_t<(bit_size+7)/8>;
    using self_type = vcustomfp<Cfg>;

    explicit vcustomfp( float f ) : m_val( cvt_to_cfp( f ) ) { }
    explicit vcustomfp( double d ) : m_val( cvt_to_cfp( d ) ) { }
    explicit vcustomfp( int_type m ) : m_val( m ) { }
    vcustomfp( const self_type & s ) : m_val( s.m_val ) { }
    vcustomfp() { }

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
	return cfg::max();
    }

private:
    static int_type cvt_to_cfp( float f ) {
#if VCUSTOMFP_BIAS
	// We do not support subnormal numbers, hence, an exponent of zero
	// can only occur for the value zero.
	// This explicit check is required only when a bias is applied to
	// the exponent.
	if( cfg::Z && f == 0.0f )
	    return int_type(0);
#endif
	
	uint32_t mask = (uint32_t(1) << (cfg::E+cfg::M)) - 1;
	uint32_t tt = type_pun<uint32_t>( f );
	uint32_t t = tt;

#if VCUSTOMFP_BIAS
	// TODO: if infty supported, need to correct for that
	if( cfg::B != 0 ) {
	    int32_t b = int32_t( cfg::B ) << 23;
	    uint32_t smask = uint32_t(1) << 31;
	    t &= ~smask;
	    t -= b;
	}
#endif
	
	t >>= (23-cfg::M);
	t &= mask;
	if( cfg::S )
	    t |= (tt>>(31-cfg::M-cfg::E)) & ( uint32_t(1) << (cfg::M+cfg::E) );
	return static_cast<int_type>( t );
    }
    static int_type cvt_to_cfp( double d ) {
	uint64_t mask = (uint64_t(1) << (cfg::E+cfg::M)) - 1;
	uint64_t tt = type_pun<uint64_t>( d );
	uint64_t t = tt;
	t >>= (52-cfg::M);
#if VCUSTOMFP_BIAS // TODO: if infty supported, need to correct for that
	if( cfg::B != 0 ) {
	    int64_t b = cfg::B << cfg::M;
	    uint32_t smask = uint64_t(1) << 63;
	    t &= ~smask;
	    t -= b;
	}
#endif
	t &= mask;
	if( cfg::S )
	    t |= (tt>>(63-cfg::M-cfg::E)) & ( uint64_t(1) << (cfg::M+cfg::E) );
	return static_cast<int_type>( t );
    }
    static float cvt_to_float( int_type m ) {
	if( cfg::Z && m == 0 )
	    return 0.0f;

	uint32_t tt = static_cast<uint32_t>( m );
	uint32_t t = tt;
	t <<= (23-cfg::M); // move exponent and mantissa into position
	if( cfg::S )
	    t &= ~( uint32_t(1) << (cfg::E+23) ); // squelch sign bit
	if( cfg::E < 8 ) {
	    uint32_t ev = ( ( uint32_t(1) << cfg::E ) - 1 ) << 23;
	    uint32_t em = ( uint32_t(1) << (cfg::E+23) ) - 1;
	    if( cfg::I && (t & em) == ev ) // Adjust exponent in case of infty
		t |= uint32_t(255) << 23;
	    else { // Or restore missing exponent bits
		uint32_t e = ( ( uint32_t(1) << (8-cfg::E-1) ) - 1 );
		e <<= (cfg::E + 23);
		t |= e;
	    }
	}
#if VCUSTOMFP_BIAS
	if( cfg::B != 0 ) { // restore bias
	    int32_t b = int32_t( cfg::B ) << 23;
	    t += b;
	}
#endif
	if( cfg::S ) // restore sign bit
	    t |= ( tt & ( uint32_t(1) << (cfg::M+cfg::E) ) )
		<< ( 31 - (cfg::M+cfg::E) );
	return type_pun<float>( t );
    }
    static double cvt_to_double( int_type m ) {
	if( cfg::Z && m == 0 )
	    return 0.0;
	uint64_t tt = static_cast<uint64_t>( m );
	uint64_t t = tt;
	t <<= (52-cfg::M);
	if( cfg::S )
	    t &= ~( uint32_t(1) << (cfg::E+52) ); // squelch sign bit
#if VCUSTOMFP_BIAS
	if( cfg::B != 0 ) { // restore bias
	    int32_t b = cfg::B << 23;
	    t += b;
	}
#endif
	if( cfg::E < 8 ) {
	    uint64_t ev = ( ( uint64_t(1) << cfg::E ) - 1 ) << 52;
	    uint64_t em = ( uint64_t(1) << (cfg::E+52) ) - 1;
	    if( cfg::I && (t & em) == ev ) // Adjust exponent in case of infty
		t |= uint64_t(2047) << 52;
	    else { // Or restore missing exponent bits
		uint64_t e = ( ( uint64_t(1) << (11-cfg::E-1) ) - 1 );
		e <<= (cfg::E + 52);
		t |= e;
	    }
	}
	unsigned ME = cfg::M + cfg::E;
	if( cfg::S ) // restore sign bit
	    t |= ( tt & ( uint64_t(1) << ME ) ) << ( 63 - ME );
	return type_pun<double>( t );
    }

private:
    int_type m_val;
};

template<typename Cfg>
std::ostream & operator << ( std::ostream & os, vcustomfp<Cfg> cfp ) {
    return os << "vcustomfp<s=" << Cfg::S
	      << ",e=" << (uint32_t)Cfg::E
#if VCUSTOMFP_BIAS
	      << '@' << (int32_t)Cfg::B
#else
	      << "@0(fix)"
#endif
	      << ",m=" << (uint32_t)Cfg::M
	      << ",z=" << Cfg::Z
#if VCUSTOMFP_INF
	      << ",i=" << Cfg::I
#else
	      << ",i=0(fix)"
#endif
	      << '=' << (uint32_t)Cfg::bit_size
	      << ">(" << cfp.get() << ')';
}

} // anonymous namespace 

using detail::vcustomfp;

/***********************************************************************
 * Traits
 ***********************************************************************/
// is T vcustomfp?
template<typename T>
struct is_vcustomfp : public std::false_type { };

template<typename Cfg>
struct is_vcustomfp<detail::vcustomfp<Cfg>> : public std::true_type { };

template<typename T>
static constexpr bool is_vcustomfp_v = is_vcustomfp<T>::value;

// is T any of std::floating_point, customfp, and vcustomfp?
template<typename Cfg>
struct is_extended_floating_point<detail::vcustomfp<Cfg>>
    : public std::true_type { };

/***********************************************************************
 * Configuration
 ***********************************************************************/
template<typename DerivedClass>
struct variable_customfp_config {
    using derived_class = DerivedClass;
    using self_type = variable_customfp_config<DerivedClass>;

    static constexpr uint8_t bit_size = derived_class::bit_size;

#if VCUSTOMFP_BIAS
    inline static int32_t B; //!< exponent bias
#else
    constexpr static int32_t B = 0; //!< exponent bias
#endif
    inline static uint8_t M; //!< mantissa bits
    inline static uint8_t E; //!< exponent bits
    inline static bool S;    //!< sign bit
    inline static bool Z;    //!< retain zero value upon conversion
#if VCUSTOMFP_INF
    inline static bool I;    //!< retain infinity value upon conversion
#else
    constexpr static bool I = false; //!< retain infinity value upon conversion
#endif

    static void set_param_for_range( float minval, float maxval,
				     bool inf = false ) {
	assert( minval <= maxval && "argument order" );
	
#if VCUSTOMFP_INF
	I = inf;
#else
	assert( !inf && "support for infinity disabled" );
#endif

	S = minval < 0;
	Z = minval <= 0;

	int32_t iminexp = exponent( minval );
	int32_t imaxexp = exponent( maxval ) + 1;
#if VCUSTOMFP_BIAS
	B = imaxexp;
#else
	assert( imaxexp == B && "bias not configurable" );
#endif
	E = rt_ilog2( imaxexp - iminexp + 1 );
	
	// assert( B == 0 && "bias currently not taken into account" );

	M = bit_size - ( S ? 1 : 0 ) - E;
	assert( M + ( S ? 1 : 0 ) + E == bit_size
		&& "all components must fit in available bits" );
    }

    template<typename ToTy, typename FromTy>
    static constexpr ToTy type_pun( FromTy f ) {
	static_assert( sizeof(ToTy) == sizeof(FromTy), "require same width" );
	return *reinterpret_cast<ToTy*>( &f );
    }

    static int32_t exponent( float f ) {
	uint32_t i = type_pun<uint32_t>( f );
	uint32_t iu = i << 1; // drop sign bit
	int32_t e = iu >> 24; // move exponent into place
	return e - 127; // remove bias
    }

    static vcustomfp<self_type> infinity() {
	using int_type = typename vcustomfp<self_type>::int_type;
#if VCUSTOMFP_INF
	// Return representation of std::numeric_limits<float>::infinity()
	if( I ) {
	    int_type v = ( ( int_type(1) << E ) - 1 ) << M;
	    return vcustomfp<self_type>( v );
	}
#endif
	return max();
    }

    static vcustomfp<self_type> max() {
	using int_type = typename vcustomfp<self_type>::int_type;
	// Return largest representable value
	// Arithmetic on this value does not follow IEEE conventions, i.e.,
	// it is possible that a + inf != inf.
	// largest positive value:
	// sign: positive (0)
	// exponent: 1...10
	// mantissa: 1...1
	int_type v = ~( int_type(1) << M );
#if VCUSTOMFP_BIAS && 0
	if( B != 0 ) { // restore bias
	    int_type b = B << M;
	    v += b;
	    int32_t b = B << M;
	    uint32_t smask = uint32_t(1) << 31;
	    t &= ~smask;
	    t -= b;
	}
#endif
	if( S ) {
	    int_type e = int_type(1) << ( M + E );
	    v &= ~e;
	}
	return vcustomfp<self_type>( v );
    }
};

#endif // GRAPTOR_VCUSTOMFP_H
