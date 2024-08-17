// -*- c++ -*-
#ifndef GRAPTOR_TARGET_SCALARINT_H
#define GRAPTOR_TARGET_SCALARINT_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/itraits.h"
#include "graptor/target/decl.h"
#include "graptor/target/scalar_bool.h"
#include "graptor/target/bitmask.h"

namespace target {

/***********************************************************************
 * Scalar integers, not bool
 ***********************************************************************/
template<typename T>
struct scalar_int {
    static_assert( is_integral_or_logical<T>::value
		   && !std::is_same<T,bool>::value,
		   "version of template class for integers, non-bool" );
public:
    using member_type = T;
    using type = T;
    using vmask_type = typename int_type_of_size<sizeof(T)>::type;
    using itype = vmask_type;
    using int_type = vmask_type;

    // TODO: traits should be bool-based?
    // using mask_traits = mask_type_traits<1>;
    using mask_traits = scalar_bool;
    using mask_type = typename mask_traits::type;

    // half_traits not defined (already down to scalar)
    using int_traits = scalar_int<int_type>;
    
    static constexpr size_t W = sizeof(T);
    static constexpr size_t B = 8*W;
    static constexpr size_t vlen = 1;
    static constexpr size_t size = W * vlen;

    static void print( std::ostream & os, type v ) {
	os << '(' << lane0(v) << ')';
    }

    static type setone() { return ~type(0); }
    static type setone_shr1() {
	if constexpr ( is_logical_v<type> )
	    return type( setone().get() >> 1 );
	else
	    return setone() >> 1;
    }
    static type setoneval() { return type(1); }
    static type setglobaloneval( size_t pos ) { return type(1) << pos; }

    // Generate a mask where all bits l and above are set, and below l are 0
    static type himask( unsigned l ) {
	if( l <= B )
	    return ~( ( type(1) << (l-1) ) - 1 );
	else
	    return 0;
    }

    static bool is_zero( type a ) { return a == 0; }
    static bool is_bitwise_and_zero( type a, type b ) { return ( a & b ) == 0; }
    
    static type create( member_type a0_ ) { return a0_; }
    static type set1( member_type a ) { return a; }
    static type set( member_type a0 ) { return a0; }
    static GG_INLINE type setzero() { return member_type(0); }
    static type set1inc( member_type a ) { return a; } // TODO: error
    static type set1inc0() { return 0; }

    static member_type lane( type a, int idx ) { return member_type(a); }
    static member_type lane0( type a ) { return member_type(a); }
    static type setlane( type a, member_type b, int idx ) { return type(b); }

    template<typename U>
    type convert( U u ) {
	static_assert( std::is_integral_v<U>, "requires integral type" );
	return u;
    }
    
    // Casting here is to do a signed cast of a bitmask for logical masks
    template<typename T2>
    static auto convert_to( type a ) {
	return T2(a);
    }

    template<typename U>
    static auto asvector( type a ) { return U(a); }

    static type abs( type a ) { return std::abs( a ); }
    static type sqrt( type a ) { return a * a; }
    
    static type add( type s, mask_type m, type a, type b ) {
	return m ? a + b : s;
    }
    static type add( type a, type b ) { return a + b; }
    static type adds( type a, type b ) {
	if constexpr ( std::is_signed_v<member_type> ) {
	    type c;
	    if( !__builtin_add_overflow( a, b, &c ) )
		c = ~(member_type)0;
	    return c;
	} else {
	    type c = a + b;
	    if( c < a )
		c = ~(member_type)0;
	    return c;
	}
    }
    static type sub( type a, type b ) { return a - b; }
    static type mul( type a, type b ) { return a * b; }
    static type mulhi( type a, type b ) {
	static_assert( 2*W == sizeof(uint64_t), "assumption" );
	return static_cast<type>(
	    ( static_cast<int64_t>( a ) * static_cast<int64_t>( b ) ) >> B );
    }
    static type div( type a, type b ) { return a / b; }
    static type mod( type a, type b ) { return a % b; }

    static member_type reduce_add( type a ) { return a; }
    static member_type reduce_bitwiseor( type a ) { return a; }

    static type logical_and( type a, type b ) { return a & b; }
    static type logical_andnot( type a, type b ) {
	if constexpr ( is_logical_v<member_type> )
	    return ~a & b;
	else
	    return !a & b;
    }
    static type logical_or( type a, type b ) { return a | b; }
    static type logical_invert( type a ) {
	if constexpr ( is_logical_v<member_type> )
	    return ~a;
	else
	    return !a;
    }
    static type bitwise_and( type a, type b ) { return a & b; }
    static type bitwise_andnot( type a, type b ) { return ~a & b; }
    static type bitwise_andnot( type a, type b, type c ) {
	return ~a & ( b & c );
    }
    static type bitwise_or( type a, type b ) { return a | b; }
    static type bitwise_or_and( type a, type b, type c ) {
	return ( a | b ) & c;
    }
    static type bitwise_xor( type a, type b ) { return a ^ b; }
    static type bitwise_xnor( type a, type b ) { return ~a ^ b; }
    static type bitwise_invert( type a ) { return ~a; }

    static auto castfp( type a ) {
	if constexpr ( W >= 8 )
	    return static_cast<double>( a );
	else
	    return static_cast<float>( a );
    }

    static vpair<type,type> divmod3( type a ) {
	return vpair<type,type>{ a/3, a%3 };
    }
    
    static vmask_type cmpeq( type a, type b, target::mt_vmask ) { return a == b ? ~vmask_type(0) : vmask_type(0); }
    static vmask_type cmpne( type a, type b, target::mt_vmask ) { return a != b ? ~vmask_type(0) : vmask_type(0); }
    static vmask_type cmplt( type a, type b, target::mt_vmask ) { return a < b ? ~vmask_type(0) : vmask_type(0); }
    static vmask_type cmple( type a, type b, target::mt_vmask ) { return a <= b ? ~vmask_type(0) : vmask_type(0); }
    static vmask_type cmpgt( type a, type b, target::mt_vmask ) { return a > b ? ~vmask_type(0) : vmask_type(0); }
    static vmask_type cmpge( type a, type b, target::mt_vmask ) { return a >= b ? ~vmask_type(0) : vmask_type(0); }
    static vmask_type cmpneg( type a, target::mt_vmask ) { return a < member_type(0) ? ~vmask_type(0) : vmask_type(0); }
    static vmask_type msbset( type a, target::mt_vmask ) {
	using sty = std::make_signed_t<member_type>;
	return sty(a) < sty(0) ? ~vmask_type(0) : vmask_type(0);
    }
    static mask_type cmpeq( type a, type b, target::mt_mask ) { return a == b ? mask_traits::setone() : mask_traits::setzero(); }
    static mask_type cmpne( type a, type b, target::mt_mask ) { return a != b ? mask_traits::setone() : mask_traits::setzero(); }
    static mask_type cmplt( type a, type b, target::mt_mask ) { return a < b ? mask_traits::setone() : mask_traits::setzero(); }
    static mask_type cmple( type a, type b, target::mt_mask ) { return a <= b ? mask_traits::setone() : mask_traits::setzero(); }
    static mask_type cmpgt( type a, type b, target::mt_mask ) { return a > b ? mask_traits::setone() : mask_traits::setzero(); }
    static mask_type cmpge( type a, type b, target::mt_mask ) { return a >= b ? mask_traits::setone() : mask_traits::setzero(); }
    static vmask_type cmpneg( type a, target::mt_mask ) {
	return a < member_type(0) ? mask_traits::setone() : mask_traits::setzero();
    }
    static vmask_type msbset( type a, target::mt_mask ) {
	using sty = std::make_signed_t<member_type>;
	return sty(a) < sty(0) ? mask_traits::setone() : mask_traits::setzero();
    }
    static bool cmpne( type a, type b, target::mt_bool ) { return a != b; }
    static bool cmpeq( type a, type b, target::mt_bool ) { return a == b; }
    static bool cmplt( type a, type b, target::mt_bool ) { return a < b; }
    static bool cmple( type a, type b, target::mt_bool ) { return a <= b; }
    static bool cmpgt( type a, type b, target::mt_bool ) { return a > b; }
    static bool cmpge( type a, type b, target::mt_bool ) { return a >= b; }
    static bool cmpneg( type a, target::mt_bool ) { return a < member_type(0); }
    static bool msbset( type a, target::mt_bool ) {
	using sty = std::make_signed_t<member_type>;
	return sty(a) < sty(0);
    }

    // Second definition is chosen if type auto-casts to bool
    // static type blend( type c, type a, type b ) { return c != 0 ? b : a; }
    static type blend( bool c, type a, type b ) { return c ? b : a; }
    static type bitblend( type m, type l, type r ) {
	return bitwise_or( bitwise_and( m, r ),
			   bitwise_andnot( m, l ) );
    }

    static member_type blendm( mask_type m, type l, type r ) {
	return m ? r : l;
    }
    static member_type iforz( mask_type m, type a ) {
	return m ? a : setzero();
    }
    static member_type reduce_setif( type val ) { return val; }
    static member_type reduce_setif( type val, mask_type mask ) {
	return mask ? val : ~member_type(0);
    }
    static mask_type from_int( type a ) { return movemask( a ); }
    static mask_type movemask( type a ) { return a ? ~mask_type(0) : mask_type(0); }
    static mask_type asvector( type a ) { return a; }
    static mask_type asmask( type a ) { return movemask( a ); }
    static member_type reduce_logicalor( type val ) { return member_type(val != 0); }
    static bool reduce_logicaland( type a, type b ) {
	return ( logical_and( a, b ) ) == 0;
    }
    static bool reduce_logicalandz( type val ) { return !(val != 0); }
    static type min( type a, type b ) { return std::min( a, b ); }
    static type max( type a, type b ) { return std::max( a, b ); }
    static member_type reduce_min( type val ) { return val; }
    static member_type reduce_max( type val ) { return val; }

    template<typename U>
    static type
    sllv( member_type val, U sh ) {
	return val << sh;
    }
    template<typename U>
    static type
    srlv( member_type val, U sh ) {
	return val >> sh;
    }
    template<typename U>
    static type
    sll( member_type val, U sh ) {
	return val << sh;
    }
    template<typename U>
    static type
    srl( member_type val, U sh ) {
	return val >> sh;
    }
    static member_type slli( member_type val, unsigned int sh ) {
	return val << sh;
    }
    static member_type srli( member_type val, unsigned int sh ) {
	return val >> sh;
    }
    static member_type bsrli( member_type val, unsigned int sh ) {
	return val >> (8*sh);
    }
    static member_type srav( member_type val, member_type sh ) {
	return static_cast<std::make_signed_t<member_type>>( val ) >> sh;
    }
    static member_type srai( member_type val, unsigned int sh ) {
	return static_cast<std::make_signed_t<member_type>>( val ) >> sh;
    }

    template<typename ReturnTy>
    static ReturnTy tzcnt( type a ) {
	if constexpr ( sizeof(member_type) == 8 ) {
	    if constexpr ( std::is_signed_v<member_type> )
		return static_cast<ReturnTy>( _mm_tzcnt_64( a ) );
	    else
		return static_cast<ReturnTy>( _tzcnt_u64( a ) );
	} else if constexpr ( sizeof(member_type) <= 4 ) {
	    if constexpr ( std::is_signed_v<member_type> )
		return static_cast<ReturnTy>( _mm_tzcnt_32( a ) );
	    else
		return static_cast<ReturnTy>( _tzcnt_u32( a ) );
	} else {
	    assert( 0 && "NYI" );
	    return 0;
	}
    }

    template<typename ReturnTy>
    static ReturnTy lzcnt( type a ) {
	if constexpr ( W <= 4 )
	    return static_cast<ReturnTy>( _lzcnt_u32( a ) );
	else if constexpr ( W <= 8 )
	    return static_cast<ReturnTy>( _lzcnt_u64( a ) );
	else {
	    assert( 0 && "NYI" );
	    return 0;
	}
    }

    static type find_first( const member_type v ) {
	return v == 0 ? 0 : 1;
    }

    static constexpr bool has_ternary = false;

    static auto popcnt( type v ) {
	if constexpr ( size == 64 )
	    return _popcnt64( v );
	else
	    return _popcnt32( v );
    }

    static type load( const member_type *a ) { return *a; }
    static type loadu( const member_type *a ) { return *a; }
    static void store( member_type *addr, type val ) { *addr = member_type(val); }
    static void storeu( member_type *addr, type val ) { *addr = member_type(val); }
    static type ntload( const member_type *a ) {
	// Need to use wider load
	void * addr = reinterpret_cast<__m128 *>(
	    uintptr_t(a) & ~uintptr_t(0xf) );
	__m128i ival;
	__asm__ __volatile__( "\n\tmovntdqa (%1),%0"
			      : "=x"(ival) : "r"(addr) : );
	if constexpr ( size == 4 ) {
	    uintptr_t off = ( uintptr_t(a) >> 2 ) & uintptr_t(0x3);
	    switch( off ) {
	    case 0: return (member_type) _mm_extract_epi32( ival, 0 );
	    case 1: return (member_type) _mm_extract_epi32( ival, 1 );
	    case 2: return (member_type) _mm_extract_epi32( ival, 2 );
	    case 3: return (member_type) _mm_extract_epi32( ival, 3 );
	    default:
		assert( 0 && "error" );
	    }
	} else if constexpr ( size == 8 ) {
	    uintptr_t off = ( uintptr_t(a) >> 3 ) & uintptr_t(0x1);
	    switch( off ) {
	    case 0: return (member_type) _mm_extract_epi64( ival, 0 );
	    case 1: return (member_type) _mm_extract_epi64( ival, 1 );
	    default:
		assert( 0 && "error" );
	    }
	} else {
	    assert( 0 && "ntload not implemented for current W" );
	}
    }
    static void ntstore( member_type *addr, type val ) {
	if constexpr ( size == 4 ) {
	    _mm_stream_si32( (int *)addr, (int)val );
	} else if constexpr ( size == 8 ) {
	    _mm_stream_si64( (long long int *)addr, (int64_t)val );
	} else
	    store( addr, val );
    }
    template<unsigned short Scale, typename IdxT>
    static type
    gather_w( const member_type *addr, IdxT idx ) {
	const char * p = reinterpret_cast<const char *>( addr );
	const char * q = p + Scale * idx;
	const member_type * r = reinterpret_cast<const member_type *>( q );
	return r[idx];
    }
    template<unsigned short Scale, typename IdxT>
    static type
    gather_w( const member_type *addr, IdxT idx, mask_type mask ) {
	if( mask ) {
	    const char * p = reinterpret_cast<const char *>( addr );
	    const char * q = p + Scale * idx;
	    const member_type * r = reinterpret_cast<const member_type *>( q );
	    return r[idx];
	} else
	    return setzero();
    }
    template<typename IdxT>
    static type gather( const member_type *addr, IdxT idx ) {
	return addr[idx];
    }
    template<typename IdxT>
    static type
    gather( const member_type *addr, IdxT idx, mask_type mask ) {
	return mask ? addr[idx] : member_type(0);
    }
    template<typename IdxT>
    static void
    scatter( member_type *a, IdxT b, type c ) {
	a[b] = lane0(c);
    }
    template<typename IdxT>
    static void
    scatter( member_type *a, IdxT b, type c, vmask_type mask ) {
	if( scalar_int<vmask_type>::lane0(mask) )
	    a[b] = lane0(c);
    }

    static bool
    cas( volatile member_type * addr, member_type oldval, member_type newval ) {
	if constexpr ( is_logical_v<member_type> ) {
	    using native_type = typename member_type::type;
	    return __sync_bool_compare_and_swap(
		reinterpret_cast<volatile native_type *>( addr ),
		oldval.get(), newval.get() );
	} else {
	    return __sync_bool_compare_and_swap(
		const_cast<member_type *>( addr ), oldval, newval );
	}
    }
};

} // namespace target

#endif // GRAPTOR_TARGET_SCALARINT_H
