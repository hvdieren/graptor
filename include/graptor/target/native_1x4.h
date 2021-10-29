// -*- c++ -*-
#ifndef GRAPTOR_TARGET_NATIVE_1x4_H
#define GRAPTOR_TARGET_NATIVE_1x4_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"
#include "graptor/target/vt_recursive.h"
// #include "graptor/target/sse42_1x16.h"

namespace target {

/***********************************************************************
 * Native 4 byte-sized integers
 * This is poorly tested
 ***********************************************************************/
template<typename T = uint8_t>
struct native_1x4 {
    static_assert( sizeof(T) == 1, 
		   "version of template class for 1-byte integers" );
public:
    using member_type = T;
    using type = uint32_t;
    using vmask_type = uint32_t;
    using itype = uint32_t;
    using int_type = uint8_t;

    using mtraits = mask_type_traits<4>;
    using mask_type = typename mtraits::type;

    // using half_traits = vector_type_int_traits<member_type,16>;
    // using recursive_traits = vt_recursive<member_type,1,32,half_traits>;
    // using int_traits = sse42_1x16<int_type>;
    
    static const size_t size = 1;
    static const size_t vlen = 4;

    static void print( std::ostream & os, type v ) {
	os << '(' << lane0(v) << ',' << lane1(v)
	   << ',' << lane2(v) << ',' << lane3(v) << ')';
    }
    
    static type create( member_type a3_, member_type a2_,
			member_type a1_, member_type a0_ ) {
	type a3 = a3_, a2 = a2_, a1 = a1_, a0 = a0_;
	return (a3 << 24) | (a2 << 16) | (a1 << 8) | (a0 << 8);
    }
    static type set1( member_type a ) {
	type aa = a;
	aa = (aa << 8) | aa;
	return (aa << 16) | aa;
    }
    static type set( member_type a3, member_type a2,
		     member_type a1, member_type a0 ) {
	type t3 = a3, t2 = a2, t1 = a1, t0 = a0;
	return (t3 << 24) | (t2 << 16) | (t1 << 8) | t0;
    }
    static type setzero() { return 0; }
    static type set1inc( member_type a ) {
	return type(0x03020100) + set1(a);
    }

    static member_type lane( type a, int idx ) {
	return ( a >> (8*idx) ) & 0xff;
    }
    static member_type lane0( type a ) { return (member_type)(a & 0xff); }
    static member_type lane1( type a ) { return (member_type)((a >> 8) & 0xff); }
    static member_type lane2( type a ) { return (member_type)((a >> 16) & 0xff); }
    static member_type lane3( type a ) { return (member_type)((a >> 24) & 0xff); }
    
    template<typename VecT2>
    static auto convert( VecT2 a ) {
	typedef vector_type_traits<
	    typename int_type_of_size<sizeof(VecT2)/vlen>::type,
	    sizeof(VecT2)> traits2;
	return set( traits2::lane3(a), traits2::lane2(a),
		    traits2::lane1(a), traits2::lane0(a) );
    }

    // Casting here is to do a signed cast of a bitmask for logical masks
    template<typename T2>
    static auto convert_to( type a ) {
	typedef vector_type_traits<T2,sizeof(T2)*vlen> traits2;
	static_assert( sizeof(typename traits2::type) == sizeof(type),
		       "NYI if sizes differ" );
	return typename traits2::type(a);
    }
    
    
    static type add( type a, type b ) {
	assert( 0 && "NYI" );
	return 0;
    }
    static type cmpeq( type a, type b ) {
	// What if we "perform" this on MMX rather than 4xbool?
	__m64 va = _mm_cvtsi32_si64( a );
	__m64 vb = _mm_cvtsi32_si64( b );
	return _mm_cvtsi64_si32( _mm_cmpeq_pi8( va, vb ) );
    }
    static type cmpne( type a, type b ) {
	// What if we "perform" this on MMX rather than 4xbool?
	__m64 va = _mm_cvtsi32_si64( a );
	__m64 vb = _mm_cvtsi32_si64( b );
	__m64 eq = _mm_cmpeq_pi8( va, vb );
	__m64 ones = _mm_cvtsi32_si64( ~type(0) );
	eq = _mm_xor_si64( ones, eq );
	return _mm_cvtsi64_si32( eq );
    }
    static type blend( type c, type a, type b ) {
	assert( 0 && "NYI" );
	return 0;
    }
    static mask_type cmpeq_mask( type a, type b ) {
	assert( 0 && "NYI" );
	return 0;
    }
    static mask_type cmpeq_mask( type a, type b, mask_type m ) {
	assert( 0 && "NYI" );
	return 0;
    }
    static auto logical_and( type a, type b ) {
	return a & b;
    }
    static auto logicalor_bool( type &a, type b ) {
	auto mod = ~a & b;
	a |= b;
	return mod;
    }
    static mask_type movemask( vmask_type a ) {
	a = a & 0x08040201;
	a |= a >> 14;
	a |= a >> 7;
	return a & 0xf;
	// return ( a | (a >> 14) | (a >> 7) | (a >> 21) ) & 0xf;
    }
    static mask_type packbyte( type a ) {
	a = a & 0x01010101;
	a |= a >> 14;
	a |= a >> 7;
	return a & 0xf;
	// return ( a | (a >> 14) | (a >> 7) | (a >> 21) ) & 0xf;
    }
    // TODO: need proper support for bool's that are all ones
    static __m256i expandmask8( type a ) { // a are boolean bytes
	__m128i ones = _mm_cvtsi32_si128( 0x01010101 );
	__m128i x = _mm_cvtsi32_si128( ~a ); // load inverse in vector (lo half)
	__m128i y = _mm_add_epi8( ones, x ); // convert 0/1 to all 0 or all 1
	return _mm256_cvtepi8_epi64( y );
    }
    static member_type reduce_setif( type val ) {
	assert( 0 && "NYI" );
	return 0; // TODO
    }
    static member_type reduce_add( type val ) {
	assert( 0 && "NYI" );
	return 0; // _mm256_reduce_add_epi64( val );
    }
    static member_type reduce_add( type val, mask_type mask ) {
	assert( 0 && "NYI" );
	return 0; // _mm256_reduce_add_epi64( val );
    }
    static member_type reduce_logicalor( type val ) { return val != 0; }

    static type load( const member_type *addr ) { return loadu( addr ); }
    static void store( member_type *addr, type val ) { storeu( addr, val ); }
    static type loadu( const member_type *addr ) {
	return *(const type *)addr;
    }
    static void storeu( member_type *addr, type val ) {
	*(type *)addr = val;
    }
    
    template<typename IdxT>
    static type gather( member_type *addr, IdxT idx ) {
	typedef vector_type_traits_with<IdxT,vlen> index_traits;
	// bool gather
	return addr[index_traits::lane0(idx)]
	    + ( addr[index_traits::lane1(idx)] << 8 )
	    + ( addr[index_traits::lane2(idx)] << 16 )
	    + ( addr[index_traits::lane3(idx)] << 24 );
    }
    template<typename IdxT>
    static type vgather( member_type *addr, IdxT idx, IdxT mask ) {
	using itraits = vector_type_traits_with<IdxT,vlen>;
	return gather( addr, idx, itraits::movemask( mask ) );
    }
    template<typename IdxT>
    static type gather( member_type *addr, IdxT idx, mask_type mask ) {
	typedef vector_type_traits_with<IdxT,vlen> index_traits;
/*
	__m256i zero = _mm256_setzero_si256();
	__m256i g = _mm256_mask_i32gather_epi32( zero, addr, idx, mask, scale );
	__m256i filter = _mm256_set1_epi32( 0xff );
	__m256i gf = g & filter;
	return gf; // too long...
*/

	// bool gather
	type agg = 0;
	if( (mask >> 0) & 1 )
	    agg |= addr[index_traits::lane0(idx)];
	if( (mask >> 1) & 1 )
	    agg |= addr[index_traits::lane1(idx)] << 8;
	if( (mask >> 2) & 1 )
	    agg |= addr[index_traits::lane2(idx)] << 16;
	if( (mask >> 3) & 1 )
	    agg |= addr[index_traits::lane3(idx)] << 24;
	return agg;
    }
    template<typename IdxT>
    static void scatter( member_type *addr,
			 typename vector_type_int_traits<IdxT,vlen*sizeof(IdxT)>::type idx,
			 type val ) {
	using traits2 = vector_type_int_traits<IdxT,vlen*sizeof(IdxT)>;
	addr[traits2::lane0(idx)] = lane0(val);
	addr[traits2::lane1(idx)] = lane1(val);
	addr[traits2::lane2(idx)] = lane2(val);
	addr[traits2::lane3(idx)] = lane3(val);
    }
    template<typename IdxT, typename MaskT>
    static void scatter( member_type *addr,
			 typename vector_type_int_traits<IdxT,vlen*sizeof(IdxT)>::type idx,
			 type val,
			 typename vector_type_int_traits<MaskT,vlen*sizeof(MaskT)>::type mask ) {
	using traits2 = vector_type_int_traits<IdxT,vlen*sizeof(IdxT)>;
	using traits3 = vector_type_int_traits<MaskT,vlen*sizeof(MaskT)>;
	if( traits3::lane0(mask) ) addr[traits2::lane0(idx)] = lane0(val);
	if( traits3::lane1(mask) ) addr[traits2::lane1(idx)] = lane1(val);
	if( traits3::lane2(mask) ) addr[traits2::lane2(idx)] = lane2(val);
	if( traits3::lane3(mask) ) addr[traits2::lane3(idx)] = lane3(val);
    }
};

} // namespace target

#endif // GRAPTOR_TARGET_NATIVE_1x4_H
