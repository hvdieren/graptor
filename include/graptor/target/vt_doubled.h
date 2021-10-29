// -*- c++ -*-
#ifndef GRAPTOR_TARGET_VTDOUBLED_H
#define GRAPTOR_TARGET_VTDOUBLED_H

/***********************************************************************
 * Recursively defined case by doubling (redundant to vt_recursive.h,
 * which is prefered)
 ***********************************************************************/
namespace target {

template<typename T, unsigned short nbytes>
struct vector_type_int_traits_doubled {
    typedef T member_type;
    using half_traits = vector_type_traits<T,nbytes/2>;
    using half_type = typename half_traits::type;
    using half_itraits = vector_type_traits<T, nbytes/2>;
    using half_itype = typename half_itraits::type;

    using type = vpair<half_type>;
    using itype = vpair<half_itype>;
    using mask_type = __mmask16;
    using vmask_type = itype;
    static const size_t size = sizeof(T);
    static const size_t vlen = sizeof(type) / sizeof(T);
    static_assert( size * vlen == sizeof(type), "type size error" );

    static typename half_traits::mask_type mask_a( mask_type mask ) {
	return mask & ((1<<(vlen/2))-1);
    }
    static typename half_traits::mask_type mask_b( mask_type mask ) {
	return mask >> (vlen/2);
    }

    static type setzero() {
	return type{ half_traits::setzero(), half_traits::setzero() };
    }

    static type set_pair( half_type hi, half_type lo ) {
	return type{ lo, hi };
    }

    static type set1( member_type a ) {
	half_type aa = half_traits::set1( a );
	return type { aa, aa };
    }
    static type setone() {
	half_type aa = half_traits::setone();
	return type { aa, aa };
    }
    static type setl0( member_type m ) {
	return type { half_traits::setl0( m ), half_traits::setzero() };
    }

    static type abs( type a ) {
	return type{ half_traits::abs( a.a ), half_traits::abs( a.b ) };
    }
    
    static __mmask16 asmask( type m ) {
	return half_traits::asmask( m.a ) << (vlen / 2)
	    | half_traits::asmask( m.b );
    }

    static bool cmpne( type a, type b, target::mt_bool ) {
	return half_traits::cmpne( a.a, b.a, target::mt_bool() )
	    && half_traits::cmpne( a.b, b.b, target::mt_bool() );
    }
    static vmask_type cmpne( type a, type b, target::mt_vmask ) {
	return type{ half_traits::cmpne( a.a, b.a, target::mt_vmask() ),
		half_traits::cmpne( a.b, b.b, target::mt_vmask() ) };
    }

    static type blend( vmask_type m, type l, type r ) {
	return type{ half_traits::blend( m.a, l.a, r.a ),
		half_traits::blend( m.b, l.b, r.b ) };
    }
    static type blend( mask_type m, type l, type r ) {
	return type{ half_traits::blend( m, l.a, r.a ),
		half_traits::blend( m >> (vlen/2), l.b, r.b ) };
    }
    
    static type logical_or( type a, type b ) {
	return type{ half_traits::logical_or( a.a, b.a ),
		half_traits::logical_or( a.b, b.b ) };
    }
    static type logical_and( type a, type b ) {
	return type{ half_traits::logical_and( a.a, b.a ),
		half_traits::logical_and( a.b, b.b ) };
    }
    static type logical_invert( type a ) {
	return type{ half_traits::logical_invert( a.a ),
		half_traits::logical_invert( a.b ) };
    }
    static type bitwise_or( type a, type b ) {
	return type{ half_traits::bitwise_or( a.a, b.a ),
		half_traits::bitwise_or( a.b, b.b ) };
    }
    static type bitwise_and( type a, type b ) {
	return type{ half_traits::bitwise_and( a.a, b.a ),
		half_traits::bitwise_and( a.b, b.b ) };
    }
    static type bitwise_invert( type a ) {
	return type{ half_traits::bitwise_invert( a.a ),
		half_traits::bitwise_invert( a.b ) };
    }

    static type add( type a, type b ) {
	return type{
	    half_traits::add( a.a, b.a ), half_traits::add( a.b, b.b ) };
    }
    static type add( type src, mask_type m, type a, type b ) {
	return type{ half_traits::add( src.a, m, a.a, b.a ),
		half_traits::add( src.b, m >> (vlen/2), a.b, b.b ) };
    }
    static type add( type src, vmask_type m, type a, type b ) {
	return type{ half_traits::add( src.a, m.a, a.a, b.a ),
		half_traits::add( src.b, m.b, a.b, b.b ) };
    }
    static type sub( type a, type b ) {
	return type{
	    half_traits::sub( a.a, b.a ), half_traits::sub( a.b, b.b ) };
    }
    static type mul( type a, type b ) {
	return type{
	    half_traits::mul( a.a, b.a ), half_traits::mul( a.b, b.b ) };
    }
    static type mul( type src, mask_type m, type a, type b ) {
	return type{ _mm512_mask_mul_pd( src.a, m, a.a, b.a ),
		_mm512_mask_mul_pd( src.b, m >> (vlen/2), a.b, b.b ) };
    }

    static member_type lane( type a, int idx ) {
	// This is ugly - beware
	member_type tmp[vlen];
	// store( tmp, a );
	*(type *)tmp = a; // perhaps compiler can do something with this
	return tmp[idx];
    }
    static half_type lower_half( type a ) { return a.a; }
    static half_type upper_half( type a ) { return a.b; }

    static member_type lane0( type a ) { return *(member_type *)&a; }
    static member_type lane1( type a ) { return *(((member_type *)&a)+1); }
    static member_type lane2( type a ) { return *(((member_type *)&a)+2); }
    static member_type lane3( type a ) { return *(((member_type *)&a)+3); }

    static type load( const member_type *a ) {
	return type { half_traits::load( a ),
		half_traits::load( ((const member_type *)a)+vlen/2 ) };
    }
    static type loadu( const member_type *a ) {
	return type { half_traits::loadu( a ),
		half_traits::loadu( ((const member_type *)a)+vlen/2 ) };
    }
    static void store( member_type *addr, type val ) {
	half_traits::store( addr, val.a );
	half_traits::store( ((member_type *)addr)+vlen/2, val.b );
    }
    static void storeu( member_type *addr, type val ) {
	half_traits::storeu( addr, val.a );
	half_traits::storeu( ((member_type *)addr)+vlen/2, val.b );
    }
    static member_type reduce_add( type val ) {
	return half_traits::reduce_add( val.a )
	    + half_traits::reduce_add( val.b );
    }
    static member_type reduce_add( type val, mask_type mask ) {
	return half_traits::reduce_add( mask, val.a )
	    + half_traits::reduce_add( mask >> (vlen/2), val.b );
    }
    static member_type reduce_bitwiseor( type val ) {
	return half_traits::reduce_bitwiseor( val.a )
	    | half_traits::reduce_bitwiseor( val.b );
    }
    static member_type reduce_logicalor( type val ) {
	return half_traits::reduce_logicalor( val.a )
	    | half_traits::reduce_logicalor( val.b );
    }
    static type asvector( mask_type mask ) {
	return type{ half_traits::asvector( mask_a( mask ) ),
		half_traits::asvector( mask_b( mask ) ) };
    }
    static type
    gather( const member_type *a, itype b ) {
	) {
	return type{ half_traits::gather( a, b.a, size ),
		half_traits::gather( a, b.b, size ) };
    }
    static type
    gather( const member_type *a, itype b, mask_type mask ) {
	half_type ra = half_traits::gather( a, b.a, mask&((1<<(vlen/2))-1) );
	half_type rb = half_traits::gather( a, b.b, mask>>(vlen/2) );
	return type{ ra, rb };
    }
    static type
    gather( const member_type *a, itype b, vmask_type mask ) {
	half_type ra = half_traits::gather( a, b.a, mask.a );
	half_type rb = half_traits::gather( a, b.b, mask.b );
	return type{ ra, rb };
    }
    static type
    gather( const member_type *a, typename half_traits::itype b ) {
	half_type ra = half_traits::gather( a, itraits::lower_half( b ) );
	half_type rb = half_traits::gather( a, itraits::upper_half( b ) );
	return type{ ra, rb };
    }
    static type
    gather( const member_type *a, typename half_traits::itype b,
	    mask_type mask ) {
	half_type ra
	    = half_traits::gather( a, half_traits::lower_half( b ), mtraits::lower_half( mask ) ); // mask&((1<<(vlen/2))-1) );
	half_type rb
	    = half_traits::gather( a, half_traits::upper_half( b ), mtraits::upper_half( mask ) ); // mask>>(vlen/2) );
	return type{ ra, rb };
    }
    static type
    gather( const member_type *a, itype b, vmask_type mask ) {
	) {
	half_type ra
	    = half_traits::gather( a, int_traits::lower_half( b ), int_traits::lower_half( mask ) );
	half_type rb
	    = half_traits::gather( a, int_traits::upper_half( b ), int_traits::upper_half( mask ) );
	return type{ ra, rb };
    }
    static type
    gather( const member_type *a, typename half_traits::itype b,
	    vmask_type mask ) {
	using itraits = typename half_traits::int_traits;
	half_type ra
	    = half_traits::gather( a, itraits::lower_half( b ), lower_half( mask ) );
	half_type rb
	    = half_traits::gather( a, itraits::upper_half( b ), upper_half( mask ) );
	return type{ ra, rb };
    }
    static void
    scatter( member_type *a, itype b, type c ) {
	half_traits::scatter( a, int_traits::lower_half( b ), c.a );
	half_traits::scatter( a, int_traits::upper_half( b ), c.b );
    }
/*
    template<typename IdxT>
    static void
    scatter( member_type *a,
	     IdxT b,
	     // typename vector_type_int_traits<IdxT,vlen*sizeof(IdxT)>::type b,
	     type c ) {
	assert( 0 && "NYI" );
	// half_traits::scatter( a, half_itraits::lower_half( b ), c.a );
	// half_traits::scatter( (a+vlen/2), half_itraits::upper_half( b ), c.b );
    }
*/
    static void
    scatter( member_type *a, itype b, type c, mask_type mask ) {
	half_traits::scatter( a, b.a, c.a, mask & ((1<<(vlen/2))-1) );
	half_traits::scatter( a, b.b, c.b, mask >> (vlen/2) );
    }
    static void
    scatter( member_type *a, itype b, type c, vmask_type mask ) {
	half_traits::scatter( a, b.a, c.a, mask.a );
	half_traits::scatter( a, b.b, c.b, mask.b );
    }
    static void
    scatter( member_type *a, typename half_traits::itype b,
	     type c, vmask_type mask ) {
	using itraits = typename half_traits::itype;
	half_traits::scatter( a, itraits::lower_half(b), c.a, mask.a );
	half_traits::scatter( a, itraits::upper_half(b), c.b, mask.b );
    }
    static void
    scatter( member_type *a, typename half_traits::itype b, type c, mask_type mask ) {
	using itraits = typename half_traits::itype;
	half_traits::scatter( a, itraits::lower_half(b), c.a, mask & ((1<<(vlen/2))-1) );
	half_traits::scatter( a, itraits::upper_half(b), c.b, mask >> (vlen/2) );
    }
};

template<typename T, unsigned short nbytes, typename Enable = void>
struct vector_type_traits_doubled;

template<typename T, unsigned short nbytes>
struct vector_type_traits_doubled<T, nbytes,
				  typename std::enable_if<std::is_integral<T>::value>::type>
    : public vector_type_int_traits_doubled<T, nbytes> { };


} // namespace target

#endif //  GRAPTOR_TARGET_VTDOUBLED_H
