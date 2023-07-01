// -*- c++ -*-
#ifndef GRAPTOR_TARGET_VTRECURSIVE_H
#define GRAPTOR_TARGET_VTRECURSIVE_H

#include "graptor/target/decl.h"
#include "graptor/target/bitmask.h"

/***********************************************************************
 * Recursively defined case
 ***********************************************************************/
namespace target {

template<typename T, unsigned short W_, unsigned short nbytes_,
	 typename lo_half_traits_, typename hi_half_traits_ = lo_half_traits_>
struct vt_recursive {
    static_assert( sizeof(T) == W_, "restriction" );
    
    static constexpr unsigned short W = W_;
    static constexpr unsigned short B = 8*W_;
    static constexpr unsigned short size = nbytes_;
    static constexpr unsigned short vlen = nbytes_/W_;
    using lo_half_traits = lo_half_traits_;
    using lo_half_int_traits = typename lo_half_traits::int_traits;
    using hi_half_traits = hi_half_traits_;
    using hi_half_int_traits = typename hi_half_traits::int_traits;

    using member_type = T;
    using lo_half_type = typename lo_half_traits::type;
    using hi_half_type = typename hi_half_traits::type;
    using type = vpair<lo_half_type,hi_half_type>;
    using itype = vpair<typename lo_half_traits::itype,
			typename hi_half_traits::itype>;
    using vmask_type = vpair<typename lo_half_traits::vmask_type,
			     typename hi_half_traits::vmask_type>;
    // using mask_type = mask_type_t<vlen>;

    // int_type covers both indices and vmask of the same width as member_type
    static_assert( std::is_same_v<
		   typename lo_half_traits::int_type,
		   typename hi_half_traits::int_type>, "int_type match" );
    using int_type = typename lo_half_traits::int_type;
    using int_traits =
	vt_recursive<int_type, W, nbytes_,
		     lo_half_int_traits, hi_half_int_traits>;
    using mask_traits = mask_type_traits<vlen>;
    using mtraits = mask_traits; // TODO: remove mtraits
    using mask_type = typename mask_traits::type;

    using mt_preferred = typename lo_half_traits::mt_preferred;

    // using hint_type = typename int_type_of_size<sizeof(int_type)/2>::type;
    // using hitraits = vector_type_traits<hint_type, sizeof(hint_type)*vlen>;
    // using hitype = typename hitraits::type;

    static bool is_zero( type v ) {
	return lo_half_traits::is_zero( lower_half( v ) )
	    && hi_half_traits::is_zero( upper_half( v ) );
    }
    
    static type setzero() {
	auto l = lo_half_traits::setzero();
	if constexpr ( std::is_same_v<lo_half_traits,hi_half_traits> )
	    return type { l, l };
	else
	    return type { l, hi_half_traits::setzero() };
    }
    static type setone() {
	auto l = lo_half_traits::setone();
	if constexpr ( std::is_same_v<lo_half_traits,hi_half_traits> )
	    return type { l, l };
	else
	    return type { l, hi_half_traits::setone() };
    }
    static type setoneval() {
	auto l = lo_half_traits::setoneval();
	if constexpr ( std::is_same_v<lo_half_traits,hi_half_traits> )
	    return type { l, l };
	else
	    return type { l, hi_half_traits::setoneval() };
    }
    template<typename AT1, typename AT0>
    static std::enable_if_t<sizeof(AT0)==lo_half_traits::size
			    && sizeof(AT1)==hi_half_traits::size,type>
    set( AT1 a1, AT0 a0 ) {
	return type { a0, a1 };
    }
    template<typename AT1, typename AT0>
    static std::enable_if_t<sizeof(AT0)==lo_half_traits::size
			    && sizeof(AT1)==hi_half_traits::size,type>
    set_pair( AT1 a1, AT0 a0 ) {
	return set( a1, a0 );
    }
/*
    template<typename AT>
    static std::enable_if_t<sizeof(AT)==(nbytes_/4),type>
    set( AT a3, AT a2, AT a1, AT a0 ) {
	return type { half_traits::set( a1, a0 ), half_traits::set( a3, a2 ) };
    }
*/
    static type set1( member_type a ) {
	return type { lo_half_traits::set1( a ), hi_half_traits::set1( a ) };
    }
    static itype set1inc( member_type a ) {
	return itype {
	    lo_half_int_traits::set1inc( a ),
		hi_half_int_traits::set1inc( lo_half_int_traits::vlen +  a ) };
    }
    static itype set1inc0() {
	return itype {
	    lo_half_int_traits::set1inc0(),
		hi_half_int_traits::set1inc( lo_half_int_traits::vlen ) };
    }
    static type setl0( member_type a ) {
	return type { lo_half_traits::setl0( a ), hi_half_traits::setzero() };
    }

    static type setglobaloneval( size_t pos ) {
	if( pos < lo_half_traits::size*8 )
	    return type { lo_half_traits::setglobaloneval( pos ),
			  hi_half_traits::setzero() };
	else
	    return type { lo_half_traits::setzero(),
			  hi_half_traits::setglobaloneval(
			      pos - lo_half_traits::size*8 ) };
    }
    // Generate a mask where all bits l and above are set, and below l are 0
    static type himask( unsigned pos ) {
	if( pos < lo_half_traits::size*8 )
	    return type { lo_half_traits::himask( pos ),
			  hi_half_traits::setone() };
	else
	    return type { lo_half_traits::setzero(),
			  hi_half_traits::himask(
			      pos - lo_half_traits::size*8 ) };
    }

    static member_type lane( type a, unsigned int l ) {
	if( l >= lo_half_traits::vlen )
	    return hi_half_traits::lane( a.b, l-lo_half_traits::vlen );
	else
	    return lo_half_traits::lane( a.a, l );
    }
    static member_type lane0( type a ) { return lo_half_traits::lane0( a.a ); }

    static type setlane( type a, member_type s, unsigned int l ) {
	if( l >= lo_half_traits::vlen )
	    return type { a.a,
		    hi_half_traits::setlane( a.b, s, l-lo_half_traits::vlen ) };
	else
	    return type { lo_half_traits::setlane( a.a, s, l ), a.b };
    }

    static lo_half_type lower_half( type a ) { return a.a; }
    static hi_half_type upper_half( type a ) { return a.b; }

    static bool is_all_false( type a ) {
	return lo_half_traits::is_all_false( lower_half( a ) )
	    && hi_half_traits::is_all_false( upper_half( a ) );
    }

    static mask_type asmask( vmask_type a ) {
	return mask_traits::set_pair(
	    lo_half_traits::asmask( lower_half( a ) ),
	    hi_half_traits::asmask( upper_half( a ) ) );
    }
    static vmask_type asvector( mask_type m ) {
	return type {
	    lo_half_traits::asvector( mask_traits::lower_half( m ) ),
		hi_half_traits::asvector( mask_traits::upper_half( m ) ) };
    }

    static uint32_t find_first( type v ) {
	uint32_t pos = lo_half_traits::find_first( v.a );
	if( pos < lo_half_traits::vlen )
	    return pos;
	pos = hi_half_traits::find_first( v.b );
	return pos + vlen;
    }
    static uint32_t find_first( type v, vmask_type m ) {
	uint32_t pos = lo_half_traits::find_first(
	    v.a, mask_traits::lower_half( m ) );
	if( pos < lo_half_traits::vlen )
	    return pos;
	pos = hi_half_traits::find_first( v.b, mask_traits::upper_half( m ) );
	return pos + vlen;
    }

    template<typename MaskTy>
    static type blendm( MaskTy m, type a, type b ) {
	// Not sure why this variant exists. Do whatever blend does.
	return blend( m, a, b );
    }
    static type bitblend( type m, type a, type b ) {
	return type {
	    lo_half_traits::bitblend( m.a, a.a, b.a ),
		hi_half_traits::bitblend( m.b, a.b, b.b ) };
    }
    static type blend( vmask_type m, type a, type b ) {
	return type {
	    lo_half_traits::blend( m.a, a.a, b.a ),
		hi_half_traits::blend( m.b, a.b, b.b ) };
    }
    template<typename MaskTy>
    static type blend( MaskTy m, type a, type b ) {
	using tr = vector_type_traits<
	    typename int_type_of_size<sizeof(MaskTy)/vlen>::type,
	    sizeof(MaskTy)>;
	return type {
	    lo_half_traits::blend( tr::lower_half( m ), a.a, b.a ),
		hi_half_traits::blend( tr::upper_half( m ), a.b, b.b ) };
    }
    static type blend( mask_type m, type a, type b ) {
	return type {
	    lo_half_traits::blend( mask_traits::lower_half( m ), a.a, b.a ),
		hi_half_traits::blend( mask_traits::upper_half( m ), a.b, b.b ) };
    }
    static type iforz( vmask_type m, type a ) {
	return type {
	    lo_half_traits::iforz( m.a, a.a ),
		hi_half_traits::iforz( m.b, a.b ) };
    }
    static type iforz( mask_type m, type a ) {
	return type {
	    lo_half_traits::iforz( mask_traits::lower_half( m.a ), a.a ),
	    hi_half_traits::iforz( mask_traits::lower_half( m.b ), a.b ) };
    }

    static constexpr bool has_ternary =
	lo_half_traits::has_ternary && hi_half_traits::has_ternary;

    template<unsigned char imm8>
    static type ternary( type a, type b, type c ) {
	return type {
	    lo_half_traits::template ternary<imm8>( a.a, b.a, c.a ),
	    hi_half_traits::template ternary<imm8>( a.b, b.b, c.b ) };
    }
    
    static itype castint( type a ) {
	return itype { lo_half_traits::castint( a.a ),
		hi_half_traits::castint( a.b ) };
    }
    static auto castfp( type a ) {
	auto fpa = lo_half_traits::castfp( a.a );
	auto fpb = hi_half_traits::castfp( a.b );
	return vpair<decltype(fpa),decltype(fpb)>{ fpa, fpb };
    }

    static vmask_type cmpne( type l, type r, mt_vmask ) {
	return vmask_type { lo_half_traits::cmpne( l.a, r.a, mt_vmask() ),
		hi_half_traits::cmpne( l.b, r.b, mt_vmask() ) };
    }
    static vmask_type cmpeq( type l, type r, mt_vmask ) {
	return vmask_type { lo_half_traits::cmpeq( l.a, r.a, mt_vmask() ),
		hi_half_traits::cmpeq( l.b, r.b, mt_vmask() ) };
    }
    static vmask_type cmpge( type l, type r, mt_vmask ) {
	return vmask_type { lo_half_traits::cmpge( l.a, r.a, mt_vmask() ),
		hi_half_traits::cmpge( l.b, r.b, mt_vmask() ) };
    }
    static vmask_type cmpgt( type l, type r, mt_vmask ) {
	return vmask_type { lo_half_traits::cmpgt( l.a, r.a, mt_vmask() ),
		hi_half_traits::cmpgt( l.b, r.b, mt_vmask() ) };
    }
    static vmask_type cmplt( type l, type r, mt_vmask ) {
	return vmask_type { lo_half_traits::cmplt( l.a, r.a, mt_vmask() ),
		hi_half_traits::cmplt( l.b, r.b, mt_vmask() ) };
    }
    static vmask_type cmple( type l, type r, mt_vmask ) {
	return vmask_type { lo_half_traits::cmple( l.a, r.a, mt_vmask() ),
		hi_half_traits::cmple( l.b, r.b, mt_vmask() ) };
    }
    static mask_type cmpne( type l, type r, mt_mask ) {
	return combine_mask<lo_half_traits::vlen,hi_half_traits::vlen>(
	    lo_half_traits::cmpne( l.a, r.a, mt_mask() ),
	    hi_half_traits::cmpne( l.b, r.b, mt_mask() ) );
    }
    static mask_type cmpeq( type l, type r, mt_mask ) {
	return combine_mask<lo_half_traits::vlen,hi_half_traits::vlen>(
	    lo_half_traits::cmpeq( l.a, r.a, mt_mask() ),
	    hi_half_traits::cmpeq( l.b, r.b, mt_mask() ) );
    }
    static mask_type cmpge( type l, type r, mt_mask ) {
	return combine_mask<lo_half_traits::vlen,hi_half_traits::vlen>(
	    lo_half_traits::cmpge( l.a, r.a, mt_mask() ),
	    hi_half_traits::cmpge( l.b, r.b, mt_mask() ) );
    }
    static mask_type cmpgt( type l, type r, mt_mask ) {
	return combine_mask<lo_half_traits::vlen,hi_half_traits::vlen>(
	    lo_half_traits::cmpgt( l.a, r.a, mt_mask() ),
	    hi_half_traits::cmpgt( l.b, r.b, mt_mask() ) );
    }
    static mask_type cmplt( type l, type r, mt_mask ) {
	return combine_mask<lo_half_traits::vlen,hi_half_traits::vlen>(
	    lo_half_traits::cmplt( l.a, r.a, mt_mask() ),
	    hi_half_traits::cmplt( l.b, r.b, mt_mask() ) );
    }
    static mask_type cmple( type l, type r, mt_mask ) {
	return combine_mask<lo_half_traits::vlen,hi_half_traits::vlen>(
	    lo_half_traits::cmple( l.a, r.a, mt_mask() ),
	    hi_half_traits::cmple( l.b, r.b, mt_mask() ) );
    }

    static bool cmpne( type l, type r, mt_bool ) { // any lane not equal
	return lo_half_traits::cmpne( l.a, r.a, mt_bool() )
	    || hi_half_traits::cmpne( l.b, r.b, mt_bool() );
    }
    static bool cmpeq( type l, type r, mt_bool ) { // all lanes equal
	return lo_half_traits::cmpeq( l.a, r.a, mt_bool() )
	    && hi_half_traits::cmpeq( l.b, r.b, mt_bool() );
    }
#if 0
    // Questionable semantics...
    static bool cmpgt( type l, type r, mt_bool ) {
	return lo_half_traits::cmpgt( l.a, r.a, mt_bool() )
	    && hi_half_traits::cmpgt( l.b, r.b, mt_bool() );
    }
    static bool cmpge( type l, type r, mt_bool ) {
	return lo_half_traits::cmpge( l.a, r.a, mt_bool() )
	    && hi_half_traits::cmpge( l.b, r.b, mt_bool() );
    }
    static bool cmplt( type l, type r, mt_bool ) {
	return lo_half_traits::cmplt( l.a, r.a, mt_bool() )
	    && hi_half_traits::cmplt( l.b, r.b, mt_bool() );
    }
    static bool cmple( type l, type r, mt_bool ) {
	return lo_half_traits::cmple( l.a, r.a, mt_bool() )
	    && hi_half_traits::cmple( l.b, r.b, mt_bool() );
    }
#endif

    static type bitwise_and( type l, type r ) {
	return type { lo_half_traits::bitwise_and( l.a, r.a ),
		hi_half_traits::bitwise_and( l.b, r.b ) };
    }
    static type bitwise_andnot( type l, type r ) {
	return type { lo_half_traits::bitwise_andnot( l.a, r.a ),
		hi_half_traits::bitwise_andnot( l.b, r.b ) };
    }
    static type bitwise_or( type l, type r ) {
	return type { lo_half_traits::bitwise_or( l.a, r.a ),
		hi_half_traits::bitwise_or( l.b, r.b ) };
    }
    static type bitwise_xor( type l, type r ) {
	return type { lo_half_traits::bitwise_xor( l.a, r.a ),
		hi_half_traits::bitwise_xor( l.b, r.b ) };
    }
    static type bitwise_invert( type l ) {
	return type { lo_half_traits::bitwise_invert( l.a ),
		hi_half_traits::bitwise_invert( l.b ) };
    }
    static type logical_and( type l, type r ) {
	return type { lo_half_traits::logical_and( l.a, r.a ),
		hi_half_traits::logical_and( l.b, r.b ) };
    }
    static type logical_andnot( type l, type r ) {
	return type { lo_half_traits::logical_andnot( l.a, r.a ),
		hi_half_traits::logical_andnot( l.b, r.b ) };
    }
    static type logical_or( type l, type r ) {
	return type { lo_half_traits::logical_or( l.a, r.a ),
		hi_half_traits::logical_or( l.b, r.b ) };
    }
    static type logical_invert( type a ) {
	return type { lo_half_traits::logical_invert( a.a ),
		hi_half_traits::logical_invert( a.b ) };
    }
    static type min( type l, type r ) {
	return type { lo_half_traits::min( l.a, r.a ),
		hi_half_traits::min( l.b, r.b ) };
    }
    static type max( type l, type r ) {
	return type { lo_half_traits::max( l.a, r.a ),
		hi_half_traits::max( l.b, r.b ) };
    }
    static GG_INLINE inline type add( type s, vmask_type m, type a, type b ) {
	return type { lo_half_traits::add( s.a, m.a, a.a, b.a ),
		hi_half_traits::add( s.b, m.b, a.b, b.b ) };
    }
    template<typename Ty>
    static GG_INLINE inline
    std::enable_if_t<sizeof(Ty)==sizeof(type)/2,type>
    add( type s, Ty m, type a, type b ) {
	using wt = vector_type_traits_vl<int_type_of_size_t<W/2>,vlen>;
	return type { lo_half_traits::add( s.a, wt::lower_half(m), a.a, b.a ),
		hi_half_traits::add( s.b, wt::upper_half(m), a.b, b.b ) };
    }
    static GG_INLINE inline type add( type s, mask_type m, type a, type b ) {
	return type {
	    lo_half_traits::add( s.a, mask_traits::lower_half( m ), a.a, b.a ),
	    hi_half_traits::add( s.b, mask_traits::upper_half( m ), a.b, b.b ) };
    }
/*
    static type add( type s, hitype m, type a, type b ) {
	return type {
	    half_traits::add( s.a, hitraits::lower_half( m ), a.a, b.a ),
		half_traits::add( s.a, hitraits::upper_half( m ), a.b, b.b ) };
    }
*/
    static type add( type l, type r ) {
	return type { lo_half_traits::add( l.a, r.a ),
		hi_half_traits::add( l.b, r.b ) };
    }
    static type sub( type l, type r ) {
	return type { lo_half_traits::sub( l.a, r.a ),
		hi_half_traits::sub( l.b, r.b ) };
    }
    static type mul( type l, type r ) {
	return type { lo_half_traits::mul( l.a, r.a ),
		hi_half_traits::mul( l.b, r.b ) };
    }
    static type div( type l, type r ) {
	return type { lo_half_traits::div( l.a, r.a ),
		hi_half_traits::div( l.b, r.b ) };
    }

    static type abs( type a ) {
	return type { lo_half_traits::abs( a.a ), hi_half_traits::abs( a.b ) };
    }

    static member_type reduce_bitwiseor( type a ) {
	return lo_half_traits::reduce_bitwiseor( a.a )
	    | hi_half_traits::reduce_bitwiseor( a.b );
    }
    static member_type reduce_bitwiseor( type a, vmask_type m ) {
	member_type ma = lo_half_traits::reduce_bitwiseor( a.a, m.a );
	member_type mb = hi_half_traits::reduce_bitwiseor( a.b, m.b );
	if( lo_half_int_traits::is_zero( m.a ) )
	    return hi_half_int_traits::is_zero( m.b ) ? ~member_type(0) : mb;
	if( hi_half_int_traits::is_zero( m.b ) )
	    return ma;
	return std::min( ma, mb );
    }
    static member_type reduce_logicalor( type a ) {
	return lo_half_traits::reduce_logicalor( a.a )
	    | hi_half_traits::reduce_logicalor( a.b );
    }
    static member_type reduce_setif( type a, vmask_type m ) {
	member_type ma = lo_half_traits::reduce_setif( a.a, m.a );
	member_type mb = hi_half_traits::reduce_setif( a.b, m.b );
	if( lo_half_int_traits::is_zero( m.a ) )
	    return hi_half_int_traits::is_zero( m.b ) ? ~member_type(0) : mb;
	if( hi_half_int_traits::is_zero( m.b ) )
	    return ma;
	return ma;
    }
    static member_type reduce_min( type a ) {
	static_assert( lo_half_traits::vlen == hi_half_traits::vlen );
	return lo_half_traits::reduce_min( lo_half_traits::min( a.a, a.b ) );
    }
    static member_type reduce_min( type a, vmask_type m ) {
	member_type ma = lo_half_traits::reduce_min( a.a, m.a );
	member_type mb = hi_half_traits::reduce_min( a.b, m.b );
	if( lo_half_int_traits::is_zero( m.a ) )
	    return hi_half_int_traits::is_zero( m.b ) ? ~member_type(0) : mb;
	if( hi_half_int_traits::is_zero( m.b ) )
	    return ma;
	return std::min( ma, mb );
    }
    static member_type reduce_max( type a ) {
	static_assert( lo_half_traits::vlen == hi_half_traits::vlen );
	return lo_half_traits::reduce_max( lo_half_traits::max( a.a, a.b ) );
    }
    static member_type reduce_max( type a, vmask_type m ) {
	member_type ma = lo_half_traits::reduce_max( a.a, m.a );
	member_type mb = hi_half_traits::reduce_max( a.b, m.b );
	if( lo_half_int_traits::is_zero( m.a ) )
	    return hi_half_int_traits::is_zero( m.b ) ? member_type(0) : mb;
	if( hi_half_int_traits::is_zero( m.b ) )
	    return ma;
	return std::max( ma, mb );
    }
    static member_type reduce_add( type a ) {
	return lo_half_traits::reduce_add( a.a )
	    + hi_half_traits::reduce_add( a.b );
    }

    template<typename ShiftTy>
    static std::enable_if_t<(sizeof(ShiftTy)!=vlen),type>
    sllv( type a, ShiftTy b ) {
	using ht = vector_type_traits<int_type_of_size_t<sizeof(ShiftTy)/vlen>,sizeof(ShiftTy)>;
	return type {
	    lo_half_traits::sllv( a.a, ht::lower_half( b ) ),
		hi_half_traits::sllv( a.b, ht::upper_half( b ) ) };
    }
    template<typename ShiftTy>
    static std::enable_if_t<(sizeof(ShiftTy)!=vlen),type>
    srlv( type a, ShiftTy b ) {
	using ht = vector_type_traits<int_type_of_size_t<sizeof(ShiftTy)/vlen>,sizeof(ShiftTy)>;
	return type {
	    lo_half_traits::srlv( a.a, ht::lower_half( b ) ),
		hi_half_traits::srlv( a.b, ht::upper_half( b ) ) };
    }
    static type sll( type a, long b ) {
	return type {
	    lo_half_traits::sll( a.a, b ), hi_half_traits::sll( a.b, b ) };
    }
    static type srl( type a, long b ) {
	return type {
	    lo_half_traits::srl( a.a, b ), hi_half_traits::srl( a.b, b ) };
    }
    static type slli( type a, int_type b ) {
	return type {
	    lo_half_traits::slli( a.a, b ), hi_half_traits::slli( a.b, b ) };
    }
    static type srli( type a, int_type b ) {
	return type {
	    lo_half_traits::srli( a.a, b ), hi_half_traits::srli( a.b, b ) };
    }
    static type srai( type a, int_type b ) {
	return type {
	    lo_half_traits::srai( a.a, b ), hi_half_traits::srai( a.b, b ) };
    }
    static type srav( type a, type b ) {
	return type {
	    lo_half_traits::srav( a.a, b.a ), hi_half_traits::srav( a.b, b.b ) };
    }

    template<typename ReturnTy>
    static auto tzcnt( type a ) {
	using traits = vector_type_traits<ReturnTy,sizeof(ReturnTy)*vlen>;
	return traits::set_pair(
	    hi_half_traits::template tzcnt<ReturnTy>( upper_half( a ) ),
	    lo_half_traits::template tzcnt<ReturnTy>( lower_half( a ) ) );
    }
    
    template<typename ReturnTy>
    static auto lzcnt( type a ) {
	using traits = vector_type_traits<ReturnTy,sizeof(ReturnTy)*vlen>;
	return traits::set_pair(
	    hi_half_traits::template lzcnt<ReturnTy>( upper_half( a ) ),
	    lo_half_traits::template lzcnt<ReturnTy>( lower_half( a ) ) );
    }
    static type popcnt( type a ) {
	return set_pair(
	    hi_half_traits::popcnt( upper_half( a ) ),
	    lo_half_traits::popcnt( lower_half( a ) ) );
    }
    
    static type loadu( const member_type * addr ) {
	return type { lo_half_traits::loadu( addr ),
		hi_half_traits::loadu( addr+lo_half_traits::vlen ) };
    }
    static type load( const member_type * addr ) {
	return type { lo_half_traits::load( addr ),
		hi_half_traits::load( addr+lo_half_traits::vlen ) };
    }
    static void storeu( member_type * addr, type val ) {
	lo_half_traits::storeu( addr, val.a );
	hi_half_traits::storeu( addr+lo_half_traits::vlen, val.b );
    }
    static void store( member_type * addr, type val ) {
	lo_half_traits::store( addr, val.a );
	hi_half_traits::store( addr+lo_half_traits::vlen, val.b );
    }
    static type ntload( const member_type * addr ) {
	return type { lo_half_traits::ntload( addr ),
		hi_half_traits::ntload( addr+lo_half_traits::vlen ) };
    }
    static void ntstore( member_type * addr, type val ) {
	lo_half_traits::ntstore( addr, val.a );
	hi_half_traits::ntstore( addr+lo_half_traits::vlen, val.b );
    }
    template<typename IdxT>
    static type gather( const member_type * addr, IdxT idx ) {
	using itraits = vector_type_traits<
	    typename int_type_of_size<sizeof(IdxT)/vlen>::type, sizeof(IdxT)>;
	return type {
	    lo_half_traits::gather( addr, itraits::lower_half( idx ) ),
		hi_half_traits::gather( addr, itraits::upper_half( idx ) ) };
    }
    template<typename IdxT>
    static type gather( const member_type * addr, IdxT idx, vmask_type mask ) {
	using itraits = vector_type_traits<
	    typename int_type_of_size<sizeof(IdxT)/vlen>::type, sizeof(IdxT)>;
	return type {
	    lo_half_traits::gather( addr, itraits::lower_half( idx ),
				    int_traits::lower_half( mask ) ),
		hi_half_traits::gather( addr, itraits::upper_half( idx ),
					int_traits::upper_half( mask ) ) };
    }
    template<typename IdxT>
    static typename std::enable_if_t<sizeof(IdxT) != sizeof(vmask_type), type>
    gather( const member_type * addr, IdxT idx, IdxT mask ) {
	using itraits = vector_type_traits<
	    typename int_type_of_size<sizeof(IdxT)/vlen>::type, sizeof(IdxT)>;
	return type {
	    lo_half_traits::gather( addr, itraits::lower_half( idx ),
				    itraits::lower_half( mask ) ),
		hi_half_traits::gather( addr, itraits::upper_half( idx ),
					itraits::upper_half( mask ) ) };
    }
    template<unsigned short Scale, typename IdxT>
    static type gather_w( const member_type * addr, IdxT idx ) {
	using itraits = vector_type_traits<
	    typename int_type_of_size<sizeof(IdxT)/vlen>::type, sizeof(IdxT)>;
	return type {
	    lo_half_traits::template gather_w<Scale>( addr, itraits::lower_half( idx ) ),
		hi_half_traits::template gather_w<Scale>( addr, itraits::upper_half( idx ) ) };
    }
    template<unsigned short Scale, typename IdxT>
    static type
    gather_w( const member_type * addr, IdxT idx, vmask_type mask ) {
	using itraits = vector_type_traits<
	    typename int_type_of_size<sizeof(IdxT)/vlen>::type, sizeof(IdxT)>;
	return type {
	    lo_half_traits::template gather_w<Scale>( addr, itraits::lower_half( idx ),
				    int_traits::lower_half( mask ) ),
		hi_half_traits::template gather_w<Scale>( addr, itraits::upper_half( idx ),
					int_traits::upper_half( mask ) ) };
    }
    template<unsigned short Scale, typename IdxT>
    static typename std::enable_if_t<sizeof(IdxT) != sizeof(vmask_type), type>
    gather_w( const member_type * addr, IdxT idx, IdxT mask ) {
	using itraits = vector_type_traits<
	    typename int_type_of_size<sizeof(IdxT)/vlen>::type, sizeof(IdxT)>;
	return type {
	    lo_half_traits::template gather_w<Scale>( addr, itraits::lower_half( idx ),
				    itraits::lower_half( mask ) ),
		hi_half_traits::template gather_w<Scale>( addr, itraits::upper_half( idx ),
					itraits::upper_half( mask ) ) };
    }
/*
    static type gather( const member_type * addr, itype idx, hitype mask ) {
	return type {
	    half_traits::gather( addr, int_traits::lower_half( idx ),
				 hitraits::lower_half( mask ) ),
		half_traits::gather( addr, int_traits::upper_half( idx ),
				     hitraits::upper_half( mask ) ) };
    }
*/
    template<typename IdxT>
    static type gather( const member_type * addr, IdxT idx, mask_type mask ) {
	using itraits = vector_type_traits<
	    typename int_type_of_size<sizeof(IdxT)/vlen>::type, sizeof(IdxT)>;
	return type {
	    lo_half_traits::gather( addr, itraits::lower_half( idx ),
				    mask_traits::lower_half( mask ) ),
		hi_half_traits::gather( addr, itraits::upper_half( idx ),
					mask_traits::upper_half( mask ) ) };
    }
    template<unsigned short Scale, typename IdxT>
    static type gather_w( const member_type * addr, IdxT idx, mask_type mask ) {
	using itraits = vector_type_traits<
	    typename int_type_of_size<sizeof(IdxT)/vlen>::type, sizeof(IdxT)>;
	return type {
	    lo_half_traits::template gather_w<Scale>(
		addr, itraits::lower_half( idx ),
		mask_traits::lower_half( mask ) ),
		hi_half_traits::template gather_w<Scale>(
		    addr, itraits::upper_half( idx ),
		    mask_traits::upper_half( mask ) ) };
    }

/*
    static type gather( const member_type * addr, hitype idx, mask_type mask ) {
	return type {
	    half_traits::gather( addr, hitraits::lower_half( idx ),
				 mtraits::lower_half( mask ) ),
		half_traits::gather( addr, hitraits::upper_half( idx ),
				     mtraits::upper_half( mask ) ) };
    }
*/
    template<typename IdxT>
    static void scatter( member_type * addr, IdxT idx, type val ) {
	using itraits = vector_type_traits<
	    typename int_type_of_size<sizeof(IdxT)/vlen>::type, sizeof(IdxT)>;
	lo_half_traits::scatter( addr, itraits::lower_half( idx ), val.a );
	hi_half_traits::scatter( addr, itraits::upper_half( idx ), val.b );
    }
    template<typename IdxT>
    static void scatter( member_type * addr, IdxT idx, type val, mask_type mask ) {
	using itraits = vector_type_traits<
	    typename int_type_of_size<sizeof(IdxT)/vlen>::type, sizeof(IdxT)>;
	lo_half_traits::scatter( addr, itraits::lower_half( idx ), val.a,
				 mask_traits::lower_half( mask ) );
	hi_half_traits::scatter( addr, itraits::upper_half( idx ), val.b,
				 mask_traits::upper_half( mask ) );
    }
    template<typename IdxT>
    static void scatter( member_type * addr, IdxT idx, type val, vmask_type mask ) {
	using itraits = vector_type_traits<
	    typename int_type_of_size<sizeof(IdxT)/vlen>::type, sizeof(IdxT)>;
	lo_half_traits::scatter( addr, itraits::lower_half( idx ), val.a,
				 int_traits::lower_half( mask ) );
	hi_half_traits::scatter( addr, itraits::upper_half( idx ), val.b,
				 int_traits::upper_half( mask ) );
    }
    template<typename IdxT, typename MaskT>
    static void scatter( member_type * addr, IdxT idx, type val, MaskT mask ) {
	using itraits = vector_type_traits<
	    typename int_type_of_size<sizeof(IdxT)/vlen>::type, sizeof(IdxT)>;
	using mtraits = vector_type_traits<
	    typename int_type_of_size<sizeof(MaskT)/vlen>::type, sizeof(MaskT)>;
	lo_half_traits::scatter( addr, itraits::lower_half( idx ), val.a,
				 mtraits::lower_half( mask ) );
	hi_half_traits::scatter( addr, itraits::upper_half( idx ), val.b,
				 mtraits::upper_half( mask ) );
    }

    template<unsigned degree_bits, unsigned degree_shift>
    class vtrec_extract_degree_bits {
	decltype(lo_half_traits::template create_extractor<degree_bits,degree_shift>()) lo_extract;
	decltype(hi_half_traits::template create_extractor<degree_bits,degree_shift>()) hi_extract;

    public:
	vtrec_extract_degree_bits()
	    : lo_extract(), hi_extract() { }
	    
	auto extract_degree( type v ) const {
	    // Cannot have more meta-data bits encoded than the size of
	    // a vertex ID
	    if constexpr ( lo_half_traits::vlen >= 8*sizeof(VID) ) {
		return lo_extract.extract_degree( v.a );
	    } else {
		// Immediately do a widening cast when receiving the degree
		using bitmask_ty = typename mask_type_traits<vlen>::type;
		bitmask_ty lo_bits = lo_extract.extract_degree( v.a );
		bitmask_ty hi_bits = hi_extract.extract_degree( v.b );
		return ( hi_bits << lo_half_traits::vlen ) | lo_bits;
	    }
	}
	type extract_source( type v ) const {
	    return type {
		lo_extract.extract_source( v.a ),
		    hi_extract.extract_source( v.b ) };
	}
	type get_mask() const {
	    return type { lo_extract.get_mask(), hi_extract.get_mask() };
	}
    };
    template<unsigned degree_bits, unsigned degree_shift>
    static vtrec_extract_degree_bits<degree_bits,degree_shift>
    create_extractor() {
	return vtrec_extract_degree_bits<degree_bits,degree_shift>();
    }
};

template<typename T>
struct is_vt_recursive : public std::false_type { };

template<typename T, unsigned short W, unsigned short nbytes,
	 typename lo_half_traits, typename hi_half_traits>
struct is_vt_recursive<vt_recursive<T,W,nbytes,lo_half_traits,hi_half_traits>>
    : public std::true_type { };

template<typename T>
static constexpr bool is_vt_recursive_v = is_vt_recursive<T>::value;

} // namespace target

#endif //  GRAPTOR_TARGET_VTRECURSIVE_H
