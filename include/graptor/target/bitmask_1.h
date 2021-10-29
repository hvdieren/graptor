// -*- c++ -*-
#ifndef GRAPTOR_TARGET_BITMASK1_H
#define GRAPTOR_TARGET_BITMASK1_H

namespace target {

/***********************************************************************
 * Bitmask traits with 1 lane
 ***********************************************************************/
template<>
struct mask_type_traits<1> {
    static constexpr unsigned short vlen = 1;
    using type = unsigned char;
    using member_type = bool;
    using pointer_type = unsigned char;

    static type setzero() { return 0; }
    static type setone() { return 1; }
    static type set1( bool v ) { return v ? 1 : 0; }

    static member_type lane( type v, unsigned short l ) { return v & 1; }
    static member_type lane0( type v ) { return v & 1; }

    static bool is_all_false( type v ) { return lane( v, 0 ) == 0; }

    template<typename T>
    static auto asvector( type m ) {
	if constexpr ( std::is_same<T,bool>::value )
			 return m != (type)0;
	else
	    return m != (type)0 ? ~(T)0 : (T)0;
    }

    static bool cmpne( type l, type r, target::mt_bool ) { return l != r; }
    static bool cmpeq( type l, type r, target::mt_bool ) { return l == r; }

    static type blendm( type sel, type l, type r ) { return sel ? r : l; }

    static type logical_and( type l, type r ) { return l & r; }
    static type logical_or( type l, type r ) { return l | r; }
    static type logical_invert( type a ) { return !a; }

    static member_type loads( const pointer_type * addr, unsigned int idx ) {
	return ( addr[idx/8] >> (idx % 8) ) & 1;
    }
    static type load( const pointer_type * addr, unsigned int idx ) {
	return loads( addr, idx );
    }
    static type loadu( const pointer_type * addr, unsigned int idx ) {
	return loads( addr, idx );
    }
    static void store( pointer_type * addr, unsigned int idx, type val ) {
	storeu( addr, idx, val );
    }
    static void storeu( pointer_type * addr, unsigned int idx, type val ) {
	unsigned short l = idx % 8;
	type m = type(1) << l;
	type vl = addr[idx/8] & ~m;
	vl |= val << l;
	addr[idx/8] = vl;
    }
    
    template<typename vindex_type>
    static type gather( pointer_type * addr, vindex_type idx ) {
	using index_type
	    = typename int_type_of_size<sizeof(vindex_type)/vlen>::type;
	// using itraits = vector_type_traits_vl<index_type,vlen>;
	// index_type lidx = itraits::lane( idx, 0 );
	index_type lidx = (index_type)idx;
	return loadu( addr, lidx );
    }
#if 0
    template<typename vindex_type>
    static void scatter( type * addr, vindex_type idx, type val ) {
	using index_type
	    = typename int_type_of_size<sizeof(vindex_type)/vlen>::type;
	using itraits = vector_type_traits_vl<index_type,vlen>;
	index_type lidx = itraits::lane( idx, 0 );
	storeu( addr, lidx, val );
    }
#endif
    template<typename vindex_type>
    static void scatter( type * addr, vindex_type idx, type val, type mask ) {
	assert( 0 && "NYI" );
    }

    static type from_int( int v ) { return type(v); }
};

} // namespace target

#endif // GRAPTOR_TARGET_BITMASK1_H
