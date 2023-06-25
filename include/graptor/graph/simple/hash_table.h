// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_SIMPLE_HASH_TABLE_H
#define GRAPTOR_GRAPH_SIMPLE_HASH_TABLE_H

#include <vector>
#include <algorithm>
#include <ostream>

#include "graptor/container/bitset.h"
#include "graptor/graph/simple/conditional_iterator.h"

namespace graptor {

namespace graph {
    
template<typename T, typename Hash = std::hash<T>>
class hash_table {
public:
    typedef T type;
    typedef Hash hash_type;
    typedef size_t size_type;
    typedef type & reference;
    typedef const type & const_reference;

    static constexpr type invalid_element = ~type(0);

public:
    explicit hash_table()
	: m_elements( 0 ),
	  m_log_size( 4 ),
	  m_table( new type[16] ),
	  m_hash( 4 ) {
	clear();
    }
    hash_table( hash_table && ) = delete;
    hash_table( const hash_table & ) = delete;
    hash_table & operator = ( const hash_table & ) = delete;

    ~hash_table() {
	delete[] m_table;
    }

    void clear() {
	m_elements = 0;
	std::fill( m_table, m_table+capacity(), invalid_element );
    }

    size_type size() const { return m_elements; }
    size_type capacity() const { return size_type(1) << m_log_size; }
    bool empty() const { return size() == 0; }

    auto begin() const { return m_table; }
    auto end() const { return m_table+capacity(); }

/*
    auto begin() const {
	return graptor::graph::make_conditional_iterator(
	    m_table, [&]( const type * && it ) {
		return it != m_table+m_size && *it != invalid_element;
	    } );
    }
    auto end() const {
	return graptor::graph::make_conditional_iterator(
	    m_table+m_size, [&]( const type * && it ) {
		return it != m_table+m_size && *it != invalid_element;
	    } );
    }
*/

    bool insert( type value ) {
	size_type index = m_hash( value ) & ( capacity() - 1 );
	while( m_table[index] != invalid_element && m_table[index] != value )
	    index = ( index + 1 ) & ( capacity() - 1 );
	if( m_table[index] == value ) {
	    return false;
	} else {
	    if( (m_elements+1) >= ( capacity() >> 1 ) ) {
		// Rehash
		size_type old_log_size = m_log_size + 1;
		type * old_table = new type[size_type(1)<<old_log_size];
		using std::swap;
		swap( old_log_size, m_log_size );
		swap( old_table, m_table );
		clear(); // sets m_elements=0; will be reset when rehashing
		size_type old_size = size_type(1) << old_log_size;
		m_hash.resize( m_log_size );
		for( size_type i=0; i < old_size; ++i )
		    if( old_table[i] != invalid_element )
			insert( old_table[i] );
		delete[] old_table;
		return insert( value );
	    } else {
		++m_elements;
		m_table[index] = value;
		return true;
	    }
	}
    }

    template<typename It>
    void insert( It && I, It && E ) {
	while( I != E )
	    insert( *I++ );
    }

    bool contains( type value ) const {
	size_type index = m_hash( value ) & ( capacity() - 1 );
	while( m_table[index] != invalid_element && m_table[index] != value )
	    index = ( index + 1 ) & ( capacity() - 1 );
	return m_table[index] == value;
    }

    template<typename Fn>
    void for_each( Fn && fn ) const {
	for( size_type i=0; i < capacity(); ++i )
	    if( m_table[i] != invalid_element )
		fn( m_table[i] );
    }

    template<typename It, typename Ot>
    Ot intersect( It I, It E, Ot O ) const {
	while( I != E ) {
	    VID v = *I;
	    if( contains( v ) ) {
		*O++ = v;
	    }
	    ++I;
	}
	return O;
    }

#if 0
    template<typename It>
    size_t intersect_size( It I, It E, size_t exceed ) const {
	size_t sz = 0;
	size_t todo = std::distance( I, E );
	while( I != E ) {
	    if( sz + todo <= exceed )
		return 0;

	    VID v = *I;
	    if( contains( v ) )
		++sz;

	    ++I;
	    --todo;
	}
	return sz;
    }
#endif

    template<typename It>
    size_t intersect_size( It I, It E, size_t exceed ) const {
	size_t d = std::distance( I, E );
#if 0
#if __AVX512F__
	if( d*sizeof(type) >= 512 )
	    return intersect_size_vector<16>( I, E, exceed );
#endif
#if __AVX2__
	if( d*sizeof(type) >= 256 )
	    return intersect_size_vector<8>( I, E, exceed );
#endif
#endif
	return intersect_size_scalar( I, E, exceed );
    }

    template<typename atype, unsigned short VL>
    auto multi_contains( typename vector_type_traits_vl<atype,VL>::type
			 index ) const {
	using tr = vector_type_traits_vl<atype,VL>;
	using vtype = typename tr::type;

	const vtype vinv = tr::setone(); // invalid_element == -1
	const vtype hmask = tr::srli( tr::setone(), tr::B - m_log_size );
	vtype v_ins = tr::setzero();

	size_type s_ins = 0;

	vtype v = index; // Assuming &*(I+1) == (&*I)+1
	vtype h = m_hash.template vectorized<VL>( v );
	vtype vidx = tr::bitwise_and( h, hmask );
	vtype e = tr::gather( m_table, vidx );
	vtype mi = tr::cmpeq( e, vinv, target::mt_vmask() );
	vtype mv = tr::cmpeq( e, v, target::mt_vmask() );
	vtype miv = tr::bitwise_or( mi, mv );
	vtype imiv = tr::bitwise_invert( miv );
	// Some lanes are neither empty nor the requested value and
	// need further probes.
	while( !tr::is_zero( imiv ) ) {
	    // TODO: hash table load is targeted below 50%. Due to the
	    // birthday paradox, there will be frequent collisions still
	    // and it may be useful to make multiple vectorized probes
	    // before resorting to scalar execution. We could even count
	    // active lanes to decide.
	    vidx = tr::add( vidx, tr::setoneval() );
	    vidx = tr::bitwise_and( vidx, hmask );
	    e = tr::gather( m_table, vidx, imiv );
	    mi = tr::cmpeq( e, vinv, target::mt_vmask() );
	    vtype mv2 = tr::cmpeq( e, v, target::mt_vmask() );
	    mv = tr::blend( imiv, mv, mv2 );
	    miv = tr::bitwise_or( mi, mv2 );
	    imiv = tr::bitwise_andnot( miv, imiv );
	}
	return mv;
    }

private:
    // Identify the table cell where we can decide on presence of value
    size_type locate( type value ) const {
	return locate_scalar( value );
    }
    size_type locate_scalar( type value ) const {
	size_type index = m_hash( value ) & ( capacity() - 1 );
	while( m_table[index] != invalid_element && m_table[index] != value )
	    index = ( index + 1 ) & ( capacity() - 1 );
	return index;
    }
    template<unsigned short VL>
    size_type locate_vector( type value ) const {
	using tr = vector_type_traits_vl<type,VL>;
	using vtype = typename tr::type;
	size_type index = m_hash( value ) & ( capacity() - 1 );
	size_type vindex = index & ~size_type( VL - 1 ); // multiple of VL

	vtype vinv = tr::setone(); // invalid_element == -1
	vtype val = tr::set1( value );
	size_type VLmsk = size_type( VL - 1 );
	vtype msk = tr::asvector( ( index & VLmsk ) ^ VLmsk );

	do {
	    vtype v = tr::loadu( m_table, vindex );
	    vtype mi = tr::cmpeq( v, vinv, target::mt_vmask() );
	    vtype mv = tr::cmpeq( v, val, target::mt_vmask() );
	    vtype mo = tr::bitwise_or( mi, mv );
	    vtype ma = tr::bitwise_andnot( mo, msk );
	    if( tr::is_zero( ma ) ) {
		msk = tr::setone();
		vindex = ( vindex + VL ) & ( capacity() - 1 );
	    } else {
		// Now there is a lane in v that is suitable. Identify it.
		unsigned lane = _tzcnt_u32( tr::asmask( ma ) );
		return vindex + lane;
	    }
	} while( true );
    }
    template<unsigned short VL>
    size_type
    intersect_size_vector( const type * I, const type * E, size_t exceed )
	const {
	using tr = vector_type_traits_vl<type,VL>;
	using vtype = typename tr::type;

	const vtype vinv = tr::setone(); // invalid_element == -1
	const vtype hmask = tr::srli( tr::setone(), tr::B - m_log_size );
	vtype v_ins = tr::setzero();

	size_type s_ins = 0;

	for( ; I+VL <= E; I += VL ) {
	    vtype v = tr::loadu( I ); // Assuming &*(I+1) == (&*I)+1
	    vtype h = m_hash.template vectorized<VL>( v );
	    vtype vidx = tr::bitwise_and( h, hmask );
	    vtype e = tr::gather( m_table, vidx );
	    vtype mi = tr::cmpeq( e, vinv, target::mt_vmask() );
	    vtype mv = tr::cmpeq( e, v, target::mt_vmask() );
	    vtype miv = tr::bitwise_or( mi, mv );
	    vtype imiv = tr::bitwise_invert( miv );
	    auto m = tr::asmask( imiv );
	    while( m != 0 ) {
		// TODO: hash table load is targeted below 50%. Due to the
		// birthday paradox, there will be frequent collisions still
		// and it may be useful to make multiple vectorized probes
		// before resorting to scalar execution. We could even count
		// active lanes to decide.
		// Some lanes are neither empty nor the requested value and
		// need further probes.
/*
		bitset<VL> probes( m );
		for( auto PI=probes.begin(), PE=probes.end(); PI != PE; ++PI )
		    s_ins += contains_2nd( *(I + *PI) ) ? 1 : 0;
*/
		vidx = tr::add( vidx, tr::setoneval() );
		vidx = tr::bitwise_and( vidx, hmask );
		e = tr::gather( m_table, vidx, imiv );
		mi = tr::cmpeq( e, vinv, target::mt_vmask() );
		vtype mv2 = tr::cmpeq( e, v, target::mt_vmask() );
		mv = tr::blend( imiv, mv, mv2 );
		miv = tr::bitwise_or( mi, mv2 );
		imiv = tr::bitwise_andnot( miv, imiv );
		m = tr::asmask( imiv );
	    }
	    v_ins = tr::add( v_ins, tr::srli( mv, tr::B-1 ) ); // +1 if found
	    if( s_ins + std::distance( I, E ) < exceed + VL )
		return 0;
	}
	s_ins += tr::reduce_add( v_ins );
	if( I != E )
	    s_ins += intersect_size_scalar( I, E, exceed - s_ins );
	return s_ins;
    }
    bool contains_2nd( type value ) const {
	// Skip first element, already probed
	size_type index = m_hash( value ) & ( capacity() - 1 );
	index = ( index + 1 ) & ( capacity() - 1 ); // skip
	while( m_table[index] != invalid_element && m_table[index] != value )
	    index = ( index + 1 ) & ( capacity() - 1 );
	return m_table[index] == value;
    }
    template<typename It>
    size_t intersect_size_scalar( It I, It E, size_t exceed ) const {
	using ty = std::make_signed_t<size_t>;
	ty d = std::distance( I, E );
	ty x = exceed;
	ty options = x - d;
	if( options >= 0 )
	    return 0;

	while( I != E ) {
	    VID v = *I;
	    if( !contains( v ) ) [[likely]] {
		if( ++options >= 0 ) [[unlikely]]
		    return 0;
	    }

	    ++I;
	}
	return x - options;
    }
	    
private:
    size_type m_elements;
    size_type m_log_size;
    type * m_table;
    hash_type m_hash;
};


/*
template<typename VID>
ostream & operator << ( ostream & os, const hash_table<VID> & s ) {
    os << "{ #" << s.size() << ": ";
    for( auto I=s.begin(), E=s.end(); I != E; ++I )
	os << ' ' << *I;
    os << " }";
    return os;
}
*/

} // namespace graph

} // namespace graptor

#endif // GRAPTOR_GRAPH_SIMPLE_HASH_TABLE_H
