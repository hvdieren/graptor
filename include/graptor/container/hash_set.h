// -*- c++ -*-
#ifndef GRAPTOR_CONTAINER_HASH_SET_H
#define GRAPTOR_CONTAINER_HASH_SET_H

#include <type_traits>
#include <algorithm>
#include <ostream>

#include "graptor/container/bitset.h"
#include "graptor/container/hash_fn.h"
#include "graptor/container/conditional_iterator.h"

#ifdef LOAD_FACTOR
#define HASH_SET_LOAD_FACTOR LOAD_FACTOR
#else
#define HASH_SET_LOAD_FACTOR 2
#endif

/*!=====================================================================*
 * TODO:
 * + ensure that high-degree vertices are closer to their intended position
 *   than low-degree vertices; or insert high-degree vertices first.
 *   The latter is insufficient due to resizing.
 *======================================================================*/

namespace graptor {

template<typename T, typename Hash = rand_hash<T>>
class hash_set {
public:
    typedef T type;
    typedef Hash hash_type;
    typedef uint32_t size_type;
    typedef type & reference;
    typedef const type & const_reference;

    static constexpr type invalid_element = ~type(0);

public:
    explicit hash_set()
	: m_elements( 0 ),
	  m_log_size( 4 ),
	  m_table( new type[16] ),
	  m_hash( 4 ),
	  m_pre_allocated( false ) {
	clear();
    }
    explicit hash_set( size_t expected_elms )
	: m_elements( 0 ),
	  m_log_size( required_log_size( expected_elms ) ),
	  m_table( new type[1<<m_log_size] ),
	  m_hash( m_log_size ),
	  m_pre_allocated( false ) {
	clear();
    }
    template<typename It>
    explicit hash_set( It begin, It end )
	: hash_set( std::distance( begin, end ) ) {
	for( It i=begin; i != end; ++i )
	    insert( *i );
    }
    explicit hash_set( type * storage, size_type num_elements,
			 size_type log_size )
	: m_elements( num_elements ),
	  m_log_size( log_size ),
	  m_table( storage ),
	  m_hash( log_size ),
	  m_pre_allocated( true ) {
	clear();
    }
    explicit hash_set( type * storage, size_type num_elements,
			 size_type log_size, const hash_type & hash )
	: m_elements( num_elements ),
	  m_log_size( log_size ),
	  m_table( storage ),
	  m_hash( hash ),
	  m_pre_allocated( true ) {
	clear();
    }
    hash_set( hash_set && ) = delete;
    hash_set( const hash_set & ) = delete;
    hash_set & operator = ( const hash_set & ) = delete;

    ~hash_set() {
	if( !m_pre_allocated )
	    delete[] m_table;
    }

    void clear() {
	m_elements = 0;
	std::fill( m_table, m_table+capacity(), invalid_element );
    }

    size_type size() const { return m_elements; }
    size_type capacity() const { return size_type(1) << m_log_size; }
    bool empty() const { return size() == 0; }

    const type * get_table() const { return m_table; }

    auto begin() const { return m_table; }
    auto end() const { return m_table+capacity(); }

    static size_type required_log_size( size_type num_elements ) {
	// Maintain fill factor of 50% at most
	// ilog2( num_elements-1 ) + 1 -> next higher power of two
	// but ensure #elms+1 does not exceed 50%
	// + 1: fill factor not exceeding 50%
	return rt_ilog2( num_elements ) + HASH_SET_LOAD_FACTOR;
    }

    bool insert_prev( type value, type ) {
	return insert( value );
    }

    bool insert( type value ) {
	size_type index = m_hash( value ) & ( capacity() - 1 );
	while( m_table[index] != invalid_element && m_table[index] != value )
	    index = ( index + 1 ) & ( capacity() - 1 );
	if( m_table[index] == value ) {
	    return false;
	} else {
	    if( (m_elements+1) >= ( capacity() >> 1 ) ) {
		assert( !m_pre_allocated
			&& "Cannot resize if not owning the storage" );
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
	if( m_elements == 0 ) // avoid infinite loop
	    return false;
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
	    type v = *I;
	    if( contains( v ) ) {
		*O++ = v;
	    }
	    ++I;
	}
	return O;
    }

    template<typename It>
    size_t intersect_size( It I, It E, size_t exceed ) const {
	return intersect_size_scalar( I, E, exceed );
    }

    template<typename U, unsigned short VL, typename MT>
    std::conditional_t<std::is_same_v<MT,target::mt_mask>,
	typename vector_type_traits_vl<U,VL>::mask_type,
	typename vector_type_traits_vl<U,VL>::vmask_type>
    multi_contains( typename vector_type_traits_vl<U,VL>::type
			 index, MT ) const {
	static_assert( sizeof( U ) == sizeof( type ) );
	using tr = vector_type_traits_vl<type,VL>;
	using vtype = typename tr::type;
#if __AVX512F__
	using mtr = typename tr::mask_traits;
	using mkind = target::mt_mask;
#else
	using mtr = typename tr::vmask_traits;
	using mkind = target::mt_vmask;
#endif
	using mtype = typename mtr::type;

	const vtype ones = tr::setone();
	const vtype vinv = ones; // invalid_element == -1
	const vtype hmask = tr::srli( ones, tr::B - m_log_size );
	vtype v_ins = tr::setzero();

	size_type s_ins = 0;

	vtype v = index; // Assuming &*(I+1) == (&*I)+1
	vtype h = m_hash.template vectorized<VL>( v );
	vtype vidx = tr::bitwise_and( h, hmask );
	vtype e = tr::gather( m_table, vidx );
	mtype imi = tr::cmpne( e, vinv, mkind() );
	mtype imv = tr::cmpne( e, v, mkind() );
	mtype imiv = mtr::logical_and( imi, imv );
	// Some lanes are neither empty nor the requested value and
	// need further probes.
	while( !mtr::is_zero( imiv ) ) {
	    // TODO: hash table load is targeted below 50%. Due to the
	    // birthday paradox, there will be frequent collisions still
	    // and it may be useful to make multiple vectorized probes
	    // before resorting to scalar execution. We could even count
	    // active lanes to decide.
	    vidx = tr::add( vidx, tr::setoneval() );
	    vidx = tr::bitwise_and( vidx, hmask );
	    // set default value for gather to prior e such that it acts
	    // as a blend.
	    e = tr::gather( e, m_table, vidx, imiv );
	    mtype imi = tr::cmpne( e, vinv, mkind() );
	    mtype imv = tr::cmpne( e, v, mkind() );
	    imiv = mtr::logical_and( imi, imv );
	}
	// Return success if found, which equals imv inverted
	// It just takes one cycle to recompute, same as invert. The code
	// below is most compact to achieve the correct return type.
/*
	{
	    mtype r = tr::cmpeq( e, v, mkind() );
	    for( unsigned l=0; l < VL; ++l ) {
		bool cs = contains( tr::lane( index, l ) );
		bool cv = tr::lane( r, l ) != 0;
		assert( cs == cv );
	    }
	}
*/
	return tr::cmpeq( e, v, MT() );
    }

    template<unsigned short VL>
    bool
    wide_contains( type value ) const {
	using tr = vector_type_traits_vl<type,VL>;
	using vtype = typename tr::type;
#if __AVX512F__
	using mtr = typename tr::mask_traits;
	using mkind = target::mt_mask;
#else
	using mtr = typename tr::vmask_traits;
	using mkind = target::mt_vmask;
#endif
	using mtype = typename mtr::type;

	if( capacity() < VL )
	    return contains( value );

	size_type index = m_hash( value ) & ( capacity() - 1 );
	vtype search = tr::set1( value );
	vtype vabsent = tr::setone();

	// How to deal with reaching over end of array?
	size_type lane = index & size_type( VL - 1 );
	index &= ~size_type( VL-1 ); // round down; array is power of 2
	vtype elm = tr::loadu( m_table+index );
	size_type above = ( size_type(1) << VL ) - ( size_type(1) << lane );
	mtype msk = mtr::from_int( above );
	mtype eq = tr::cmpeq( msk, elm, search, mkind() );
	if( !mtr::is_zero( eq ) ) {
	    return true;
	}
	mtype abs = tr::cmpeq( msk, elm, vabsent, mkind() );
	if( !mtr::is_zero( abs ) ) {
	    return false;
	}

	index = ( index + VL ) & ( capacity() - 1 );

	while( true ) {
	    vtype elm = tr::loadu( m_table+index );
	    mtype eq = tr::cmpeq( elm, search, mkind() );
	    if( !mtr::is_zero( eq ) ) {
		return true;
	    }
	    mtype abs = tr::cmpeq( elm, vabsent, mkind() );
	    if( !mtr::is_zero( abs ) ) {
		return false;
	    }

	    index = ( index + VL ) & ( capacity() - 1 );
	}
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
	    type v = *I;
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
    bool m_pre_allocated;
};

template<typename T, typename Hash>
ostream & operator << ( ostream & os, const hash_set<T,Hash> & s ) {
    os << "{ #" << s.size() << ": ";
    for( auto I=s.begin(), E=s.end(); I != E; ++I )
	os << ' ' << *I;
    os << " }";
    return os;
}

template<typename HashTable>
struct hash_insert_iterator;

template<typename T, typename Hash>
struct hash_insert_iterator<hash_set<T,Hash>> {
    hash_insert_iterator( hash_set<T,Hash> & table, const T * start )
	: m_table( table ), m_start( start ) { }

    void push_back( const T * t, const T * = nullptr ) {
	m_table.insert( t - m_start );
    }

    // Interface for intersection collectors
    template<bool rhs>
    bool record( const T * l, const T * r, bool ins ) {
	static_assert( rhs == true, "alternate case not considered" );
	if( ins )
	    m_table.insert( l - m_start );
	return true;
    }

    template<bool rhs>
    bool record( const T * l, T value, bool ins ) {
	static_assert( rhs == true, "alternate case not considered" );
	if( ins )
	    m_table.insert( l - m_start );
	return true;
    }

    template<bool rhs>
    void remainder( size_t l, size_t r ) { }

    auto & return_value() const { return *this; }

    bool terminated() const { return false; }
    
private:
    hash_set<T,Hash> & m_table;
    const T * m_start;
};

} // namespace graptor

#endif // GRAPTOR_CONTAINER_HASH_SET_H
