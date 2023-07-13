// -*- c++ -*-
#ifndef GRAPTOR_CONTAINER_HASH_FN_H
#define GRAPTOR_CONTAINER_HASH_FN_H

#include <random>
#include <thread>

#include "graptor/target/vector.h"

namespace graptor {

template<typename T>
struct java_hash;

template<>
struct java_hash<uint32_t> {
    using type = uint32_t;

    explicit java_hash( uint32_t log_size ) { }

    void resize( uint32_t ) { }

    type operator() ( uint32_t h ) const {
	h ^= (h >> 20) ^ (h >> 12);
	return h ^ (h >> 7) ^ (h >> 4);
    }

    template<unsigned short VL>
    typename vector_type_traits_vl<uint32_t,VL>::type
    vectorized( typename vector_type_traits_vl<uint32_t,VL>::type h ) const {
	using tr = vector_type_traits_vl<type,VL>;
	using vtype = typename tr::type;

	vtype h20 = tr::srli( h, 20 );
	vtype h12 = tr::srli( h, 12 );
	h = tr::bitwise_xor( h20, tr::bitwise_xor( h, h12 ) );
	vtype h7 = tr::srli( h, 7 );
	vtype h4 = tr::srli( h, 4 );
	h = tr::bitwise_xor( h7, tr::bitwise_xor( h, h4 ) );
	return h;
    }
};

template<typename T>
struct rand_hash;

template<>
struct rand_hash<uint32_t> {
    // Same RNG as Blanusa's code.
    using type = uint32_t;

    explicit rand_hash( uint32_t log_size ) {
	resize( log_size );
    }

    void resize( uint32_t log_size ) {
	m_shift = 32 - log_size - 1;
	m_a = rand() | 1;
	m_b = rand() & ((uint32_t(1) << m_shift) - 1);
    }

    type operator() ( uint32_t h ) const {
	h = h * m_a + m_b;
	return h >> m_shift;
    }

    template<unsigned short VL>
    typename vector_type_traits_vl<uint32_t,VL>::type
    vectorized( typename vector_type_traits_vl<uint32_t,VL>::type h ) const {
	using tr = vector_type_traits_vl<type,VL>;
	using vtype = typename tr::type;

	vtype a_vec = tr::set1( m_a );
	vtype b_vec = tr::set1( m_b );
	vtype c = tr::mul( a_vec, h );
	vtype d = tr::add( c, b_vec );
	vtype e = tr::srli( d, m_shift );
	return e;
    }
    
private:
    uint32_t rand() {
	static thread_local mt19937* generator = nullptr;
	if( !generator ) {
	    pthread_t self = pthread_self();
	    generator = new mt19937( clock() + self );
	}
	uniform_int_distribution<int> distribution;
	return distribution( *generator );
    }

private:
    uint32_t m_a, m_b;
    uint32_t m_shift;
};

template<typename T>
struct murmur_hash;

template<>
struct murmur_hash<uint64_t> {
    using type = uint64_t;
    
    type operator()( type h ) const {
	h ^= h >> 33;
	h *= 0xff51afd7ed558ccdL;
	h ^= h >> 33;
	h *= 0xc4ceb9fe1a85ec53L;
	h ^= h >> 33;
	return h;
    }
};

template<>
struct murmur_hash<uint32_t> {
    using type = uint32_t;
    
    type operator()( type h ) const {
	h ^= h >> 16;
	h *= 0x85ebca6b;
	h ^= h >> 13;
	h *= 0xc2b2ae35;
	h ^= h >> 16;

	return h;
    }

    template<unsigned short VL>
    typename vector_type_traits_vl<uint32_t,VL>::type
    vectorized( typename vector_type_traits_vl<uint32_t,VL>::type h ) const {
	using tr = vector_type_traits_vl<type,VL>;
	using vtype = typename tr::type;
	const vtype c1 = tr::set1( 0x85ebca6b );
	const vtype c2 = tr::set1( 0xc2b2ae35 );
	h = tr::bitwise_xor( h, tr::srli( h, 16 ) );
	h = tr::mul( h, c1 );
	h = tr::bitwise_xor( h, tr::srli( h, 13 ) );
	h = tr::mul( h, c2 );
	h = tr::bitwise_xor( h, tr::srli( h, 16 ) );
	return h;
    }
};


} // namespace graptor

#endif // GRAPTOR_CONTAINER_HASH_FN_H
