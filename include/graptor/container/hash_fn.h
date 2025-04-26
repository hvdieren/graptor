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
	h = tr::bitwise_xor( h20, h, h12 );
	vtype h7 = tr::srli( h, 7 );
	vtype h4 = tr::srli( h, 4 );
	h = tr::bitwise_xor( h7, h, h4 );
	return h;
    }
};

template<typename T, T a0, T a1, T b0, T b1>
struct java_hash_variation {
    using type = T;

    explicit java_hash_variation( T log_size ) { }

    void resize( T ) { }

    type operator() ( T h ) const {
	h ^= (h >> a0) ^ (h >> a1);
	return h ^ (h >> b0) ^ (h >> b1);
    }

    template<unsigned short VL>
    typename vector_type_traits_vl<uint32_t,VL>::type
    vectorized( typename vector_type_traits_vl<uint32_t,VL>::type h ) const {
	using tr = vector_type_traits_vl<type,VL>;
	using vtype = typename tr::type;

	vtype h0 = tr::srli( h, a0 );
	vtype h1 = tr::srli( h, a1 );
	h = tr::bitwise_xor( h0, h, h1 );
	vtype g0 = tr::srli( h, b0 );
	vtype g1 = tr::srli( h, b1 );
	h = tr::bitwise_xor( g1, h, g0 );
	return h;
    }
};

template<typename T>
using java_hash_variation_A = java_hash_variation<T,14,18,3,8>;

template<typename T>
using java_hash_variation_B = java_hash_variation<T,5,9,13,21>;

template<typename T>
using java_hash_variation_C = java_hash_variation<T,3,6,11,18>;


template<typename T, typename Enable = void>
struct rand_hash;

template<typename T>
struct rand_hash<T, std::enable_if_t<std::is_integral_v<T>>> {
    // Same RNG as Blanusa's code.
    using type = T;
    static constexpr size_t bits = 8 * sizeof(T);

    explicit rand_hash( type log_size ) {
	resize( log_size );
    }

    void resize( type log_size ) {
	m_shift = bits - log_size - 1;
	m_a = rand() | 1;
	m_b = rand() & ((type(1) << m_shift) - 1);
    }

    type operator() ( type h ) const {
	h = h * m_a + m_b;
	return h >> m_shift;
    }

    template<unsigned short VL>
    typename vector_type_traits_vl<type,VL>::type
    vectorized( typename vector_type_traits_vl<type,VL>::type h ) const {
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
    type m_a, m_b;
    type m_shift;
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

template<typename T, typename H1, typename H2>
struct hash_fn_pair {
    using type = T;

    explicit hash_fn_pair( uint32_t log_size )
	: m_hash1( log_size ), m_hash2( log_size ) { }

    void resize( uint32_t log_size ) {
	m_hash1.resize( log_size );
	m_hash2.resize( log_size );
    }

    type operator() ( uint32_t h ) const { return fn1( h ); }

    type fn1( uint32_t h ) const { return m_hash1( h ); }
    type fn2( uint32_t h ) const { return m_hash2( h ); }

    template<unsigned short VL>
    typename vector_type_traits_vl<uint32_t,VL>::type
    vectorized( typename vector_type_traits_vl<uint32_t,VL>::type h ) const {
	return m_hash1.template vectorized<VL>( h );
    }

    template<unsigned short VL>
    typename vector_type_traits_vl<uint32_t,VL>::type
    vectorized1( typename vector_type_traits_vl<uint32_t,VL>::type h ) const {
	return m_hash1.template vectorized<VL>( h );
    }
    template<unsigned short VL>
    typename vector_type_traits_vl<uint32_t,VL>::type
    vectorized2( typename vector_type_traits_vl<uint32_t,VL>::type h ) const {
	return m_hash2.template vectorized<VL>( h );
    }

private:
    H1 m_hash1;
    H2 m_hash2;
};

} // namespace graptor

#endif // GRAPTOR_CONTAINER_HASH_FN_H
