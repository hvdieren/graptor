// -*- c++ -*-
#ifndef GRAPTOR_LONGINT_H
#define GRAPTOR_LONGINT_H

#include <x86intrin.h>
#include <immintrin.h>

/***********************************************************************
 * This class defines really long integers. We only seek to implement
 * bitwise logical operations on these integers (they are effectively
 * bitmasks). Primary usage is in FMv.
 ***********************************************************************/

// An integer with W bytes
template<unsigned short W, typename> //  = void>
class longint;

// Base case (assumes W >= 8)
#define LONGINT_BASE_SIZE 8
template<>
class longint<8> {
public:
    using type = uint64_t;
    static constexpr unsigned short W = 8;
    using self_type = longint<8>;
    using halfint = uint32_t;

    explicit longint( type v ) : m_value( v ) { }
    longint() { }

    static self_type setbit( size_t pos ) { return self_type( type(1)<<pos ); }

    static self_type setzero() { return self_type( 0 ); }

    type get() const { return m_value; }
    bool getbit( size_t pos ) { return (m_value >> pos) & 1; }
    halfint get_hi() const { return get() >> W/2; }
    halfint get_lo() const { return get(); }

    operator type () { return get(); }

    static self_type bitwiseor( self_type l, self_type r ) {
	return self_type( l.get() | r.get() );
    }
    static self_type bitwiseand( self_type l, self_type r ) {
	return self_type( l.get() & r.get() );
    }
    static self_type invert( self_type a ) {
	return self_type( ~a.get() );
    }

    bool operator == ( self_type r ) const {
	return get() == r.get();
    }
    bool operator != ( self_type r ) const {
	return get() != r.get();
    }

    self_type operator << ( unsigned int sh ) const {
	return self_type( get() << sh );
    }
    self_type operator >> ( unsigned int sh ) const {
	return self_type( get() >> sh );
    }

private:
    type m_value;
};

#ifdef __SSE4_1__

#if LONGINT_BASE_SIZE < 16
#undef LONGINT_BASE_SIZE 
#define LONGINT_BASE_SIZE 16
#endif

// Another base case of 16 bytes
template<>
class longint<16> {
public:
    using type = __m128i;
    static constexpr unsigned short W = 16;
    using self_type = longint<16>;
    using halfint = longint<8>;

    /*explicit*/ longint( type v ) : m_value( v ) { }
    explicit longint( halfint v ) : m_value( _mm_cvtsi64_si128( v.get() ) ) { }
    longint( uint64_t lo ) : m_value( _mm_cvtsi64_si128( lo ) ) { }
    longint() { }

    static longint set_pair( uint64_t hi, uint64_t lo ) {
	return longint(
	    _mm_insert_epi64(
		_mm_cvtsi64_si128( lo ), hi, 1 ) );
    }

    static longint set_pair( halfint hi, halfint lo ) {
	return set_pair( hi.get(), lo.get() );
    }

    static self_type setbit( size_t pos );

    static self_type setzero() { return self_type( _mm_setzero_si128() ); }

    type get() const { return m_value; }
    bool getbit( size_t pos ) {
	if( pos >= (W*8)/2 )
	    return get_hi().getbit( pos - (W*8)/2 );
	else
	    return get_lo().getbit( pos );
    }
    halfint get_hi() const { return halfint( _mm_extract_epi64( get(), 1 ) ); }
    halfint get_lo() const { return halfint( _mm_extract_epi64( get(), 0 ) ); }

    static self_type bitwiseor( self_type l, self_type r ) {
	return self_type( l.get() | r.get() );
    }
    static self_type bitwiseand( self_type l, self_type r ) {
	return self_type( l.get() & r.get() );
    }
    static self_type invert( self_type a ) {
	return self_type( ~a.get() );
    }

    bool operator == ( self_type r ) const;
    bool operator != ( self_type r ) const;

    template<uint8_t sh>
    self_type bslli() const {
	return self_type( _mm_bslli_si128( get(), sh ) );
    }
    
    self_type operator << ( const unsigned int sh ) const {
	assert( 0 && "NYI" );
	// return self_type( _mm_slli_si128( get(), sh ) );
    }
    self_type operator >> ( const unsigned int sh ) const {
	assert( 0 && "NYI" );
	// return self_type( _mm_srli_si128( get(), sh ) );
    }

private:
    type m_value;
};

#endif // __SSE4_1__

#ifdef __AVX__

#if LONGINT_BASE_SIZE < 32
#undef LONGINT_BASE_SIZE 
#define LONGINT_BASE_SIZE 32
#endif

// Another base case of 32 bytes
template<>
class longint<32> {
public:
    using type = __m256i;
    static constexpr unsigned short W = 32;
    using self_type = longint<32>;
    using halfint = longint<16>;

    /*explicit*/ longint( type v ) : m_value( v ) { }
    explicit longint( halfint v ) : m_value( _mm256_castsi128_si256( v.get() ) ) { }
    longint( uint64_t lo ) {
	__m256i a = _mm256_castsi128_si256( _mm_cvtsi64_si128( lo ) );
	m_value = _mm256_permute2x128_si256( a, a, 0x80 );
    }
    longint() { }

    static longint set_pair( halfint hi, halfint lo ) {
	return longint(
	    _mm256_inserti128_si256(
		_mm256_castsi128_si256( lo.get() ), hi.get(), 1 ) );
    }

    static self_type setbit( size_t pos );

    static self_type setzero() { return self_type( _mm256_setzero_si256() ); }

    type get() const { return m_value; }
    bool getbit( size_t pos ) {
	if( pos >= (W*8)/2 )
	    return get_hi().getbit( pos - (W*8)/2 );
	else
	    return get_lo().getbit( pos );
    }
    halfint get_hi() const {
	return halfint( _mm256_extracti128_si256( get(), 1 ) );
    }
    halfint get_lo() const {
	return halfint( _mm256_castsi256_si128( get() ) );
    }

    static self_type bitwiseor( self_type l, self_type r ) {
	return self_type( l.get() | r.get() );
    }
    static self_type bitwiseand( self_type l, self_type r ) {
	return self_type( l.get() & r.get() );
    }
    static self_type invert( self_type a ) {
	return self_type( ~a.get() );
    }

    bool operator == ( self_type r ) const;
    bool operator != ( self_type r ) const;

private:
    type m_value;
};

#endif // __AVX__

// Recursive case (assumes W > 8 and power of 2)
template<unsigned short W_>
class longint<W_,
	      typename std::enable_if<!(W_<=LONGINT_BASE_SIZE) && (W_ & (W_-1)) == 0>::type> {
public:
    static constexpr unsigned short W = W_;
    using halfint = longint<W/2>;
    using self_type = longint<W>;

    longint( halfint hi, halfint lo ) : m_hi( hi ), m_lo( lo ) { }
    explicit longint( halfint v ) : m_hi( 0 ), m_lo( v.get() ) { }
    explicit longint( int lo ) : m_hi( 0 ), m_lo( lo ) { }
    longint() { }

    static self_type setbit( size_t pos ) {
	if( pos >= W*4 )
	    return longint( halfint::setbit(pos-W*4), halfint(0) );
	else
	    return longint( halfint(0), halfint::setbit(pos) );
    }

    halfint get_hi() const { return m_hi; }
    halfint get_lo() const { return m_lo; }

    static self_type bitwiseor( self_type l, self_type r ) {
	return self_type(
	    halfint::bitwiseor( l.get_hi(), r.get_hi() ),
	    halfint::bitwiseor( l.get_lo(), r.get_lo() ) );
    }
    static self_type bitwiseand( self_type l, self_type r ) {
	return self_type(
	    halfint::bitwiseand( l.get_hi(), r.get_hi() ),
	    halfint::bitwiseand( l.get_lo(), r.get_lo() ) );
    }
    static self_type invert( self_type a ) {
	return self_type(
	    halfint::invert( a.get_hi() ),
	    halfint::invert( a.get_lo() ) );
    }

    bool operator == ( self_type r ) const {
	return get_hi() == r.get_hi() && get_lo() == r.get_lo();
    }
    bool operator != ( self_type r ) const {
	return get_hi() != r.get_hi() || get_lo() != r.get_lo();
    }

    self_type operator << ( unsigned int sh ) const {
	assert( 0 && "NYI" );
	//halfint shi = ( m_hi << sh ) | ( m_lo >> (8*W - sh) );
	//halfint slo = m_lo << sh;
	//return self_type( shi, slo );
    }
    self_type operator >> ( unsigned int sh ) const {
	assert( 0 && "NYI" );
	//halfint slo = ( m_lo >> sh ) | ( m_hi << (8*W - sh) );
	//halfint shi = m_hi >> sh;
	//return self_type( shi, slo );
    }

private:
    halfint m_hi, m_lo;
};

// Generic definition
template<unsigned short W>
static longint<W> operator | ( longint<W> l, longint<W> r ) {
    return longint<W>::bitwiseor( l, r );
}

template<unsigned short W>
static longint<W> operator & ( longint<W> l, longint<W> r ) {
    return longint<W>::bitwiseand( l, r );
}

template<unsigned short W>
static longint<W> operator ~ ( longint<W> a ) {
    return longint<W>::invert( a );
}


/***********************************************************************
 * Traits
 ***********************************************************************/
template<typename T>
struct is_longint : public std::false_type { };

template<unsigned short W>
struct is_longint<longint<W>> : public std::true_type { };

template<typename T>
constexpr bool is_longint_v = is_longint<T>::value;

#endif // GRAPTOR_LONGINT_H
