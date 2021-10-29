// -*- c++ -*-
#ifndef GRAPTOR_BITFIELD_H
#define GRAPTOR_BITFIELD_H

#include <ostream>

template<unsigned short Bits>
class bitfield {
public:
    static constexpr unsigned short bits = Bits;
    static constexpr unsigned short factor = 8/bits;
    static_assert( bits == 1 || bits == 2 || bits == 4,
		   "assuming a whole number of bitfields per byte" );
    using storage_type = uint8_t;

public:
    bitfield() { }
    bitfield( storage_type v ) : m_value( v & ((storage_type(1)<<bits)-1) ) { }
    bitfield( const bitfield & b ) : m_value( b.m_value ) { }

    storage_type get() const { return m_value; }
    operator storage_type () const { return m_value; }
    template<unsigned short Bytes>
    operator logical<Bytes> () const {
	// Assume that bitfield is interpreted as a logical mask
	return logical<Bytes>::get_val( ( m_value >> (bits-1) ) & 1 );
    }

    template<typename Bl>
    static bitfield get_val( Bl b ) {
	return b ? bitfield<bits>( ~0 ) : bitfield<bits>( 0 );
    }

private:
    storage_type m_value;
};

template<unsigned short Bits>
std::ostream & operator << ( std::ostream & os, bitfield<Bits> b ) {
    return os << b.get();
}

template<typename T>
struct is_bitfield : public std::false_type { };

template<unsigned short Bits>
struct is_bitfield<bitfield<Bits>> : public std::true_type { };

template<typename T>
constexpr bool is_bitfield_v = is_bitfield<T>::value;

#endif //  GRAPTOR_BITFIELD_H
