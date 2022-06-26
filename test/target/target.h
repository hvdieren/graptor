#include <ostream>
#include <iostream>
#include <random>
#include <sstream>

#include "graptor/target/vector.h"

#define GT_CAT_NX(a,b) a ## b
#define GT_CAT(a,b) GT_CAT_NX(a,b)

#define AS_STRING2(x) #x
#define AS_STRING(x) AS_STRING2(x)

using logical_1 = logical<1>;
using logical_2 = logical<2>;
using logical_4 = logical<4>;
using logical_8 = logical<8>;

template<typename tr>
auto random_generate( typename tr::member_type * vals ) {
    using type = typename tr::type;
    using member_type = typename tr::member_type;

    static std::random_device rd;
    static std::mt19937_64 eng(rd());

    if constexpr ( std::is_floating_point_v<member_type> ) {
	std::uniform_real_distribution<member_type> distr;
	for( unsigned short l=0; l < tr::vlen; ++l )
	    vals[l] = distr( eng );
	return tr::loadu( vals );
    } else if constexpr ( is_logical_v<member_type> ) {
	std::uniform_int_distribution<int> distr;
	for( unsigned short l=0; l < tr::vlen; ++l )
	    vals[l] = ( distr( eng ) & 1 )
		? member_type::true_val()
		: member_type::false_val();
	return tr::loadu( vals );
    } else if constexpr ( is_bitfield_v<member_type> ) {
	std::uniform_int_distribution<int> distr;
	typename tr::type vec;
	for( unsigned short l=0; l < tr::vlen; ++l ) {
	    vals[l] = member_type( distr( eng ) & ((1<<tr::bits)-1) );
	    vec = tr::setlane( vec, vals[l], l );
	}
	return vec;
    } else if constexpr ( std::is_same_v<member_type,bool> ) {
	std::uniform_int_distribution<int> distr;
	for( unsigned short l=0; l < tr::vlen; ++l )
	    vals[l] = member_type( ( distr( eng ) & 1 ) ? true : false );
	return tr::loadu( vals );
    } else {
	std::uniform_int_distribution<member_type> distr;
	for( unsigned short l=0; l < tr::vlen; ++l )
	    vals[l] = distr( eng );
	return tr::loadu( vals );
    }
}

template<typename T>
std::string display( const T x ) {
    return std::to_string( x );
}

template<unsigned short B>
std::string display( const logical<B> & x ) {
    std::stringstream s( std::ios_base::out );
    s << x;
    return s.str();
}

template<>
std::string display( const char x ) {
    std::string s = 'c' + std::to_string( (const int)x );
    return s;
}

template<typename tr>
void show(
    typename tr::type v,
    typename tr::member_type s[tr::vlen]
    ) {
    using type = typename tr::type;
    using member_type = typename tr::member_type;

    std::cerr << std::hex;
    for( unsigned short l=0; l < tr::vlen; ++l ) {
	std::cerr << "lane " << l << ": vector: "
		  << display( tr::lane( v, l ) )
		  << " scalar: " << display( s[l] ) << "\n";
    }
    std::cerr << std::dec;
}

template<typename tr>
unsigned short element_compare(
    typename tr::type v,
    typename tr::member_type s[tr::vlen]
    ) {
    using type = typename tr::type;
    using member_type = typename tr::member_type;

    unsigned short diff = 0;
    for( unsigned short l=0; l < tr::vlen; ++l ) {
	if constexpr ( is_logical_v<member_type> ) {
	    /*
	    if( ( tr::lane( v, l ) != member_type::true_val()
		  && tr::lane( v, l ) != member_type::false_val() )
		|| tr::lane( v, l ) != s[l] )
	    */
	    if( tr::lane( v, l ) != s[l] ) // redefined != for logical < 8 bytes
		++diff;
	} else if constexpr ( std::is_same_v<member_type,bool> ) {
	    if( (!tr::lane( v, l )) != (!s[l]) )
		++diff;
	} else {
	    if( tr::lane( v, l ) != s[l] )
		++diff;
	}
    }

    return diff;
}

template<typename tr>
unsigned short mask_compare(
    typename tr::mask_type v,
    bool s[tr::vlen]
    ) {
    using type = typename tr::type;
    using member_type = typename tr::member_type;
    using mtr = typename tr::mask_traits;

    unsigned short diff = 0;
    for( unsigned short l=0; l < tr::vlen; ++l ) {
	if( (!mtr::lane( v, l )) != (!s[l]) )
	    ++diff;
    }

    return diff;
}

template<typename tr>
unsigned short vmask_compare(
    typename tr::vmask_type v,
    bool s[tr::vlen]
    ) {
    using type = typename tr::type;
    using member_type = typename tr::member_type;
    using mtr = typename tr::int_traits;

    unsigned short diff = 0;
    for( unsigned short l=0; l < tr::vlen; ++l ) {
	if( (!mtr::lane( v, l )) != (!s[l]) )
	    ++diff;
    }

    return diff;
}

