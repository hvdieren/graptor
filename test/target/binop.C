#include "target.h"

// binop: type x type -> type
template<typename T, unsigned short VL>
bool test() {
    using tr = vector_type_traits_vl<T,VL>;
    using type = typename tr::type;
    using member_type = typename tr::member_type;
    using str = vector_type_traits_vl<T,1>;

    static_assert( std::is_same_v<typename str::member_type,member_type>,
		   "expect the same types in vector and scalar traits" );
    static_assert( std::is_same_v<T,member_type>,
		   "expect the member_type in the traits equals T" );

    member_type avals[VL], bvals[VL], cvals[VL];

    type a = random_generate<tr>( avals );
    type b = random_generate<tr>( bvals );

    type c = tr::BINOP( a , b );

    for( unsigned short l=0; l < VL; ++l )
	cvals[l] = str::BINOP( avals[l], bvals[l] );

    bool ret = true;
    
    unsigned short adiff = element_compare<tr>( a, avals );
    if( adiff != 0 ) {
	std::cerr << "vector A encoding error in " << adiff << " positions\n";
	ret = false;
    }

    unsigned short bdiff = element_compare<tr>( b, bvals );
    if( bdiff != 0 ) {
	std::cerr << "vector A encoding error in " << bdiff << " positions\n";
	ret = false;
    }

    unsigned short cdiff = element_compare<tr>( c, cvals );
    if( cdiff != 0 ) {
	std::cerr << "vector C calculation error in " << cdiff << " positions\n";
	ret = false;
    }

    if( adiff || bdiff || cdiff ) {
	std::cerr << "vector A:\n";
	show<tr>( a, avals );
	std::cerr << "vector B:\n";
	show<tr>( b, bvals );
	std::cerr << "vector C:\n";
	show<tr>( c, cvals );
    }

    return ret;
}

int main( int argc, char * argv[] ) {
    std::cerr << "unit-test target binary operation '"
	      << AS_STRING(BINOP)
	      << "' type '" << AS_STRING(TYPE)
	      << "' vector length " << VECTOR_LENGTH << "\n";
    
    return test<TYPE,VECTOR_LENGTH>() ? 0 : 1;
}
