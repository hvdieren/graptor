#include "target.h"

using bitfield1 = bitfield<1>;
using bitfield2 = bitfield<2>;

// binop: type x type -> mask_type
template<typename T, typename U, unsigned short VL>
bool test() {
    using t_tr = vector_type_traits_vl<T,VL>;
    using t_type = typename t_tr::type;
    using t_member_type = typename t_tr::member_type;
    using t_str = vector_type_traits_vl<T,1>;

    using u_tr = vector_type_traits_vl<U,VL>;
    using u_type = typename u_tr::type;
    using u_member_type = typename u_tr::member_type;
    using u_str = vector_type_traits_vl<U,1>;

    static_assert( std::is_same_v<typename t_str::member_type,t_member_type>,
		   "expect the same types in vector and scalar traits" );
    static_assert( std::is_same_v<T,t_member_type>,
		   "expect the member_type in the traits equals T" );
    static_assert( std::is_same_v<typename u_str::member_type,u_member_type>,
		   "expect the same types in vector and scalar traits" );
    static_assert( std::is_same_v<U,u_member_type>,
		   "expect the member_type in the traits equals U" );

    t_member_type avals[VL];
    u_member_type cvals[VL];

    t_type a = random_generate<t_tr>( avals );

    u_type c = conversion_traits<T,U,VL>::convert( a );

    for( unsigned short l=0; l < VL; ++l )
	cvals[l] = static_cast<U>( avals[l] );

    bool ret = true;
    
    unsigned short adiff = element_compare<t_tr>( a, avals );
    if( adiff != 0 ) {
	std::cerr << "vector A encoding error in " << adiff << " positions\n";
	ret = false;
    }

    unsigned short cdiff = element_compare<u_tr>( c, cvals );
    if( cdiff != 0 ) {
	std::cerr << "vector C conversion error in " << cdiff << " positions\n";
	ret = false;
    }

    if( adiff || cdiff ) {
	std::cerr << "vector A:\n";
	show<t_tr>( a, avals );
	std::cerr << "vector C:\n";
	show<u_tr>( c, cvals );
    }

    return ret;
}

int main( int argc, char * argv[] ) {
    std::cerr << "unit-test target conversion operation from type '"
	      << AS_STRING(T_TYPE)
	      << "' to type '" << AS_STRING(U_TYPE)
	      << "' vector length " << VECTOR_LENGTH << "\n";
    
    return test<T_TYPE,U_TYPE,VECTOR_LENGTH>() ? 0 : 1;
}
