#include "graptor/target/vector.h"

int main( int argc, char * argv[] ) {
    static constexpr unsigned short VL = 4;
    using type = long;
    using traits = vector_type_traits_vl<type,VL>;
    using vtype = typename traits::type;

    vtype a = traits::setzero();
    vtype b = traits::setone();
    vtype c = traits::add( a, b );

    bool is_true = true;
    for( unsigned short l=0; l < VL; ++l )
	is_true = is_true & ( traits::lane(b,l) == traits::lane(c,l) );

    return is_true ? 0 : -1;
}
