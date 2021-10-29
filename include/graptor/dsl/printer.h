// -*- c++ -*-
#ifndef GRAPTOR_DSL_PRINTER_H
#define GRAPTOR_DSL_PRINTER_H

/***********************************************************************
 * Printing expressions
 ***********************************************************************/
template<typename value_map_type, typename Cache>
struct printer {
    printer( Cache &c_, std::ostream &os_ )
	: m_cache( c_ ), os( os_ ), indent( 0 ) { }

    template<typename Tr>
    void print( value<Tr,vk_src> v ) {
	ind();
	os << "value VL=" << Tr::VLS << " VEW=" << Tr::W << " type=source";
    }
    template<typename Tr>
    void print( value<Tr,vk_smk> v ) {
	ind();
	os << "value VL=" << Tr::VLS << " VEW=" << Tr::W << " type=source-mask";
    }
    template<typename Tr>
    void print( value<Tr,vk_dst> v ) {
	ind();
	os << "value VL=" << Tr::VLD << " VEW=" << Tr::W << " type=destination";
    }
    template<typename Tr>
    void print( value<Tr,vk_pid> v ) {
	ind();
	os << "value VL=" << Tr::VLD << " VEW=" << Tr::W << " type=partition-ID";
    }
    template<typename Tr>
    void print( value<Tr,vk_cstone> v ) {
	ind();
	os << "value VL=" << Tr::VLD << " VEW=" << Tr::W << " type=cstone";
    }
    template<typename Tr>
    void print( value<Tr,vk_dmk> v ) {
	ind();
	os << "value VL=" << Tr::VLS << " VEW=" << Tr::W << " type=destination-mask";
    }
    template<typename Tr>
    void print( value<Tr,vk_any> v ) {
	ind();
	os << "value VL=" << Tr::VL << " VEW=" << Tr::W << " type=vk_any ( ";
	// print( v.data() );
	os << v.data();
	os << " )";
    }
    template<typename Tr>
    void print( value<Tr,vk_truemask> v ) {
	ind();
	os << "value VL=" << Tr::VL << " VEW=" << Tr::W << " type=one-val";
    }
    template<typename Tr>
    void print( value<Tr,vk_zero> v ) {
	ind();
	os << "value VL=" << Tr::VL << " VEW=" << Tr::W << " type=zero-val";
    }
    template<typename Tr>
    void print( value<Tr,vk_true> v ) {
	ind();
	os << "value VL=" << Tr::VL << " VEW=" << Tr::W << " type=true-mask";
    }

    template<typename Expr, typename UnOp>
    void print( unop<Expr, UnOp> uop ) {
	ind();
	os << "unop " << UnOp::name << " VL=" << Expr::VL << " VEW="
	   << sizeof(typename unop<Expr,UnOp>::type) << " (\n";
	indr();
	print( uop.data() );
	os << " )";
	indl();
    }

      template<typename E1, typename E2, typename BinOp>
      void print( binop<E1,E2,BinOp> bop ) {
	ind();
os << "binop " << BinOp::name << " VL=" << binop<E1,E2,BinOp>::VL
	   << " VEW=" << sizeof(typename binop<E1,E2,BinOp>::type)
	   << " (\n";
	indr();
	print( bop.data1() );
	os << ",\n";
	print( bop.data2() );
	os << " )";
	indl();
    }

    template<typename T, typename U, short AID>
    void print( array_ro<T,U,AID> array ) {
	ind();
	os << "array_ro ( " << array.ptr() << " sizeof(index)="
	   << sizeof(U) << " VEW=" << sizeof(T)
	   << " AID=" << AID << " )";
    }

    template<typename A, typename T, unsigned short VL>
    void print( refop<A,T,VL> op ) {
	ind();
	os << "refop VL=" << refop<A,T,VL>::VL
	   << " VEW=" << sizeof(typename refop<A,T,VL>::type)
	   << " (\n";
	indr();
	print( op.array() );
	os << ",\n";
	print( op.index() );
	os << " )";
	indl();
    }

    template<typename A, typename T, typename M, unsigned short VL>
    void print( maskrefop<A,T,M,VL> op ) {
	ind();
	os << "maskrefop VL=" << VL
	   << " VEW=" << sizeof(typename maskrefop<A,T,M,VL>::type)
	   << " (\n";
	indr();
	print( op.array() );
	os << ",\n";
	print( op.index() );
	os << ",\n";
	print( op.mask() );
	os << " )";
	indl();
    }

    template<unsigned cid, typename Tr>
    void print( cacheop<cid,Tr> op ) {
	// Note: the cache is applied only to destination fields, therefore
	//       its vector length must be VLD
	static_assert( Tr::VL == value_map_type::VLS, "sanity" );
	ind();
	os << "cacheop " << cid << " VL=" << Tr::VL
	    // << '/' << std::get<cid>(m_cache).VL
	   << " VEW=" << sizeof(typename cacheop<cid,Tr>::type);
    }

    template<typename R, typename T>
    void print( storeop<R,T> op ) {
	lprint( op.ref() ) = print( op.val() );
	ind();
	os << "storeop VL=" << storeop<R,T>::VL << " (\n";
	indr();
	print( op.ref() );
	os << ",\n";
	print( op.val() );
	os << " )";
	indl();
    }

    template<typename E1, typename E2, typename RedOp> // enable_if<is_redop<RedOp>>
    void print( redop<E1,E2,RedOp> op ) {
	ind();
	os << "redop " << RedOp::name << " VL=" << redop<E1,E2,RedOp>::VL
	   << " VEW=" << sizeof(typename redop<E1,E2,RedOp>::type)
	   << " (\n";
	indr();
	print( op.ref() );
	os << ",\n";
	print( op.val() );
	os << " )";
	indl();
    }
    
private:
    void ind() {
	static const char spaces[] =
	    "                                                                "
	    "                                                                ";
	os << &spaces[128-indent];
    }
    void indr() { indent += 4; }
    void indl() { indent -= 4; }

private:
    Cache &m_cache;    // updated values for reductions
    std::ostream &os;
    size_t indent;
};

#endif // GRAPTOR_DSL_PRINTER_H
