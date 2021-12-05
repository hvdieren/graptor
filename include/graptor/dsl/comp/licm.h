// -*- c++ -*-
#ifndef GRAPTOR_DSL_COMP_LICM_H
#define GRAPTOR_DSL_COMP_LICM_H

namespace expr {

namespace detail {

/***********************************************************************
 * LICM for the part of the expression operating only on the destination.
 * This is used for CSC, as the destination index is constant and LICM is
 * correct
 ***********************************************************************/

/***********************************************************************
 * Pre-declarations
 ***********************************************************************/
template<bool nt, typename R, typename T>
auto licm_split( storeop<nt,R,T> s );

template<typename Expr, typename UnOp>
auto licm_split( unop<Expr, UnOp> u );

template<typename R, typename V, typename RedOp>
auto licm_split( redop<R,V,RedOp> r );

template<typename E1, typename E2, typename BinOp>
auto licm_split( binop<E1,E2,BinOp> b );

template<typename A, typename T, unsigned short VL>
auto licm_split( refop<A,T,VL> r );

/***********************************************************************
 * Predicates to determine if LICM can be applied.
 ***********************************************************************/
template<typename Expr>
struct licm_peel_p : std::false_type { };

template<typename Tr, value_kind VKind>
struct licm_peel_p<value<Tr, VKind>> {
    static constexpr bool value = VKind != vk_src && VKind != vk_smk;
};

template<typename Expr, typename UnOp>
struct licm_peel_p<unop<Expr, UnOp>> : licm_peel_p<Expr> { };

template<typename E1, typename E2, typename BinOp>
struct licm_peel_p<binop<E1,E2,BinOp>> {
    static constexpr bool value
	= licm_peel_p<E1>::value && licm_peel_p<E2>::value;
};

template<typename E1, typename E2>
struct licm_peel_p<binop<E1,E2,binop_seq>> {
    static constexpr bool value
	= licm_peel_p<E1>::value || licm_peel_p<E2>::value;
};

template<typename E1, typename E2, typename E3, typename TernOp>
struct licm_peel_p<ternop<E1,E2,E3,TernOp>> {
    static constexpr bool value
	= licm_peel_p<E1>::value && licm_peel_p<E2>::value
	&& licm_peel_p<E3>::value;
};

template<typename A, typename T, unsigned short VL>
struct licm_peel_p<refop<A,T,VL>> {
    // TODO: how much splitting can be done on a storeop?
    static constexpr bool value = licm_peel_p<T>::value;
};

template<typename S, typename U, typename C, typename DFSAOp>
struct licm_peel_p<dfsaop<S,U,C,DFSAOp>> {
    // TODO: how much splitting can be done on a dfsaop?
    static constexpr bool value = false; // licm_peel_p<T>::value;
};


template<bool nt, typename R, typename T>
struct licm_peel_p<storeop<nt,R,T>> {
    static constexpr bool value = licm_peel_p<R>::value;
};

template<typename R, typename V, typename RedOp>
struct licm_peel_p<redop<R,V,RedOp>> {
    static constexpr bool value = licm_peel_p<R>::value;
};

template<typename Red, typename CacheOp, typename LE, typename PE_>
struct licm_result {
public:
    using PE = PE_;

    licm_result( Red red, CacheOp cop, LE le, PE pe )
	: m_red( red ), m_cop( cop ), m_le( le ), m_pe( pe ) { }

    Red red() const { return m_red; }
    CacheOp cop() const { return m_cop; }
    
    LE le() const { return m_le; }
    PE pe() const { return m_pe; }

private:
    Red m_red;
    CacheOp m_cop;
    LE m_le;
    PE m_pe;
};

template<typename LE>
struct licm_result<void,void,LE,noop> {
public:
    using PE = noop;

    licm_result( LE le ) : m_le( le ) { }

    LE le() const { return m_le; }
    noop pe() const { return noop(); }
    
private:
    LE m_le;
};

template<typename Red, typename CacheOp, typename LE, typename PE>
auto make_licm_result( Red red, CacheOp cop, LE le, PE pe ) {
    return licm_result<Red,CacheOp,LE,PE>( red, cop, le, pe );
}

template<typename LE>
auto make_licm_result_loop( LE le ) {
    return licm_result<void,void,LE,noop>( le );
}

template<typename Red1, typename CacheOp1, typename LE1, typename PE1,
	 typename Red2, typename CacheOp2, typename LE2, typename PE2>
auto merge_licm_result_seq( licm_result<Red1,CacheOp1,LE1,PE1> l,
			    licm_result<Red2,CacheOp2,LE2,PE2> r ) {
    // Check if either LICM has as its PE a bare value<Tr,vk_smk>.
    // If so, LICM was not applied
    if constexpr ( is_value_vk<PE1,vk_smk>::value ) {
	if constexpr ( is_value_vk<PE2,vk_smk>::value ) {
	    return make_licm_result_loop( make_seq( l.le(), r.le() ) );
	} else if constexpr ( std::is_same_v<expr::noop,PE2> ) {
	    return make_licm_result( l.red(), l.cop(),
				     make_seq( l.le(), r.le() ),
				     l.pe() );
	} else {
	    return make_licm_result( r.red(), r.cop(),
				     make_seq( l.le(), r.le() ),
				     r.pe() );
	}
    } else {
	if constexpr ( is_value_vk<PE2,vk_smk>::value ) {
	    return make_licm_result( l.red(), l.cop(),
				     make_seq( l.le(), r.le() ),
				     l.pe() );
	} else {
	    return make_licm_result( r.red(), r.cop(),
				     make_seq( l.le(), r.le() ),
				     make_seq( l.pe(), r.pe() ) );
	}
    }
}

template<typename LE1, typename LE2>
auto merge_licm_result_seq( licm_result<void,void,LE1,noop> l,
			    licm_result<void,void,LE2,noop> r ) {
    return make_licm_result_loop( make_seq( l.le(), r.le() ) );
}

template<typename LICM, typename PE>
auto replace_licm_result_pe( LICM licm, PE pe ) {
    return make_licm_result( licm.red(), licm.cop(), licm.le(), pe );
}

template<typename Tr, value_kind VKind>
auto licm_split( value<Tr, VKind> v ) {
    return make_licm_result_loop( v );
}

template<typename Tr>
auto licm_split( value<Tr, vk_src> v ) {
    return make_licm_result_loop( v );
}

template<typename Tr>
auto licm_split( value<Tr, vk_smk> v ) {
    return make_licm_result_loop( v );
}

template<typename Expr, typename UnOp>
auto licm_split( unop<Expr, UnOp> u ) {
    auto licm = licm_split( u.data() );
    static constexpr bool do_licm
	= licm_peel_p<Expr>::value
	&& !std::is_same<noop,typename decltype(licm)::PE>::value;
    if constexpr ( do_licm ) {
	return replace_licm_result_pe(
	    licm,
	    make_unop( licm.pe(), UnOp() ) );
    } else {
	return make_licm_result_loop( u );
    }
}

template<typename E1, typename E2, typename BinOp>
auto licm_split( binop<E1,E2,BinOp> b ) {
    if constexpr ( std::is_same_v<BinOp,binop_seq> ) {
	auto licm_l = licm_split( b.data1() );
	auto licm_r = licm_split( b.data2() );
	return merge_licm_result_seq( licm_l, licm_r );
    } else {
	return make_licm_result_loop( b );
    }
}

template<typename E1, typename E2, typename E3, typename TernOp>
auto licm_split( ternop<E1,E2,E3,TernOp> t ) {
    return make_licm_result_loop( t );
}

template<bool nt, typename R, typename T>
auto licm_split( storeop<nt,R,T> s ) {
    return make_licm_result_loop( s );
}

template<bool peel, typename LICM, typename A, typename T, unsigned short VL>
auto licm_split_refop( LICM licm, refop<A,T,VL> r,
		       typename std::enable_if<peel>::type * = nullptr ) {
    return replace_licm_result_pe(
	licm,
	make_refop( r.array(), licm.pe() ) );
}

template<bool peel, typename LICM, typename A, typename T, unsigned short VL>
auto licm_split_refop( LICM licm, refop<A,T,VL> r,
		       typename std::enable_if<!peel>::type * = nullptr ) {
    return make_licm_result_loop( r );
}

template<typename A, typename T, unsigned short VL>
auto licm_split( refop<A,T,VL> r ) {
    auto licm = licm_split( r.index() );
    static constexpr bool do_licm
	= licm_peel_p<T>::value
	&& !std::is_same<noop,typename decltype(licm)::PE>::value;
    return licm_split_refop<do_licm>( licm, r );
}

template<typename R, typename V, typename RedOp>
auto licm_split( redop<R,V,RedOp> r ) {
    if constexpr ( licm_peel_p<R>::value ) {
	// (v, e) <- licm_split(V)
	auto ve = licm_split( r.val() ); // .rexpr() );
	if constexpr ( std::is_same_v<typename decltype(ve)::PE,noop> ) {
	    static constexpr unsigned short VL = redop<R,V,RedOp>::VL;
	    using T = typename V::type;
	    using Tr = simd::ty<T,VL>;
	    // We capture the return value of the reduction operation. The tree
	    // is broken in two parts at this point. The return value of the
	    // loop expression part will be fed into the post-loop expression
	    // part. Note that this is always a mask.
	    value<Tr,vk_smk> rret; // Note: it is a hack to reuse vk_smk for this
	    return make_licm_result( r, rret,
				     // redop expression remains inside loop
				     make_redop( r.ref(), ve.le(), RedOp() ),
				     // its return value becomes an rret
				     rret );
	} else {
	    return replace_licm_result_pe(
		ve, make_redop( r.ref(), ve.pe(), RedOp() ) );
	}
    } else 
	return make_licm_result_loop( r );
}

template<typename S, typename U, typename C, typename DFSAOp>
auto licm_split( dfsaop<S,U,C,DFSAOp> r ) {
    return make_licm_result_loop( r ); // don't peel off code
}

} // namespace detail

#if DISABLE_FISSION
template<typename Expr>
auto licm_split_main( Expr e ) {
    // Loop fission disabled, return expression as part of loop
    return detail::make_licm_result_loop( e );
}

#else // DISABLE_FISSION (enabled below)

template<typename Expr>
auto licm_split_main( Expr e ) {
    return detail::licm_split( e );
}

// Conditionally enabled/disable LICM depending on whether the code uses
// an owner-writes construction, or LICM is otherwise meaningful
template<bool enabled, typename Expr>
auto licm_split( Expr e ) {
    if constexpr ( enabled )
	return detail::licm_split( e );
    else
	return detail::make_licm_result_loop( e );
}

#endif // DISABLE_FISSION (enabled)

} // namespace expr

#endif // GRAPTOR_DSL_COMP_LICM_H
