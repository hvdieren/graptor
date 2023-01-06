// -*- c++ -*-
#ifndef GRAPTOR_DSL_PTRSET_H
#define GRAPTOR_DSL_PTRSET_H

#include "graptor/dsl/ast.h"
#include "graptor/dsl/comp/cache.h"

namespace expr {

/**********************************************************************
 * Utilities for removing, what are assumed to be, recurring pointers
 * in multiple arrays in an AST. As the AST is a tree, if an array is
 * read multiple times, then the base address of the array is stored
 * multiple times. The C++ compiler, however, does not know these are
 * the same pointers. This pass collects the unique pointers, which
 * reduces the number of pointers used and register pressure, and creates
 * opportunities for reusing data.
 * The method aid_aliased and struct aid_is_aliased are used to determine
 * if AIDs are aliased. This is based on the AIDs used for the arrays.
 **********************************************************************/

namespace ast_ptrset {

// Auxiliary: make pointers volatile?
namespace {
#if ENABLE_VOLATILE_PTRS
template<typename T>
std::add_volatile_t<T *> cvt( T * ptr ) {
    return const_cast<std::add_volatile_t<T *>>( ptr );
}
#else
template<typename T>
T * cvt( T * ptr ) {
    return ptr;
}
#endif
} // namespace anonymous

template<unsigned Index, typename ValueTy, typename Map,
	 typename Enable = void>
struct add_if_absent;

template<unsigned Index, typename ValueTy, typename Map>
struct add_if_absent<Index,ValueTy,Map,
		     std::enable_if_t<expr::map_contains_v<Index,Map>>> {
    using type = Map;
};

template<unsigned Index, typename ValueTy, typename... Entries>
struct add_if_absent<Index,ValueTy,expr::map_new<Entries...>,
		     std::enable_if_t<!expr::map_contains_v<Index,expr::map_new<Entries...>>>> {
    using type = expr::map_new<expr::map_entry<Index,ValueTy>, Entries...>;
};

template<typename EntryType, typename Map>
using add_if_absent_t = typename add_if_absent<
    EntryType::index,typename EntryType::value_type,Map>::type;

template<typename P1, typename P2>
struct merge_maps;

template<typename RHS>
struct merge_maps<expr::map_new<>,RHS> {
    using type = RHS;
};
    
template<typename L0, typename... Ls, typename RHS>
struct merge_maps<expr::map_new<L0,Ls...>,RHS> {
    using type =
	typename add_if_absent<L0::index,typename L0::value_type,RHS>::type;
};


template<typename SyntaxTree, typename Map>
struct ptrset;

template<typename Map>
struct ptrset<noop,Map> {
    using map_type = Map;

    template<typename MapTy>
    static void initialize( MapTy & map, const noop & ) { }
};

template<short AID, typename T, typename Map>
struct ptrset_pointer {
    using entry_type = expr::map_entry<(unsigned)aid_key(array_aid(AID)), T *>;
    using map_type = add_if_absent_t<entry_type,Map>;
    static_assert( expr::map_contains_v<entry_type::index,map_type>, "check" );

    template<typename MapTy>
    static void initialize( MapTy & map, T * p ) {
	map.template get<(unsigned)aid_key(array_aid(AID))>() = cvt( p );
    }
};

template<typename T, typename U, short AID, typename Enc, bool NT,
	 typename Map>
struct ptrset<array_ro<T, U, AID, Enc, NT>, Map> {
    using entry_type = expr::map_entry<(unsigned)aid_key(array_aid(AID)),
				       typename Enc::storage_type *>;
    using map_type = add_if_absent_t<entry_type,Map>;

    template<typename MapTy>
    static void initialize( MapTy & map,
			    const array_ro<T, U, AID, Enc, NT> & a ) {
	map.template get<(unsigned)aid_key(array_aid(AID))>()
	    = cvt( a.ptr() );
    }
};

template<typename T, typename U, short AID, typename Map>
struct ptrset<bitarray_ro<T, U, AID>, Map> {
    using entry_type = expr::map_entry<(unsigned)aid_key(array_aid(AID)), T *>;
    using map_type = add_if_absent_t<entry_type,Map>;

    template<typename MapTy>
    static void initialize( MapTy & map, const bitarray_ro<T, U, AID> & a ) {
	map.template get<(unsigned)aid_key(array_aid(AID))>()
	    = cvt( a.ptr() );
    }
};

template<typename T, typename U, short AID, typename Enc, bool NT, typename Map>
struct ptrset<array_intl<T, U, AID, Enc, NT>, Map> {
    using entry_type = void;
    using map_type = Map;

    template<typename MapTy>
    static void initialize( MapTy & map, const array_intl<T, U, AID, Enc, NT> & a ) { }
};

template<typename T, typename U, short AID, typename Map>
struct ptrset<bitarray_intl<T, U, AID>, Map> {
    using entry_type = void;
    using map_type = Map;

    template<typename MapTy>
    static void initialize( MapTy & map, const bitarray_intl<T, U, AID> & a ) { }
};

template<typename Tr, value_kind VKind, typename Map>
struct ptrset<value<Tr, VKind>, Map> {
    using entry_type = void;
    using map_type = Map;

    template<typename MapTy>
    static void initialize( MapTy & map, const value<Tr,VKind> & ) { }
};

template<typename Expr, typename UnOp, typename Map>
struct ptrset<unop<Expr,UnOp>, Map> {
    using entry_type = void;
    using map_type = typename ptrset<Expr,Map>::map_type;
    
    template<typename MapTy>
    static void initialize( MapTy & map, const unop<Expr,UnOp> & u ) {
	ptrset<Expr,Map>::initialize( map, u.data() );
    }
};

template<typename E1, typename E2, typename BinOp, typename Map>
struct ptrset<binop<E1,E2,BinOp>, Map> {
    using entry_type = void;
    using map_type0 = typename ptrset<E2,Map>::map_type;
    using map_type = typename ptrset<E1,map_type0>::map_type;

    template<typename MapTy>
    static void initialize( MapTy & map, const binop<E1,E2,BinOp> & b ) {
	ptrset<E1,map_type0>::initialize( map, b.data1() );
	ptrset<E2,Map>::initialize( map, b.data2() );
    }
};

template<typename E1, typename E2, typename E3, typename TernOp, typename Map>
struct ptrset<ternop<E1,E2,E3,TernOp>, Map> {
    using entry_type = void;
    using map_type0 = typename ptrset<E3,Map>::map_type;
    using map_type1 = typename ptrset<E2,map_type0>::map_type;
    using map_type = typename ptrset<E1,map_type1>::map_type;
    
    template<typename MapTy>
    static void initialize( MapTy & map, const ternop<E1,E2,E3,TernOp> & t ) {
	ptrset<E1,map_type1>::initialize( map, t.data1() );
	ptrset<E2,map_type0>::initialize( map, t.data2() );
	ptrset<E3,Map>::initialize( map, t.data3() );
    }
};

template<typename A, typename T, unsigned short VL, typename Map>
struct ptrset<refop<A,T,VL>, Map> {
    using entry_type = void;
    using map_type0 = typename ptrset<T,Map>::map_type;
    using map_type = typename ptrset<A,map_type0>::map_type;
    
    template<typename MapTy>
    static void initialize( MapTy & map, const refop<A,T,VL> & r ) {
	ptrset<A,map_type0>::initialize( map, r.array() );
	ptrset<T,Map>::initialize( map, r.index() );
    }
};

template<typename A, typename T, typename M, unsigned short VL, typename Map>
struct ptrset<maskrefop<A,T,M,VL>, Map> {
    using entry_type = void;
    using map_type0 = typename ptrset<M,Map>::map_type;
    using map_type1 = typename ptrset<T,map_type0>::map_type;
    using map_type = typename ptrset<A,map_type1>::map_type;
    
    template<typename MapTy>
    static void initialize( MapTy & map, const maskrefop<A,T,M,VL> & r ) {
	ptrset<A,map_type1>::initialize( map, r.array() );
	ptrset<T,map_type0>::initialize( map, r.index() );
	ptrset<M,Map>::initialize( map, r.mask() );
    }
};

template<bool nt, typename R, typename T, typename Map>
struct ptrset<storeop<nt,R,T>, Map> {
    using entry_type = void;
    using map_type0 = typename ptrset<T,Map>::map_type;
    using map_type = typename ptrset<R,map_type0>::map_type;

    template<typename MapTy>
    static void initialize( MapTy & map, const storeop<nt,R,T> & s ) {
	ptrset<R,map_type0>::initialize( map, s.ref() );
	ptrset<T,Map>::initialize( map, s.value() );
    }
};

template<unsigned cid, typename Tr, typename Map>
struct ptrset<cacheop<cid,Tr>, Map> {
    using entry_type = void;
    using map_type = Map;

    template<typename MapTy>
    static void initialize( MapTy & map, const cacheop<cid,Tr> & ) { }
};

template<typename E1, typename E2, typename RedOp, typename Map>
struct ptrset<redop<E1,E2,RedOp>, Map> {
    using entry_type = void;
    using map_type0 = typename ptrset<E2,Map>::map_type;
    using map_type =
	typename ptrset<E1,map_type0>::map_type;

    template<typename MapTy>
    static void initialize( MapTy & map, const redop<E1,E2,RedOp> & r ) {
	ptrset<E1,map_type0>::initialize( map, r.ref() );
	ptrset<E2,Map>::initialize( map, r.val() );
    }
};

template<typename S, typename U, typename C, typename DFSAOp, typename Map>
struct ptrset<dfsaop<S,U,C,DFSAOp>, Map> {
    using entry_type = void;
    using map_type0 = typename ptrset<C,Map>::map_type;
    using map_type1 = typename ptrset<U,map_type0>::map_type;
    using map_type = typename ptrset<S,map_type1>::map_type;
    
    template<typename MapTy>
    static void initialize( MapTy & map, const dfsaop<S,U,C,DFSAOp> & d ) {
	ptrset<S,map_type1>::initialize( map, d.state() );
	ptrset<U,map_type0>::initialize( map, d.update() );
	ptrset<C,Map>::initialize( map, d.condition() );
    }
};

template<typename Map>
struct ptrset<expr::cache<>, Map> {
    using entry_type = void;
    using map_type = Map;

    template<typename MapTy>
    static void initialize( MapTy & map, cache<>& ) { }
};
    
template<typename C, typename... Cs, typename Map>
struct ptrset<expr::cache<C,Cs...>, Map> {
    using entry_type = void;
    using map_type0 = typename ptrset<expr::cache<Cs...>,Map>::map_type;
    using ref_type = typename C::orig_ref_type;
    using map_type = typename ptrset<ref_type,map_type0>::map_type;

    template<typename MapTy>
    static void initialize( MapTy & map, const cache<C,Cs...> & c ) {
	ptrset<ref_type,map_type0>::initialize( map, expr::car( c ).get_ref() );
	if constexpr ( sizeof...( Cs ) > 0 )
	    ptrset<expr::cache<Cs...>,Map>::initialize( map, expr::cdr( c ) );
    }
};

template<typename Map, typename... C>
struct ptrset_list;

template<typename Map, typename C>
struct ptrset_list<Map,C> {
    using entry_type = void;
    using map_type = typename ptrset<std::decay_t<C>,Map>::map_type;

    template<typename MapTy>
    static void initialize( MapTy & map, C c ) {
	ptrset<std::decay_t<C>,Map>::initialize( map, c );
    }
};
    
template<typename Map, typename C, typename... Cs>
struct ptrset_list<Map,C,Cs...> {
    using entry_type = void;
    using map_type0 = typename ptrset_list<Map,Cs...>::map_type;
    using map_type = typename ptrset<std::decay_t<C>,map_type0>::map_type;

    template<typename MapTy>
    static void initialize( MapTy & map, C c0, Cs... cs ) {
	ptrset<std::decay_t<C>,map_type0>::initialize( map, c0 );
	ptrset_list<Map,Cs...>::initialize( map, cs... );
    }
};
     
    
// Decl
template<typename PtrSet, bool nt, typename R, typename T>
static constexpr auto extract_ptrset( storeop<nt,R,T> s, PtrSet && set );

template<typename PtrSet, typename E1, typename E2, typename BinOp>
static constexpr auto extract_ptrset( binop<E1,E2,BinOp> b, PtrSet && set );

template<typename PtrSet, typename E1, typename E2, typename E3,
	 typename TernOp>
static constexpr auto extract_ptrset( ternop<E1,E2,E3,TernOp> b, PtrSet && set );

template<typename PtrSet, unsigned cid, typename Tr>
static constexpr auto extract_ptrset( cacheop<cid,Tr> c, PtrSet && set );

template<typename PtrSet, typename A, typename T, unsigned short VL>
static constexpr auto extract_ptrset( refop<A,T,VL> r, PtrSet && set );

template<typename PtrSet, typename A, typename T, typename M, unsigned short VL>
static constexpr auto extract_ptrset( maskrefop<A,T,M,VL> r, PtrSet && set );

template<typename PtrSet, typename E1, typename E2, typename RedOp>
static constexpr auto extract_ptrset( redop<E1,E2,RedOp> r, PtrSet && set );

template<typename PtrSet, typename S, typename U, typename C, typename DFSAOp>
static constexpr auto extract_ptrset( dfsaop<S,U,C,DFSAOp> s, PtrSet && set );

// Impl
    
template<typename PtrSet>
static constexpr
auto extract_ptrset( noop, PtrSet && set ) {
    return set;
}

template<typename PtrSet, typename T, typename U, short AID,
	 typename Enc, bool NT>
static constexpr
auto extract_ptrset( array_ro<T, U, AID, Enc, NT> a, PtrSet && set ) {
    return map_set_if_absent_v2<(unsigned)aid_key(array_aid(AID))>(
	set, cvt( a.ptr() ) );
}

template<typename PtrSet, typename T, typename U, short AID,
	 typename Enc, bool NT>
static constexpr
auto extract_ptrset( array_intl<T, U, AID, Enc, NT> a, PtrSet && set ) {
    return set;
}

template<typename PtrSet, typename T, typename U, short AID>
static constexpr
auto extract_ptrset( bitarray_intl<T, U, AID> a, PtrSet && set ) {
    return set;
}

template<typename PtrSet, typename T, typename U, short AID>
static constexpr
auto extract_ptrset( bitarray_ro<T, U, AID> a, PtrSet && set ) {
    return map_set_if_absent_v2<(unsigned)aid_key(array_aid(AID))>(
	set, cvt( a.ptr() ) );
}

template<typename PtrSet, typename Tr, value_kind VKind>
static constexpr
auto extract_ptrset( value<Tr, VKind>, PtrSet && set ) {
    return std::forward<PtrSet>( set );
}

template<typename PtrSet, typename Expr, typename UnOp>
static constexpr
auto extract_ptrset( unop<Expr,UnOp> u, PtrSet && set ) {
    return extract_ptrset( u.data(), std::forward<PtrSet>( set ) );
}

template<typename PtrSet, typename E1, typename E2, typename BinOp>
static constexpr
auto extract_ptrset( binop<E1,E2,BinOp> b, PtrSet && set ) {
    return extract_ptrset( b.data1(), extract_ptrset( b.data2(), std::forward<PtrSet>( set ) ) );
}

template<typename PtrSet, typename E1, typename E2, typename E3,
	 typename TernOp>
static constexpr auto extract_ptrset( ternop<E1,E2,E3,TernOp> e, PtrSet && set ) {
    return extract_ptrset( e.data1(),
			   extract_ptrset( e.data2(),
					   extract_ptrset( e.data3(), std::forward<PtrSet>( set ) ) ) );
}

template<typename PtrSet, typename A, typename T, unsigned short VL>
static constexpr
auto extract_ptrset( refop<A,T,VL> r, PtrSet && set ) {
    auto set2 = extract_ptrset( r.index(), std::forward<PtrSet>( set ) );
    return extract_ptrset( r.array(), std::forward<decltype(set2)>( set2 ) );
}

template<typename PtrSet, typename A, typename T, typename M, unsigned short VL>
static constexpr
auto extract_ptrset( maskrefop<A,T,M,VL> r, PtrSet && set ) {
    return extract_ptrset( r.array(),
			   extract_ptrset( r.index(),
					   extract_ptrset( r.mask(), std::forward<PtrSet>( set ) ) ) );
}

template<typename PtrSet, bool nt, typename R, typename T>
static constexpr
auto extract_ptrset( storeop<nt,R,T> s, PtrSet && set ) {
    return extract_ptrset( s.ref(), extract_ptrset( s.value(), std::forward<PtrSet>( set ) ) );
}

template<typename PtrSet, unsigned cid, typename Tr>
static constexpr
auto extract_ptrset( cacheop<cid,Tr> c, PtrSet && set ) {
    return set;
}

template<typename PtrSet, typename E1, typename E2, typename RedOp>
static constexpr
auto extract_ptrset( redop<E1,E2,RedOp> r, PtrSet && set ) {
    return extract_ptrset( r.ref(), extract_ptrset( r.val(), std::forward<PtrSet>( set ) ) );
}

template<typename PtrSet, typename S, typename U, typename C, typename DFSAOp>
static constexpr
auto extract_ptrset( dfsaop<S,U,C,DFSAOp> s, PtrSet && set ) {
    return extract_ptrset( s.state(),
			   extract_ptrset( s.update(),
					   extract_ptrset( s.condition(),
							   std::forward<PtrSet>( set ) ) ) );
}

template<typename PtrSet>
static constexpr
auto extract_ptrset( expr::cache<> cache, PtrSet && set ) {
    return set;
}

template<typename PtrSet, typename C, typename... Cs>
static constexpr
auto extract_ptrset( expr::cache<C,Cs...> cache, PtrSet && set ) {
    return extract_ptrset( expr::car( cache ).get_ref(),
			   extract_ptrset( expr::cdr( cache ),
					   std::forward<PtrSet>( set ) ) );
}

} // namespace ast_ptrset

// Also accepting expr::cache<> as argument and will iterate of its references
template<typename PtrSet, typename AST>
static constexpr
auto extract_pointer_set_with_extend( const PtrSet & ptrset, AST ast ) {
    // Assumes PtrSet already contains all relevant entries
    return ast_ptrset::extract_ptrset( ast, ptrset );
}

template<typename PtrSet, typename AST0, typename... AST>
static constexpr
auto extract_pointer_set_with_extend( const PtrSet & ptrset, AST0 ast0, AST... astn ) {
    // Assumes PtrSet already contains all relevant entries
    return ast_ptrset::extract_ptrset( ast0, extract_pointer_set_with_extend( ptrset, astn... ) );
}

template<typename AST>
static constexpr
auto extract_pointer_set( AST ast ) {
    using map_type = decltype(ast_ptrset::extract_ptrset( ast, create_map() ));
    return ast_ptrset::extract_ptrset( ast, map_type() );
}

template<typename AST0, typename... AST>
static constexpr
auto extract_pointer_set( AST0 ast0, AST... astn ) {
    using map_type = decltype(ast_ptrset::extract_ptrset( ast0, extract_pointer_set( astn... ) ));
    map_type m;
    auto x = ast_ptrset::extract_ptrset(
	ast0, extract_pointer_set_with_extend( std::forward<map_type>( m ),
					       astn... ) );
    static_assert( std::is_same_v<std::remove_reference_t<decltype(x)>,map_type>, "oops" );
    return x;
}

template<typename MapTy, typename PtrSet, typename AST>
static constexpr
auto extract_pointer_set_with_into( MapTy & m, const PtrSet & ptrset, AST ast ) {
    using map_type = MapTy;
    // using map_type = decltype(ast_ptrset::extract_ptrset( ast, ptrset ));
    // map_type m;
    initialize_map( m, ptrset.get_entries() );
    return ast_ptrset::extract_ptrset( ast, std::forward<map_type>( m ) );
}

template<typename map_type, typename PtrSet, typename AST0, typename... AST>
static constexpr
auto extract_pointer_set_with_into( map_type & m, const PtrSet & ptrset, AST0 ast0, AST... astn ) {
    // using map_type = decltype(ast_ptrset::extract_ptrset( ast0, extract_pointer_set_with( ptrset, astn... ) ));
    // map_type m;
    initialize_map( m, ptrset.get_entries() );
    return ast_ptrset::extract_ptrset( ast0, extract_pointer_set_with_extend( m, astn... ) );
}

template<typename PtrSet, typename AST>
static constexpr
auto extract_pointer_set_with( const PtrSet & ptrset, AST ast ) {
    using map_type = decltype(ast_ptrset::extract_ptrset( ast, ptrset ));
    map_type m;
    initialize_map( m, ptrset.get_entries() );
    return ast_ptrset::extract_ptrset( ast, std::forward<map_type>( m ) );
}

template<typename PtrSet, typename AST0, typename... AST>
static constexpr
auto extract_pointer_set_with( const PtrSet & ptrset, AST0 ast0, AST... astn ) {
    using map_type = decltype(ast_ptrset::extract_ptrset( ast0, extract_pointer_set_with( ptrset, astn... ) ));
    map_type m;
    initialize_map( m, ptrset.get_entries() );
    return ast_ptrset::extract_ptrset( ast0, extract_pointer_set_with_extend( m, astn... ) );
}

template<typename PtrSet, typename AST0, typename... AST>
static constexpr
void extract_pointer_set_with_new(
    const PtrSet & ptrset, AST0 ast0, AST... astn ) {

    // ast_ptrset::ptrset_list<AST0,AST...>::initialize( ptrset, ast0, astn... );
}

} // namespace expr

#endif // GRAPTOR_DSL_PTRSET_H
