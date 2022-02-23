// -*- c++ -*-
#ifndef GRAPTOR_DSL_EVAL_EVALUATOR_H
#define GRAPTOR_DSL_EVAL_EVALUATOR_H

/**
 * TODO:
 * List of changes required in response to rb::[lr]value mechanism
 *
 * + in load() (all variations): need to distinguish serial load from
 *   gather. The former does not require to know the mask, and so a mask
 *   should not be converted/created in this case.
 *   Alternatively, if a mask needs to be converted/created for a gather,
 *   then it should be retained.
 *
 * + binop_mask will be repeated in ASTs. Need to remove redundancy by
 *   a combination of AST rewriting and priming of the evaluation with an
 *   initial mask.
 * 
 */

namespace expr {

// Evaluation
template<typename value_map_type, typename Cache, typename array_map_type,
	 bool AtomicUpdate>
struct evaluator {
    // static constexpr unsigned short VLS = value_map_type::VLS;
    // static constexpr unsigned short VLD = value_map_type::VLD;
    // using S = typename value_map_type::S;
    // using D = typename value_map_type::D;
    
    GG_INLINE evaluator( Cache &c, const value_map_type & m,
			 const array_map_type & arrays )
	: m_cache( c ), m_vmap( m ), m_arrays( arrays ) { }

    // Cannot return a value for this non-expression
    template<typename MPack>
    static constexpr
    void evaluate( noop n, const MPack & mpack ) { }

    // Generic definition for value kinds stored in value map
    template<typename Tr, value_kind K, typename MPack>
    GG_INLINE
    auto evaluate( const value<Tr,K> & v, const MPack & mpack ) {
	auto rval = m_vmap.template get<K>();
	static_assert( decltype(rval)::VL == Tr::VL, "VL match on value" );
	return make_rvalue( rval, mpack );
    }
    
    template<typename Tr, typename MPack>
    GG_INLINE
    auto
    evaluate( const value<Tr,vk_smk> & v, const MPack & mpack ) {
	return make_rvalue( m_vmap.template get<vk_smk>().asmask(), mpack );
    }
    template<typename Tr, typename MPack>
    __attribute__((always_inline))
    auto evaluate( const value<Tr,vk_dmk> & v, const MPack & mpack ) {
	return make_rvalue( m_vmap.destination_mask(), mpack );
    }
    template<typename Tr, typename MPack>
    __attribute__((always_inline))
    auto evaluate( const value<Tr,vk_true> & v, const MPack & mpack ) {
	return make_rvalue( simd::detail::vector_impl<Tr>::true_mask(), mpack );
    }
    template<typename Tr, typename MPack>
    __attribute__((always_inline))
    auto evaluate( const value<Tr,vk_false> & v, const MPack & mpack ) {
	return make_rvalue( simd::detail::vector_impl<Tr>::false_mask(), mpack );
    }
    template<typename Tr, typename MPack>
    GG_INLINE auto
    evaluate( const value<Tr,vk_any> & v, const MPack & mpack ) {
	return make_rvalue( simd::detail::vector_impl<Tr>( v.data() ), mpack );
    }
    template<typename Tr, typename MPack>
    GG_INLINE auto
    evaluate( const value<Tr,vk_cstone> & v, const MPack & mpack ) {
	auto rval = simd::detail::vector_impl<Tr>::one_val();
	return make_rvalue( rval, mpack );
    }
    template<typename Tr, typename MPack>
    GG_INLINE auto
    evaluate( const value<Tr,vk_inc> & v, const MPack & mpack ) {
	auto vec = simd::detail::vector_impl<Tr>::s_set1inc0();
	return make_rvalue( vec, mpack );
    }
    template<typename Tr, typename MPack>
    GG_INLINE auto
    evaluate( const value<Tr,vk_truemask> & v, const MPack & mpack ) {
	return make_rvalue( simd::detail::vector_impl<Tr>::allones_val(), mpack );
    }
    template<typename Tr, typename MPack>
    GG_INLINE auto
    evaluate( const value<Tr,vk_zero> & v, const MPack & mpack ) {
	return make_rvalue( simd::detail::vector_impl<Tr>::zero_val(), mpack );
    }
    template<typename Tr, typename MPack>
    __attribute__((always_inline))
    auto evaluate( const value<Tr,vk_vid> & v, const MPack & mpack ) {
	auto val = m_vmap.template get<vk_vid>();
	static_assert( Tr::VL == decltype(val)::VL, "vector lengths must match" );
	return make_rvalue( val, mpack );
    }
    template<typename Tr, typename MPack>
    __attribute__((always_inline))
    auto evaluate( const value<Tr,vk_pid> & v, const MPack & mpack ) {
	auto val = m_vmap.template get<vk_pid>();
	static_assert( Tr::VL == decltype(val)::VL, "vector lengths must match" );
	return make_rvalue( val, mpack );
    }

    template<typename Expr, typename UnOp, typename MPack>
    GG_INLINE inline auto evaluate( const unop<Expr,UnOp> & uop,
				    const MPack & mpack ) {
	auto r = evaluate( uop.data(), mpack );
	return UnOp::evaluate( r.uvalue(), r.mpack() );
    }

    template<typename E1, typename E2, typename BinOp, typename MPack>
    GG_INLINE inline
    // __attribute__((noinline))
    auto evaluate( const binop<E1,E2,BinOp> & bop, const MPack & mpack ) {
	if constexpr ( std::is_same_v<BinOp,binop_setmask> ) {
	    // binop_setmask is evaluated inline because it alters the
	    // deviates from the normal evaluation order and mask pack passing.
	    // Evaluate LHS and merge with the mask pack using logical and.
	    // The resulting value becomes the new mask pack.
	    // Evaluate RHS using defined mask pack.
	    auto arg1 = evaluate( bop.data1(), mpack );
	    auto mask0 = arg1.mpack().get_mask_for( arg1.value() );
	    auto mask = mask0 && arg1.value();
	    auto mpack2 = sb::create_mask_pack( mask );
	    auto arg2 = evaluate( bop.data2(), mpack2 );
	    return arg2;
	} else
	/*if constexpr ( std::is_same_v<BinOp,binop_mask> ) {
	    // ignore binop_mask
	    return evaluate( bop.data1(), mpack );
	    } else*/ {
	    // Force evaluation order for lazy execution and binop_seq
	    auto arg1 = evaluate( bop.data1(), mpack );
	    auto arg2 = evaluate( bop.data2(), arg1.mpack() );
	    return BinOp::evaluate(
		arg1.uvalue(), arg2.uvalue(), arg2.mpack() );
	}
    }

    template<typename E1, typename E2, typename E3, typename TernOp,
	     typename MPack>
    GG_INLINE inline auto evaluate( const ternop<E1,E2,E3,TernOp> & top,
				    const MPack & mpack ) {
	auto arg1 = evaluate( top.data1(), mpack );
	auto arg2 = evaluate( top.data2(), arg1.mpack() );
	auto arg3 = evaluate( top.data3(), arg2.mpack() );
	return TernOp::evaluate( arg1.uvalue(), arg2.uvalue(), arg3.uvalue(),
				 arg3.mpack() );
    }

    template<typename VTr, layout_t Layout, short AID, typename T,
	     typename Enc, bool NT>
    GG_INLINE
    auto levaluate( const array_intl<T,typename VTr::member_type,AID,Enc,NT> & array,
		    sb::rvalue<VTr,Layout> idx ) {
	using ATr = simd::detail::vdata_traits<T,VTr::VL>;
	return 
	    simd::create_vector_ref_vec<
	    ATr,typename VTr::member_type,Enc,NT,Layout>(
		get_ptr( array ), idx.value() );
    }

    template<unsigned short VL, typename T, typename VTr, short AID,
	     typename Enc, bool NT, layout_t Layout>
    GG_INLINE
    auto levaluate_incseq( const array_intl<T,typename VTr::member_type,AID,Enc,NT> & array,
			   sb::rvalue<VTr,Layout> idx ) {
	static_assert( VTr::VL == 1, "incseq requirement" );
	using ATr = simd::ty<T,VL>;
	return simd::template create_vector_ref_cacheop<
	    ATr,typename VTr::member_type,Enc,NT>(
		get_ptr( array ), idx.value().data() );
    }

    template<unsigned short VL, typename T, typename VTr, short AID,
	     layout_t Layout>
    GG_INLINE
    auto levaluate_incseq( const bitarray_intl<T,typename VTr::member_type,AID> & array,
			   sb::rvalue<VTr,Layout> idx ) {
	static_assert( VTr::VL == 1, "incseq requirement" );
	static_assert( VL >= 8, "coding scheme" );
/*
	using Ty = int_type_of_size_t<VL/8>;
	using Enc = array_encoding<Ty>;
	using ATr = simd::detail::vdata_traits<Ty,1>;
	using STy = typename Enc::storage_type;
	return make_lvalue(
	    simd::template create_vector_ref_scalar<
	    ATr,typename VTr::member_type,Enc,false,lo_linalgn>(
		reinterpret_cast<STy *>( get_ptr( array ) ),
		idx.value().data() ) );
*/
	using ATr = simd::detail::mask_bit_traits<VL>;
	auto lv = simd::template create_vector_ref_cacheop<
	    ATr, typename VTr::member_type, array_encoding_bit<1>, false>(
		reinterpret_cast<typename ATr::pointer_type *>(
		    get_ptr( array ) ),
		idx.value().data() );
	return lv;
    }

    template<unsigned short VL, typename T, typename VTr, short AID,
	     typename Enc, bool NT, layout_t Layout>
    GG_INLINE
    auto levaluate_broadcast( const array_intl<T,typename VTr::member_type,AID,Enc,NT> & array,
			      sb::rvalue<VTr,Layout> idx ) {
	static_assert( VTr::VL == 1, "broadcast requirement" );
	using ATr = simd::detail::vdata_traits<T,VL>;
	return 
	    simd::template create_vector_ref_scalar<ATr,typename VTr::member_type,Enc,NT,lo_constant>(
		get_ptr( array ), idx.value().data() );
    }

    template<typename T, typename VTr, short AID,
	     typename Enc, bool NT, layout_t Layout, typename... MTr>
    GG_INLINE
    auto evaluate( const array_intl<T,typename VTr::member_type,AID,Enc,NT> & array,
		   sb::rvalue<VTr,Layout,MTr...> idx ) {
	auto lv = levaluate( array, idx.uvalue() );
	auto rv = lv.load( idx.mpack().template get_mask<VTr>() );
	return make_rvalue( rv, idx.mpack() );
    }

    template<typename T, typename VTr, short AID, typename Enc, bool NT, layout_t Layout>
    GG_INLINE
    auto evaluate( const array_intl<T,typename VTr::member_type,AID,Enc,NT> & array,
		   rvalue<VTr,Layout,simd::detail::mask_bool_traits> idx ) {
	using ATr = simd::detail::vdata_traits<T,VTr::VL>;
	auto ref = simd::template create_vector_ref_vec<
	    ATr,typename VTr::member_type,Enc,NT,Layout>(
		get_ptr( array ), idx.value() );
	return make_rvalue( ref.load( idx.mask() ), idx.mask() );
    }

    template<typename T, typename VTr, short AID, typename Enc, bool NT, layout_t Layout>
    GG_INLINE
    auto evaluate( const array_intl<T,typename VTr::member_type,AID,Enc,NT> & array,
		   rvalue<VTr,Layout,void> idx ) {
	using ATr = simd::detail::vdata_traits<T,VTr::VL>;
	// simd::detail::vector_ref_impl<ATr,typename VTr::member_type,Enc,NT> ref(
	// get_ptr( array ), idx.value() );
	auto ref = simd::create_vector_ref_vec<
	    ATr,typename VTr::member_type,Enc,NT,Layout>(
		get_ptr( array ), idx.value() );
	return make_rvalue( ref.load() );
    }

    template<unsigned short VL, typename T, typename VTr, short AID,
	     typename Enc, bool NT, layout_t Layout>
    GG_INLINE
    auto evaluate_incseq( array_intl<T,typename VTr::member_type,AID,Enc,NT> array,
			  sb::rvalue<VTr,Layout> idx ) {
	static_assert( VTr::VL == 1, "incseq requirement" );
	using ATr = simd::detail::vdata_traits<T,VL>;
	auto lv = simd::template
	    create_vector_ref_scalar<
		ATr,typename VTr::member_type,Enc,NT,lo_linalgn>(
		    get_ptr( array ), idx.value().data() );
	// Ignore the mask in index as this is a linear-align fetch, which
	// should always have a valid base pointer
	auto rv = lv.load();
	return make_rvalue( rv, idx.mpack() );
    }

    template<unsigned short VL, typename T, typename VTr, short AID,
	     typename Enc,  bool NT, layout_t Layout>
    GG_INLINE
    auto evaluate_broadcast( array_intl<T,typename VTr::member_type,AID,Enc,NT> array,
			     rvalue<VTr,Layout,void> idx ) {
	static_assert( VTr::VL == 1, "broadcast requirement" );
	using ATr = simd::detail::vdata_traits<T,VL>;
	// simd::detail::vector_ref_impl<ATr,typename VTr::member_type,Enc,NT> ref(
	// get_ptr( array ), idx.value(), lo_constant );
	auto ref = simd::template create_vector_ref_scalar<ATr,typename VTr::member_type,Enc,NT,lo_constant>(
	    get_ptr( array ), idx.value().data() );
	return make_rvalue( ref.load() );
    }

    // bitarray
    template<typename T, short AID, typename VTr, layout_t Layout>
    GG_INLINE
    auto levaluate( const bitarray_intl<T,typename VTr::member_type,AID> & array,
		    sb::rvalue<VTr,Layout> idx ) {
	return 
	    simd::detail::vector_ref_impl<simd::detail::mask_bit_traits<VTr::VL>,typename VTr::member_type,array_encoding<T>,false,lo_unknown>(
		reinterpret_cast<typename simd::detail::mask_bit_traits<VTr::VL>::pointer_type *>( get_ptr( array ) ), idx.value().data(),
		idx.value().get_layout() );
    }
    template<typename T, typename VTr, short AID, layout_t Layout>
    GG_INLINE
    auto levaluate( const bitarray_intl<T,typename VTr::member_type,AID> & array,
		    sb::rvalue<VTr,Layout> idx ) {
	static_assert( sizeof(T)*8 == VTr::VL || VTr::VL == 1, "VL check" );
	return 
	    simd::detail::vector_ref_impl<simd::detail::mask_bit_traits<VTr::VL>,typename VTr::member_type,array_encoding<T>,false,lo_unknown>(
		reinterpret_cast<typename simd::detail::mask_bit_traits<VTr::VL>::pointer_type *>( get_ptr( array ) ), idx.value() );
    }

    template<typename T, typename VTr, short AID, layout_t Layout,
	     typename... MTr>
    GG_INLINE
    auto evaluate( const bitarray_intl<T,typename VTr::member_type,AID> & array,
		   sb::rvalue<VTr,Layout,MTr...> idx ) {
	simd::detail::mask_ref_impl<simd::detail::mask_bit_traits<VTr::VL>,
				    typename VTr::member_type> ref(
	    get_ptr( array ), idx.value() );
	auto v = ref.load( idx.mpack().template get_mask<VTr>() );
	return make_rvalue( v, idx.mpack() );
    }

    template<typename T, typename VTr, short AID, layout_t Layout>
    GG_INLINE
    auto evaluate( const bitarray_intl<T,typename VTr::member_type,AID> & array,
		   rvalue<VTr,Layout,void> idx ) {
	simd::detail::mask_ref_impl<simd::detail::mask_bit_traits<VTr::VL>,
				    typename VTr::member_type> ref(
					get_ptr( array ), idx.value() );
	return make_rvalue( ref.load() );
    }
    template<unsigned short VL, typename T, typename VTr, short AID,
	     layout_t Layout>
    GG_INLINE
    auto evaluate_incseq( bitarray_intl<T,typename VTr::member_type,AID> array,
			  sb::rvalue<VTr,Layout> idx ) {
	static_assert( VTr::VL == 1, "incseq requirement" );
	simd::detail::mask_ref_impl<simd::detail::mask_bit_traits<VL>,
				    typename VTr::member_type>
	    ref( get_ptr( array ), idx.value().data() );
	return make_rvalue( ref.load(), idx.mpack() );
    }
    // end bitarray
    
    template<typename A, typename T, unsigned short VL, typename MPack>
    GG_INLINE auto evaluate( const refop<A,T,VL> & op, const MPack & mpack ) {
	auto idx = evaluate( op.index(), mpack );
	static_assert( decltype(idx.value())::VL == VL, "VL match" );
	return evaluate( op.array(), idx );
    }

    template<typename A, typename T, unsigned short VL, typename MPack>
    GG_INLINE auto evaluate( const refop<A,unop<T,unop_incseq<VL>>,VL> & op,
	const MPack & mpack ) {
	// Don't evaluate unop_incseq, just pick up the data
	auto idx = evaluate( op.index().data(), mpack );
	static_assert( decltype(idx.value())::VL == 1, "VL match" );
	// Any mask applied to index has been set prior to VL width; we need
	// to assume here that the base pointer is always valid in a linear
	// fetch, such that we can fetch unconditionally
	auto rv = evaluate_incseq<VL>( op.array(), idx.uvalue() );
	return make_rvalue( rv.value(), idx.mpack() );
    }

    template<typename A, typename T, unsigned short VL>
    GG_INLINE auto evaluate( const refop<A,unop<T,unop_broadcast<VL>>,VL> & op ) {
	// Don't evaluate unop_broadcast, just pick up the data
	auto idx = evaluate( op.index().data() );
	static_assert( decltype(idx.value())::VL == 1, "VL match" );
	return evaluate_broadcast<VL>( op.array(), idx );
    }

    template<typename A, typename T, typename M, unsigned short VL,
	     typename MPack>
    GG_INLINE
    auto levaluate( const maskrefop<A,T,M,VL> & op, const MPack & mpack ) {
	auto idx = evaluate( op.index(), mpack );
	auto msk = evaluate( op.mask(), idx.mpack() );
	auto mpack2 = binop_mask::set_mask( msk.uvalue(), msk.mpack() );
	return make_lvalue( levaluate_incseq<VL>( op.array(), idx.uvalue() ),
			    mpack2 );
    }

    template<typename A, typename T, typename M, unsigned short VL,
	     typename MPack>
    GG_INLINE
    auto levaluate( const maskrefop<A,unop<T,unop_incseq<VL>>,M,VL> & op,
		    const MPack & mpack ) {
	// A scalar-index (incseq) load with a mask is performed unconditionally
	// and the mask is inserted on the loaded value
	// The scalar is unconditionally true. Beware for memory accesses
	// performed within the index expression.
	auto idx = evaluate( op.index().data(), mpack );
	auto msk = evaluate( op.mask(), idx.mpack() );
	// auto lval = levaluate_incseq<VL>( op.array(), idx );
	// return make_lvalue( lval.value(), msk.mask() );

	auto mpack2 = binop_mask::set_mask( msk.uvalue(), msk.mpack() );
	return make_lvalue(
	    levaluate_incseq<VL>( op.array(), idx.uvalue() ), mpack2 );
    }

    template<typename A, typename T, unsigned short VL, typename MPack>
    GG_INLINE
    auto levaluate( const refop<A,T,VL> & op, const MPack & mpack ) {
	// We don't expect a mask here - if a mask was present, the refop
	// should have been converted to a maskrefop
	auto idx = evaluate( op.index(), mpack );
	return make_lvalue( levaluate( op.array(), idx.uvalue() ),
			    idx.mpack() );
    }

    template<typename A, typename T, unsigned short VL, typename MPack>
    GG_INLINE
    auto levaluate( const refop<A,unop<T,unop_incseq<VL>>,VL> & op,
		    const MPack & mpack ) {
	// We don't expect a mask here - if a mask was present, the refop
	// should have been converted to a maskrefop
	// Don't evaluate unop_incseq, just pick up the data
	auto idx = evaluate( op.index().data(), mpack );
	return make_lvalue( levaluate_incseq<VL>( op.array(), idx.uvalue() ),
			    idx.mpack() );
    }

    template<typename A, typename T, unsigned short VL>
    GG_INLINE
    auto levaluate( const refop<A,unop<T,unop_broadcast<VL>>,VL> & op ) {
	// We don't expect a mask here - if a mask was present, the refop
	// should have been converted to a maskrefop
	// Don't evaluate unop_broadcast, just pick up the data
	auto idx = evaluate( op.index().data() );
	return make_lvalue( levaluate_broadcast<VL>( op.array(), idx.uvalue() ),
			    idx.mpack() );
    }

    template<typename A, typename T, typename M, unsigned short VL,
	     typename MPack>
    GG_INLINE
    auto evaluate( const maskrefop<A,T,M,VL> & op, const MPack & mpack ) {
	auto i = evaluate( op.index(), mpack );
	auto m = evaluate( op.mask(), i.mpack() );
	return evaluate( op.array(), make_rvalue( i.value(), m.mpack() ) );
    }

    template<unsigned cid, typename Tr, typename MPack>
    GG_INLINE
    auto evaluate( const cacheop<cid,Tr> & op, const MPack & mpack ) {
	// Note: the cached values are sized to meet the vector length
	//       of the computation
	auto val = m_cache.template get<cid>();
	static_assert( Tr::VL == decltype(val)::VL, "VL check" );
	return make_rvalue( val, mpack );
    }

    template<unsigned cid, typename Tr, typename MPack>
    GG_INLINE
    auto levaluate( const cacheop<cid,Tr> & op, const MPack & mpack ) {
	// Note: vector length must always be VLS in the supported use cases:
	//       1) VLS>1 and VLD=1
	//       2) VLS=VLD
	// static_assert( Tr::VL == VLS, "sanity" );
	using index_type = typename Tr::index_type;
	auto lv = simd::template create_vector_ref_cacheop<
	    Tr, index_type, array_encoding<typename Tr::member_type>,false>(
		&m_cache.template get<cid>(), index_type(0) );
	return make_lvalue( lv, mpack );
    }

    template<unsigned cid, typename Tr, typename M, typename MPack>
    GG_INLINE
    auto levaluate( const binop<cacheop<cid,Tr>,M,binop_predicate> & op,
		    const MPack & mpack ) {
	auto mask = evaluate( op.data2(), mpack );
	using index_type = typename Tr::index_type;
	auto lv = simd::template create_vector_ref_cacheop<
	    Tr, index_type, array_encoding<typename Tr::member_type>,
	    false, lo_unknown>(
		&m_cache.template get<cid>(), index_type(0) );
	return make_lvalue( lv, mask.value(), mask.mpack() );
    }

    template<unsigned cid, typename Tr, typename M, typename MPack>
    GG_INLINE
    auto levaluate( const binop<cacheop<cid,Tr>,M,binop_mask> & op,
		    const MPack & mpack ) {
	// ignore mask entirely -- not
	auto mask = evaluate( op.data2(), mpack );
	using index_type = typename Tr::index_type;
	auto lv = simd::template create_vector_ref_cacheop<
	    Tr, index_type, array_encoding<typename Tr::member_type>,
	    false, lo_unknown>(
		&m_cache.template get<cid>(), index_type(0) );
	return make_lvalue( lv, mask.mpack() );
	// return make_lvalue( lv, mpack );
    }

    template<bool nt, typename R, typename T, typename MPack>
    GG_INLINE inline
    auto evaluate( const storeop<nt,R,T> & op, const MPack & mpack ) {
	auto p = levaluate( op.ref(), mpack );
	auto v = evaluate( op.value(), p.mpack() );
	return storeop<nt,R,T>::evaluate( p.uvalue(), v.uvalue(), v.mpack() );
    }

    template<bool nt, typename R, typename T, typename M, typename MPack>
    GG_INLINE inline
    auto evaluate( const storeop<nt,R,binop<T,M,binop_predicate>> & op,
		   const MPack & mpack ) {
	auto p = levaluate( op.ref(), mpack );
	auto v = evaluate( op.value().data1(), p.mpack() );
	auto m = evaluate( op.value().data2(), v.mpack() );
	auto pred = binop_mask::update_mask( m );
	auto tmppack = sb::create_mask_pack( pred.value() );
	auto r = storeop<nt,R,T>::evaluate( p.uvalue(), v.uvalue(), tmppack );
	return make_rvalue( r.value(), m.mpack() );
    }

    template<typename E1, typename E2, typename RedOp, typename MPack>
    GG_INLINE
    auto evaluate( const redop<E1,E2,RedOp> & op, const MPack & mpack ) {
	// TODO: do not use levaluate here for the ref when the ref is
	//       a cacheop. Use vector assignment 
	//       rather than traits::store or traits::scatter
	auto ref = levaluate( op.ref(), mpack );
	auto val = evaluate( op.val(), ref.mpack() );
	if constexpr ( AtomicUpdate )
	    return RedOp::evaluate_atomic( ref.uvalue(), val.uvalue(),
					   val.mpack() );
	else
	    return RedOp::evaluate( ref.uvalue(), val.uvalue(), val.mpack() );
    }

    template<typename E1, typename E2, typename C,
	     typename RedOp, typename MPack>
    GG_INLINE
    auto evaluate( const redop<E1,binop<E2,C,binop_predicate>,RedOp> & op,
		   const MPack & mpack ) {
	// TODO: do not use levaluate here for the ref when the ref is
	//       a cacheop. Use vector assignment 
	//       rather than traits::store or traits::scatter
	auto ref = levaluate( op.ref(), mpack );
	auto val = evaluate( op.val().data1(), ref.mpack() );
	auto mask = evaluate( op.val().data2(), val.mpack() );
	auto cond = binop_mask::update_mask( mask );
	auto tmppack = sb::create_mask_pack( cond.value() );
	
	if constexpr ( AtomicUpdate ) {
	    auto r = RedOp::evaluate_atomic( ref.uvalue(), val.uvalue(),
					     tmppack );
	    return make_rvalue( r.value(), cond.mpack() );
	} else {
	    auto r = RedOp::evaluate( ref.uvalue(), val.uvalue(), tmppack );
	    return make_rvalue( r.value(), cond.mpack() );
	}
    }
    
    template<unsigned cid, typename TrC, typename E2,
	     typename RedOp, typename MPack>
    GG_INLINE
    auto evaluate( const redop<cacheop<cid,TrC>,E2,RedOp> & op,
		   const MPack & mpack ) {
	// Special case: avoid store operations such that the cached value
	// can live in registers
	auto i = evaluate( op.val(), mpack );
	auto& r = m_cache.template get<cid>(); // reference into cache
	static_assert( decltype(i)::VL == 1 ||
		       decltype(i)::VL == std::remove_reference<decltype(r)>::type::VL,
		       "VL match" ); // we may have long vector with scalar access
	// bitarray vs array cases
	if constexpr ( std::is_void_v<typename TrC::member_type> ) {
	    using Tr = TrC;

	    using U = index_type_of_size<sizeof(VID)>;

	    auto vr = simd::template create_vector_ref_scalar<Tr,U,array_encoding<typename Tr::element_type>,false,lo_linalgn>(
		reinterpret_cast<typename Tr::pointer_type *>( &r.data() ), U(0) );
	    auto l = make_lvalue( vr, i.mpack() );
	    return RedOp::evaluate( l.uvalue(), i.uvalue(), i.mpack() );
	} else {
	    using Tr = typename TrC::template rebindVL<decltype(i)::VL>::type;

	    // Cut of at too wide widths
	    // 4-byte int is more likely index supported in gather than
	    // 1 or 2-byte int
	    using U = index_type_of_size<std::max(Tr::W,(unsigned short)4)>;

	    auto vr = simd::template create_vector_ref_scalar<Tr,U,array_encoding<typename Tr::member_type>,false,lo_linalgn>(
		reinterpret_cast<typename Tr::pointer_type *>( &r.data() ), U(0) );

	    auto l = make_lvalue( vr, i.mpack() );
	    static_assert( sizeof(typename decltype(l.value())::index_type) != 1, "err" );
	    return RedOp::evaluate( l.uvalue(), i.uvalue(), i.mpack() );
	}
    }
	
    template<unsigned cid, typename TrC, typename E2, typename C,
	     typename RedOp, typename MPack>
    GG_INLINE
    auto evaluate( const redop<cacheop<cid,TrC>,binop<E2,C,binop_predicate>,RedOp> & op,
		   const MPack & mpack ) {
	// Special case: avoid store operations such that the cached value
	// can live in registers
	auto i = evaluate( op.val().data1(), mpack );
	auto& r = m_cache.template get<cid>(); // reference into cache
	static_assert( decltype(i)::VL == 1 ||
		       decltype(i)::VL == std::remove_reference<decltype(r)>::type::VL,
		       "VL match" ); // we may have long vector with scalar access

	auto mask = evaluate( op.val().data2(), i.mpack() );
	auto cond = binop_mask::update_mask( mask );
	auto tmppack = sb::create_mask_pack( cond.value() );

	// bitarray vs array cases
	if constexpr ( std::is_void_v<typename TrC::member_type> ) {
	    using Tr = TrC;

	    using U = index_type_of_size<sizeof(VID)>;

	    auto vr = simd::template create_vector_ref_scalar<Tr,U,array_encoding<typename Tr::element_type>,false,lo_linalgn>(
		reinterpret_cast<typename Tr::pointer_type *>( &r.data() ), U(0) );
	    auto l = make_lvalue( vr, i.mpack() );
	    auto r = RedOp::evaluate( l.uvalue(), i.uvalue(), tmppack );
	    return make_rvalue( r.value(), cond.mpack() );
	} else {
	    using Tr = typename TrC::template rebindVL<decltype(i)::VL>::type;

	    // Cut of at too wide widths
	    // 4-byte int is more likely index supported in gather than
	    // 1 or 2-byte int
	    using U = index_type_of_size<std::max(Tr::W,(unsigned short)4)>;

	    auto vr = simd::template create_vector_ref_scalar<Tr,U,array_encoding<typename Tr::member_type>,false,lo_linalgn>(
		reinterpret_cast<typename Tr::pointer_type *>( &r.data() ), U(0) );

	    auto l = make_lvalue( vr, i.mpack() );
	    static_assert( sizeof(typename decltype(l.value())::index_type) != 1, "err" );
	    auto r = RedOp::evaluate( l.uvalue(), i.uvalue(), tmppack );
	    return make_rvalue( r.value(), cond.mpack() );
	}
    }

    template<typename S, typename U, typename C, typename DFSAOp,
	     typename MPack>
    GG_INLINE
    auto evaluate( const dfsaop<S,U,C,DFSAOp> & op, const MPack & mpack ) {
	auto s = levaluate( op.state(), mpack );
	auto u = evaluate( op.update(), s.mpack() );
	auto c = evaluate( op.condition(), u.mpack() );
	if constexpr ( AtomicUpdate )
	    return DFSAOp::evaluate_atomic( s.uvalue(), u.uvalue(), c.uvalue(),
					    c.mpack() );
	else
	    return DFSAOp::evaluate( s.uvalue(), u.uvalue(), c.uvalue(),
				     c.mpack() );
    }

    template<typename C, typename B, typename F,
	     typename MPack>
    GG_INLINE
    auto evaluate( const ternop<C,B,F,loopop_while> & op,
		   const MPack & mpack ) {
	// Iterate
	evaluate_loop( op, mpack );

	// Loop terminated; return with final value
	return evaluate( op.data3(), mpack );
    }
    
    template<typename C, typename B, typename F,
	     typename MPack>
    void evaluate_loop( const ternop<C,B,F,loopop_while> & op,
			const MPack & mpack ) {
	// Note: leaves mask_pack unmodified.
	    
	// Evaluate condition
	auto cnd = evaluate( op.data1(), mpack ).value();

	// Build mask
	auto m0 = mpack.get_mask_for( cnd );

	// Disable terminated lanes
	auto m = m0 && cnd;
	
	if( !m.is_all_false() ) {
	    // Create mask_pack with terminated lanes disabled
	    auto mpack2 = sb::create_mask_pack( m );

	    // Execute loop body
	    auto b = evaluate( op.data2(), mpack2 );

	    // Continue loop using tail recursion
	    evaluate_loop( op, b.mpack() );
	}
    }
    
private:
    template<array_aid AID>
    auto get_ptr() const {
	return m_arrays.template get<(unsigned)aid_key(AID)>();
    }

    template<typename T, typename U, short AID, typename Enc, bool NT>
    auto get_ptr( const array_intl<T,U,AID,Enc,NT> & a ) const {
	return get_ptr<array_aid(AID)>();
    }

    template<typename T, typename U, short AID>
    auto get_ptr( const bitarray_intl<T,U,AID> & a ) const {
	return get_ptr<array_aid(AID)>();
    }

private:
    Cache &m_cache;    // updated values for reductions
    const value_map_type & m_vmap; // read-only values
    const array_map_type & m_arrays; // unique (un-aliased by AID) array addresses

// printer<S, VLS, D, VLD, Cache> m_pr;
};

template<bool AtomicUpdate, typename Cache, typename value_map_type,
	 typename Expr>
__attribute__((always_inline))
inline auto evaluate( Cache &c, const value_map_type & m, Expr e ) {
    auto array_map = extract_pointer_set( e );
    return expr::evaluator<value_map_type, Cache, decltype(array_map),
			   AtomicUpdate>( c, m, array_map )
	.evaluate( e, sb::mask_pack<>() );
}

template<typename Cache, typename value_map_type,
	 typename Expr>
__attribute__((always_inline))
static inline auto evaluate( Cache &c, const value_map_type & m, Expr e ) {
    auto array_map = extract_pointer_set( e );
    return expr::evaluator<value_map_type, Cache, decltype(array_map), false>(
	c, m, array_map ).evaluate( e );
}

template<typename Cache, typename ValueMap, typename Expr>
// GG_INLINE inline
__attribute__((noinline))
bool evaluate_bool( Cache & c, const ValueMap & m, Expr expr ) {
    // Check all lanes are active.
    // Type-check on the expression to see if it is constant true.
    // If so, omit the check.
    if constexpr ( expr::is_constant_true<Expr>::value )
	return true;
    else {
	auto r = expr::evaluate( c, m, expr );
	static_assert( decltype(r)::VL == 1, "expect 1D vector or kmask" );
	return expr::is_true( r );
    }
}

template<typename Cache, typename ValueMap, typename Expr>
GG_INLINE inline
bool evaluate_bool_any( Cache & c, ValueMap & m, Expr expr ) {
    // Check all lanes are active.
    // Type-check on the expression to see if it is constant true.
    // If so, omit the check.
    if constexpr ( expr::is_constant_true<Expr>::value )
	return true;
    else {
	auto r = expr::evaluate( c, m, expr );
	static_assert( decltype(r)::VL == 1, "expect 1D vector or kmask" );
	return !expr::is_false( r );
    }
}

template<typename Cache, typename Output, typename Expr, unsigned short VL>
__attribute__((always_inline))
inline void evaluate_post( Cache & c, Output output, simd_vector<VID,VL> dst, simd_vector<VID,VL> pid, Expr & e ) {
    auto dmk = simd_vector<VID,VL>::true_mask();
    simd_vector<VID,1> src; // unused!
    // static_assert( sizeof(typename Output::type) > 1, "sz" );
    // TODO: value in licm is licm.cop() -> need to set/input value
    auto m = expr::create_value_map_new<VL>(
	expr::create_entry<expr::vk_dst>( dst ),
	expr::create_entry<expr::vk_smk>( output ),
	expr::create_entry<expr::vk_pid>( pid ) );
    expr::evaluate( c, m, e );
}

template<typename Cache, typename Output, typename Expr, unsigned short VL>
GG_INLINE
inline
void evaluate_post( Cache & c, Output output, simd_vector<VID,VL> dst, Expr & e ) {
    auto dmk = simd_vector<VID,VL>::true_mask();
    simd_vector<VID,1> src; // unused!
    // TODO: value in licm is licm.cop() -> need to set/input value
    auto m = expr::create_value_map_new<VL>(
	expr::create_entry<expr::vk_dst>( dst ),
	expr::create_entry<expr::vk_smk>( output ) );
    expr::evaluate( c, m, e );
}

template<typename Cache, typename Output, unsigned short VL>
GG_INLINE
inline
void evaluate_post( Cache & c, Output & output, simd_vector<VID,VL> dst, simd_vector<VID,VL> pid, noop e ) { }

template<typename Cache, typename Output, unsigned short VL>
GG_INLINE
inline
void evaluate_post( Cache & c, Output & output, simd_vector<VID,VL> dst, noop e ) { }

    
} // namespace expr

#endif // GRAPTOR_DSL_EVAL_EVALUATOR_H
