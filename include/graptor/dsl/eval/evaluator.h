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
	 typename Accum, bool AtomicUpdate>
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
	    // binop_setmask is evaluated inline because it
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
	} else if constexpr ( std::is_same_v<BinOp,binop_seq> ) {
	    // In a sequence, do not pass the mask pack from one command to
	    // the other, because it is possible that some lanes in the first
	    // command are not updated, but they are updated in the second
	    // command and vice versa.
	    // The downside is that a conversion of a mask may occur once
	    // for every command in the sequence.
	    // Evaluate first argument and return second argument.
	    if constexpr ( E1::VL == 1 && !MPack::is_empty() ) {
		// In case of scalar execution with a mask, do a short-circuit
		// evaluation of the mask just once.
		if( mpack.template get_mask<typename E1::data_type>().data() ) {
		    auto empty = sb::mask_pack<>();
		    evaluate( bop.data1(), empty );
		    auto arg2 = evaluate( bop.data2(), empty );
		    return make_rvalue( arg2.value(), mpack );
		} else {
		    // Copy the layout of the previous branch
		    using ret_type = decltype( evaluate( bop.data2(),
							 sb::mask_pack<>() ) );
		    constexpr simd::layout_t layout
				  = ret_type::value_type::layout;
		    return make_rvalue(
			simd::vec<typename E2::data_type,layout>(
			    E2::data_type::traits::setzero() ),
			mpack );
		}
	    } else {
		evaluate( bop.data1(), mpack );
		return evaluate( bop.data2(), mpack );
	    }
	} else {
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

    template<unsigned short VL, bool aligned,
	     typename T, typename VTr, short AID, typename Enc, bool NT,
	     layout_t Layout>
    GG_INLINE
    auto levaluate_incseq(
	const array_intl<T,typename VTr::member_type,AID,Enc,NT> & array,
	sb::rvalue<VTr,Layout> idx ) {
	static_assert( VTr::VL == 1, "incseq requirement" );
	static_assert( aligned, "default value; extension not checked" );
	using ATr = simd::ty<T,VL>;
	return simd::template create_vector_ref_cacheop<
	    ATr,typename VTr::member_type,Enc,NT>(
		get_ptr( array ), idx.value().data() );
    }

    template<unsigned short VL, bool aligned,
	     typename T, typename VTr, short AID, layout_t Layout>
    GG_INLINE
    auto levaluate_incseq(
	const bitarray_intl<T,typename VTr::member_type,AID> & array,
	sb::rvalue<VTr,Layout> idx ) {
	static_assert( VTr::VL == 1, "incseq requirement" );
	static_assert( VL >= 8, "coding scheme" );
	static_assert( aligned, "default value; extension not checked" );
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

#if 0
    template<typename T, typename VTr, short AID, typename Enc, bool NT, layout_t Layout>
    [[deprecated("rvalue replaced by sb::rvalue")]]
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
    [[deprecated("rvalue replaced by sb::rvalue")]]
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
#endif

    template<unsigned short VL, bool aligned,
	     typename T, typename VTr, short AID, typename Enc, bool NT,
	     layout_t Layout>
    GG_INLINE
    auto evaluate_incseq( array_intl<T,typename VTr::member_type,AID,Enc,NT> array,
			  sb::rvalue<VTr,Layout> idx ) {
	static_assert( VTr::VL == 1, "incseq requirement" );
	using ATr = simd::detail::vdata_traits<T,VL>;
	auto lv = simd::template
	    create_vector_ref_scalar<
		ATr,typename VTr::member_type,Enc,NT,
		aligned ? lo_linalgn : lo_linear>(
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
#if 0
    template<typename T, short AID, typename VTr, layout_t Layout>
    GG_INLINE
    auto levaluate( const bitarray_intl<T,typename VTr::member_type,AID> & array,
		    sb::rvalue<VTr,Layout> idx ) {
	return 
	    simd::detail::vector_ref_impl<simd::detail::mask_bit_traits<VTr::VL>,typename VTr::member_type,array_encoding<T>,false,lo_unknown>(
		reinterpret_cast<typename simd::detail::mask_bit_traits<VTr::VL>::pointer_type *>( get_ptr( array ) ), idx.value().data(),
		idx.value().get_layout() );
    }
#else
    template<typename T, typename VTr, short AID, layout_t Layout>
    GG_INLINE
    auto levaluate( const bitarray_intl<T,typename VTr::member_type,AID> & array,
		    sb::rvalue<VTr,Layout> idx ) {
	static_assert( sizeof(T)*8 == VTr::VL || VTr::VL == 1, "VL check" );
/*
	return 
	    simd::detail::vector_ref_impl<simd::detail::mask_bit_traits<VTr::VL>,typename VTr::member_type,array_encoding<T>,false,Layout>(
		reinterpret_cast<typename simd::detail::mask_bit_traits<VTr::VL>::pointer_type *>( get_ptr( array ) ), idx.value() );
*/
	using ATr = simd::detail::mask_bit_traits<VTr::VL>;
	return 
	    simd::create_vector_ref_vec<
		ATr,typename VTr::member_type,array_encoding_bit<1>,false,Layout>(
		    reinterpret_cast<typename ATr::pointer_type *>(
			get_ptr( array ) ), idx.value() );
    }
#endif

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
    template<unsigned short VL, bool aligned,
	     typename T, typename VTr, short AID, layout_t Layout>
    GG_INLINE
    auto evaluate_incseq( bitarray_intl<T,typename VTr::member_type,AID> array,
			  sb::rvalue<VTr,Layout> idx ) {
	static_assert( VTr::VL == 1, "incseq requirement" );
	static_assert( aligned, "default value; extension not checked" );
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

    template<typename A, typename T, unsigned short VL, bool aligned,
	     typename MPack>
    GG_INLINE
    auto
    evaluate( const refop<A,unop<T,unop_incseq<VL,aligned>>,VL> & op,
	      const MPack & mpack ) {
	// Don't evaluate unop_incseq, just pick up the data
	auto idx = evaluate( op.index().data(), mpack );
	static_assert( decltype(idx.value())::VL == 1, "VL match" );
	// Any mask applied to index has been set prior to VL width; we need
	// to assume here that the base pointer is always valid in a linear
	// fetch, such that we can fetch unconditionally
	auto rv = evaluate_incseq<VL,aligned>( op.array(), idx.uvalue() );
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
	return make_lvalue( levaluate_incseq<VL,true>(
				op.array(), idx.uvalue() ), mpack2 );
    }

    template<typename A, typename T, typename M, unsigned short VL,
	     bool aligned, typename MPack>
    GG_INLINE
    auto
    levaluate( const maskrefop<A,unop<T,unop_incseq<VL,aligned>>,M,VL> & op,
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
	    levaluate_incseq<VL,aligned>( op.array(), idx.uvalue() ), mpack2 );
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

    template<typename A, typename T, unsigned short VL, bool aligned,
	     typename MPack>
    GG_INLINE
    auto levaluate( const refop<A,unop<T,unop_incseq<VL,aligned>>,VL> & op,
		    const MPack & mpack ) {
	// We don't expect a mask here - if a mask was present, the refop
	// should have been converted to a maskrefop
	// Don't evaluate unop_incseq, just pick up the data
	auto idx = evaluate( op.index().data(), mpack );
	return make_lvalue(
	    levaluate_incseq<VL,aligned>( op.array(), idx.uvalue() ),
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

    template<unsigned cid, typename Tr, short aid, cacheop_flags flags,
	     typename MPack>
    GG_INLINE
    auto evaluate( const cacheop<cid,Tr,aid,flags> & op, const MPack & mpack ) {
	// Note: the cached values are sized to meet the vector length
	//       of the computation
	auto val = m_cache.template get<cid>();
	static_assert( Tr::VL == decltype(val)::VL, "VL check" );
	return make_rvalue( val, mpack );
    }

    template<unsigned cid, typename Tr, short aid, cacheop_flags flags,
	     typename MPack>
    GG_INLINE
    auto levaluate( const cacheop<cid,Tr,aid,flags> & op, const MPack & mpack ) {
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

    template<unsigned cid, typename Tr, short aid, cacheop_flags flags,
	     typename M, typename MPack>
    GG_INLINE
    auto levaluate( const binop<cacheop<cid,Tr,aid,flags>,M,binop_predicate> & op,
		    const MPack & mpack ) {
	auto mask = evaluate( op.data2(), mpack );
	using index_type = typename Tr::index_type;
	auto lv = simd::template create_vector_ref_cacheop<
	    Tr, index_type, array_encoding<typename Tr::member_type>,
	    false, lo_unknown>(
		&m_cache.template get<cid>(), index_type(0) );
	return make_lvalue( lv, mask.value(), mask.mpack() );
    }

    template<unsigned cid, typename Tr, short aid, cacheop_flags flags,
	     typename M, typename MPack>
    GG_INLINE
    auto levaluate( const binop<cacheop<cid,Tr,aid,flags>,M,binop_mask> & op,
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
    
    template<unsigned cid, typename TrC, short aid, cacheop_flags flags,
	     typename E2,
	     typename RedOp, typename MPack>
    GG_INLINE
    auto evaluate( const redop<cacheop<cid,TrC,aid,flags>,E2,RedOp> & op,
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
	
#if 0
    template<unsigned cid, typename TrC, short aid, cacheop_flags flags,
	     typename E2, typename C,
	     typename RedOp, typename MPack, typename TTT>
    GG_INLINE
    auto evaluate( const redop<cacheop<cid,TrC,aid,flags>,unop<binop<E2,C,binop_predicate>,unop_cvt_data_type<TTT>>,
		   RedOp> & op,
		   const MPack & mpack ) {
	// Special case: avoid store operations such that the cached value
	// can live in registers
	if constexpr ( RedOp::is_commutative
		       && is_unop_cvt_data_type<E2>::value ) {
	    // For vector-masks (AVX2), avoid conversion of the width of both
	    // the value and the mask. Use the unit of the redop (v+unit ==v)
	    // to replace the mask with disabled values.
	    // This works, provided that the RedOp's update-condition does not
	    // need to be converted itself.
	    
	    // Evaluate value before conversion
	    auto val = evaluate( op.val().data1().data(), mpack );
	    // Evaluate mask without conversion
	    auto mask = evaluate( op.val().data2(), val.mpack() );
	    auto cond = binop_mask::update_mask( mask );
	    // Replace disabled lanes with the RedOp's unit (v+unit == v)
	    auto dis
		= set_unit_if_disabled<RedOp>( val.uvalue(), cond.uvalue() );
	    // No need to be selective as the end result is the same
	    auto empty_mpack = sb::create_mask_pack();
	    // Perform cast operation
	    auto cst = E2::unop_type::evaluate( dis.uvalue(), empty_mpack );

	    auto r = evaluate_redop<RedOp,cid,TrC>( cst.uvalue(), empty_mpack );
	    return make_rvalue( r.value(), cond.mpack() );
	} else {
	    auto val = evaluate( op.val().data1(), mpack );
	    auto mask = evaluate( op.val().data2(), val.mpack() );
	    auto cond = binop_mask::update_mask( mask );
	    auto redop_mpack = sb::create_mask_pack( cond.value() );

	    auto r = evaluate_redop<RedOp,cid,TrC>( val.uvalue(), redop_mpack );
	    return make_rvalue( r.value(), cond.mpack() );
	}
    }
#else
    template<unsigned cid, typename TrC, short aid, cacheop_flags flags,
	     typename E2, typename C,
	     typename RedOp, typename MPack>
    GG_INLINE
    auto evaluate( const redop<cacheop<cid,TrC,aid,flags>,binop<E2,C,binop_predicate>,
		   RedOp> & op,
		   const MPack & mpack ) {
	// Special case: avoid store operations such that the cached value
	// can live in registers
	if constexpr ( RedOp::is_commutative
		       && is_unop_cvt_data_type<E2>::value ) {
	    // For vector-masks (AVX2), avoid conversion of the width of both
	    // the value and the mask. Use the unit of the redop (v+unit ==v)
	    // to replace the mask with disabled values.
	    // This works, provided that the RedOp's update-condition does not
	    // need to be converted itself.
	    
	    // Evaluate value before conversion
	    auto val = evaluate( op.val().data1().data(), mpack );
	    // Evaluate mask without conversion
	    auto mask = evaluate( op.val().data2(), val.mpack() );
	    auto cond = binop_mask::update_mask( mask );
	    // Replace disabled lanes with the RedOp's unit (v+unit == v)
	    auto dis
		= set_unit_if_disabled<RedOp>( val.uvalue(), cond.uvalue() );
	    // No need to be selective as the end result is the same
	    auto empty_mpack = sb::create_mask_pack();

	    // Perform cast operation
	    // Where possible, allow the reduced values to end up in any lane.
	    // It is hard to know at this place in the code when this is
	    // valid. In general, it should be valid when all lanes will be
	    // reduced to a single scalar and the reduction operation is
	    // commutative.
	    // In this instance, we will consider this for counting active
	    // edges, as this will be correct.
	    if constexpr ( expr::cache_get_accum_aid<cid,Accum>::valid
			   && expr::cache_get_accum_aid<cid,Accum>::aid
			   == expr::aid_frontier_nacte ) {
		auto cst = E2::unop_type::evaluate_confuse_lanes(
		    dis.uvalue() );

		auto r = evaluate_redop<RedOp,cid,TrC>(
		    cst.uvalue(), empty_mpack );
		auto redop_mpack = sb::create_mask_pack( cond.value() );
		return make_rvalue( r.value(), redop_mpack );
	    } else {
		auto cst = E2::unop_type::evaluate( dis.uvalue(), empty_mpack );

		auto r = evaluate_redop<RedOp,cid,TrC>(
		    cst.uvalue(), empty_mpack );
		auto redop_mpack = sb::create_mask_pack( cond.value() );
		return make_rvalue( r.value(), redop_mpack );
	    }
	} else {
	    auto val = evaluate( op.val().data1(), mpack );
	    auto mask = evaluate( op.val().data2(), val.mpack() );
	    auto cond = binop_mask::update_mask( mask );
	    auto redop_mpack = sb::create_mask_pack( cond.value() );

	    auto r = evaluate_redop<RedOp,cid,TrC>( val.uvalue(), redop_mpack );
	    return make_rvalue( r.value(), cond.mpack() );
	}
    }
#endif

    template<typename RedOp, unsigned cid, typename TrC,
	     typename VTr, layout_t Layout,
	     typename MPack>
    GG_INLINE
    auto evaluate_redop(
	sb::rvalue<VTr,Layout> val,
	const MPack & redop_mpack ) {
	// Special case: avoid store operations such that the cached value
	// can live in registers

	auto& r = m_cache.template get<cid>(); // reference into cache
	static_assert( decltype(val)::VL == 1 ||
		       decltype(val)::VL == std::remove_reference<decltype(r)>::type::VL,
		       "VL match" ); // we may have long vector with scalar access

	// bitarray vs array cases
	if constexpr ( std::is_void_v<typename TrC::member_type> ) {
	    using U = index_type_of_size<sizeof(VID)>;

	    auto vr = simd::template create_vector_ref_scalar<VTr,U,array_encoding<typename VTr::element_type>,false,lo_linalgn>(
		reinterpret_cast<typename VTr::pointer_type *>( &r.data() ), U(0) );
	    auto l = make_lvalue( vr, redop_mpack );
	    auto r = RedOp::evaluate( l.uvalue(), val, redop_mpack );
	    return r;
	} else {
	    // Cut of at too wide widths
	    // 4-byte int is more likely index supported in gather than
	    // 1 or 2-byte int
	    using U = index_type_of_size<std::max(VTr::W,(unsigned short)4)>;

	    auto vr = simd::template create_vector_ref_scalar<
		VTr,U,array_encoding<typename VTr::member_type>,
		false,lo_linalgn>(
		    reinterpret_cast<typename VTr::pointer_type *>( &r.data() ),
		    U(0) );

	    auto l = make_lvalue( vr, redop_mpack );
	    static_assert( sizeof(typename decltype(l.value())::index_type) != 1, "err" );
	    auto r = RedOp::evaluate( l.uvalue(), val, redop_mpack );
	    return r;
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

	if constexpr ( decltype(cnd)::VL == 1 ) {
	    // We could turn this into iterative code instead of recursive code
	    // by checking the return value of evaluate( data2() ) and ensuring
	    // a relevant mask exists in the mask pack. Then iterate the loop as
	    // validness of lanes is not affected throughout iterations.
	    // TODO: convert mask if necessary.

	    // Build mask
	    auto m0 = mpack.get_mask_for( cnd );

	    if( m0.data() && cnd.data() ) {
		do {
		    // Execute loop body
		    auto b = evaluate( op.data2(), mpack );

		    // Evaluate condition
		    cnd = evaluate( op.data1(), mpack ).value();
		} while( cnd.data() );
	    }
	} else {
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
    }

    template<typename S, typename E, typename V, typename MPack>
    auto evaluate( const ternop<S,E,V,ternop_find_first> & op,
		   const MPack & mpack ) {
	return evaluate_find_first( op, mpack );
    }

    template<typename ATr, typename VTr, short AID, typename Enc, bool NT,
	     layout_t Layout1, layout_t Layout2, layout_t Layout3,
	     typename MPack>
    auto evaluate_find_first_loop(
	const array_intl<typename VTr::member_type,typename ATr::member_type,
			 AID,Enc,NT> & array,
	simd::vec<ATr,Layout1> start,
	simd::vec<ATr,Layout2> end,
	simd::vec<VTr,Layout3> value,
	const MPack & mpack ) {
	static_assert( ATr::VL == 1, "scalar assumption" );
	using type_list = typename expr::detail::create_type_list<
	    typename VTr::member_type>;
#if __AVX512F__
	using recommendation
	    = expr::detail::recommended_vectorization<type_list,64,16,true>;
#elif __AVX2__
	using recommendation
	    = expr::detail::recommended_vectorization<type_list,32,16,true>;
#else
	using recommendation
	    = expr::detail::recommended_vectorization<type_list,1,1,true>;
#endif
	static constexpr short VL = recommendation::vlen;
	using ATrVL = typename ATr::rebindVL<VL>::type;
	using ATy = typename ATr::member_type;

	// Loop index
	auto idx = start;
	
	// Loop increment
	auto inc = simd::create_scalar<ATr>( VL );

	if( value.data() == 0 ) { // TODO: adjust target::find_first
	    // Allow to over-run end of range, then correct
	    auto cnd = idx < end;
	    while( cnd.data() ) { 
		// Execute loop body
		auto velm = evaluate_incseq<VL,false>(
		    array, make_rvalue( idx, expr::sb::mask_pack<>() )
		    ).value();
		auto fnd = simd::find_first( velm );
		if( fnd.data() < VL ) {
		    auto fst = fnd.template convert_data_type<ATr>() + idx;
		    if( fst.data() >= end.data() )
			return make_rvalue( end, mpack );
		    else
			return make_rvalue( fst, mpack );
		}

		// Evaluate condition
		idx += inc;
		cnd = idx < end;
	    }
	    return make_rvalue( end, mpack );
	} else {
	    auto cnd = idx < end;
	    auto one = simd::create_scalar<ATr>( 1 );
	    while( cnd.data() ) { 
		// Execute loop body
		auto elm = evaluate( array, make_rvalue( idx, mpack ) ).value();
		if( elm.data() == value.data() )
		    return make_rvalue( idx, mpack );

		// Evaluate condition
		idx += one;
		cnd = idx < end;
	    }
	    return make_rvalue( end, mpack );
	}
    }

    template<typename ATr, typename VTr, short AID, typename Enc, bool NT,
	     layout_t Layout1, layout_t Layout2, layout_t Layout3,
	     layout_t Layout4, typename MPack>
    auto evaluate_find_first_vec(
	const array_intl<typename VTr::member_type,typename ATr::member_type,
			 AID,Enc,NT> & array,
	simd::vec<ATr,Layout1> start,
	simd::vec<ATr,Layout2> end,
	simd::vec<VTr,Layout3> value,
	simd::vec<ATr,Layout4> result,
	const MPack & mpack ) {

	static_assert( ATr::VL != 1, "vector assumption" );
	using ATy = typename ATr::member_type;

	// Evaluate condition.
	auto idx = start;
	auto cnd = idx < end;

	// Build mask
	auto m0 = mpack.get_mask_for( cnd );

	// Disable terminated lanes
	auto m = m0 && cnd;
	
	if( m.is_all_false() ) {
	    return result;
	} else {
	    // Create mask_pack with terminated lanes disabled
	    auto mpack2 = sb::create_mask_pack( m );

	    // Execute loop body
	    auto velm = evaluate( array, make_rvalue( idx, mpack2 ) ).value();
	    auto fnd = velm == value;
	    auto mc = m.template convert_data_type<typename VTr::prefmask_traits>();
	    auto updated = ::iif( mc && fnd, result, idx );
	    auto one = simd::detail::vector_impl<ATr>::one_val();
	    auto nxt = idx + one;

	    auto mpack3 = sb::create_mask_pack( mc && !fnd );
	    return evaluate_find_first_vec( array, nxt, end, value,
					    updated, mpack3 );
	}
    }

#if 0
    template<typename VTr, typename ATr, short AID, typename Enc, bool NT,
	     layout_t Layout1, layout_t Layout2, layout_t Layout3,
	     layout_t Layout4, typename MPack>
    auto evaluate_find_first_vec_wide(
	const array_intl<typename VTr::member_type,typename ATr::member_type,
			 AID,array_encoding<VTy>,NT> & array,
	simd::vec<ATr,Layout1> start,
	simd::vec<ATr,Layout2> end,
	simd::vec<VTr,Layout3> value,
	simd::vec<ATr,Layout4> result,
	const MPack & mpack ) {

	static_assert( VTr::W == ATr::W, "width must match" );
	static_assert( ATr::VL != 1, "vector assumption" );
	using ATy = typename ATr::member_type;

	// Evaluate condition.
	auto idx = start;
	auto cnd = idx < end;

	// Build mask
	auto m0 = mpack.get_mask_for( cnd );

	// Disable terminated lanes
	auto m = m0 && cnd;
	
	if( m.is_all_false() ) {
	    return result;
	} else {
	    // Create mask_pack with terminated lanes disabled
	    auto mpack2 = sb::create_mask_pack( m );

	    // Execute loop body
	    auto velm = evaluate( array, make_rvalue( idx, mpack2 ) ).value();
	    auto fnd = velm == value;
	    auto mc = m.template convert_data_type<typename VTr::prefmask_traits>();
	    auto updated = ::iif( mc && fnd, result, idx );
	    auto one = simd::detail::vector_impl<ATr>::one_val();
	    auto nxt = idx + one;

	    auto mpack3 = sb::create_mask_pack( mc && !fnd );
	    return evaluate_find_first_vec( array, nxt, end, value,
					    updated, mpack3 );
	}
    }
#endif


    template<typename S, typename E, typename V, typename MPack>
    auto evaluate_find_first( const ternop<S,E,V,ternop_find_first> & op,
			      const MPack & mpack ) {
	static_assert( S::array_type::AID == E::array_type::AID,
		       "Start and end range must refer to same array" );

	// Note: leaves mask_pack unmodified.
	    
	// Evaluate condition.
	// It is assumed that S and E are simple refop
	auto array = op.data1().array();
	auto start = evaluate( op.data1().index(), mpack ).value();
	auto end = evaluate( op.data2().index(), mpack ).value();
	auto value = evaluate( op.data3(), mpack ).value();
	auto idx = start;
	auto cnd = idx < end;
	using ATr = typename decltype(idx)::data_type;
	using ATy = typename ATr::member_type;

	if constexpr ( decltype(cnd)::VL == 1 ) {
	    // Build mask
	    if constexpr ( !MPack::is_empty() ) {
		if( !mpack.template get_any<ATr>().data() )
		    // Re-package as scalar such that all exit points have
		    // the same metadata (layout) on the return value.
		    return make_rvalue( simd::create_scalar<ATr>( end.data() ),
					mpack );
	    }

	    // For high-degree vertices / long arrays, use nested parallelism.
	    ATy range = end.data() - start.data();
	    static constexpr ATy granularity = 2048;

	    auto num_threads = graptor_num_threads();
	    if( num_threads > 1 && range > 2 * granularity ) {
		// Chunks and their sizes
		const ATy chunks
		    = std::min( ATy( num_threads * 8 ),
				ATy( range / granularity ) );
		const ATy chunk_size
		    = ( ( range + chunks - 1 ) / chunks + 31 ) & ~ATy(31);

		// Shared variable to cancel unnecessary work
		volatile ATy lowest = end.data();

		// std::cerr << "chunks=" << chunks << " size=" << chunk_size << "\n";

		parallel_loop( (ATy)0, chunks, 1, [&]( ATy k ) {
		    auto s_start = k * chunk_size;
		    auto s_end = std::min( range, ( k + 1 ) * chunk_size );
		    auto v_start = start + simd::create_scalar<ATr>( s_start );
		    auto v_end = start + simd::create_scalar<ATr>( s_end );

		    if( lowest >= v_end.data() ) {
			auto res = evaluate_find_first_loop(
			    op.data1().array(), v_start, v_end, value, mpack )
			    .value().data();
			if( res != v_end.data() ) {
			    ATy l;
			    do {
				l = lowest;
				if( l < res )
				    break;
			    } while( !__sync_bool_compare_and_swap(
					 &lowest, l, res ) );
			}
		    }
		} );

		// Post-process
		ATy result = lowest;

		return make_rvalue( simd::create_scalar<ATr>( result ), mpack );
	    } else {
		auto res = evaluate_find_first_loop(
		    op.data1().array(), start, end, value, mpack ).value();
		return make_rvalue( simd::create_scalar<ATr>( res.data() ),
				    mpack );
	    } 
	} else {
		using ArrayTy = decltype( array );
		using T = typename ArrayTy::type;
		if constexpr (
		    std::is_integral_v<T> &&
		    std::is_same_v<typename ArrayTy::encoding,
		    array_encoding<T>>
#if __AVX512F__
		    && false
#endif
		    ) {
		    // In case of integral values, do a wide gather.
		    // Pre-convert the sought value to the wider width.
		    // This will avoid any conversion of vector lane width
		    // during find_first_vec.
		    // This is disabled when AVX512 is available, as this
		    // uses 1-bit masks for comparisons, regardless of
		    // lane width.
		    using encoding = array_encoding_wide<T>;
		    using basic_wide_type = int_type_of_size_t<sizeof(ATy)>;
		    using wide_type = std::conditional_t<
			std::is_signed_v<T>,
			std::make_signed_t<basic_wide_type>,
			std::make_unsigned_t<basic_wide_type>>;
		    using array_wide = array_intl<
			wide_type, typename ArrayTy::index_type,
			ArrayTy::AID, encoding, ArrayTy::NT>;

		    auto cvalue = value.template convert_to<wide_type>();

		    auto pos = evaluate_find_first_vec(
			array_wide(), start, end, cvalue, end, mpack );
		    return make_rvalue( pos, mpack );
		} else {
		    auto pos = evaluate_find_first_vec(
			array, start, end, value, end, mpack );
		    return make_rvalue( pos, mpack );
		}
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
	 typename Accum,
	 typename Expr>
__attribute__((always_inline))
inline auto evaluate( Cache &c, const value_map_type & m, Expr e ) {
    auto array_map = extract_pointer_set( e );
    return expr::evaluator<value_map_type, Cache, decltype(array_map),
			   Accum, AtomicUpdate>( c, m, array_map )
	.evaluate( e, sb::mask_pack<>() );
}

template<typename Cache, typename value_map_type,
	 typename Accum,
	 typename Expr>
__attribute__((always_inline))
static inline auto evaluate( Cache &c, const value_map_type & m, Expr e ) {
    auto array_map = extract_pointer_set( e );
    return expr::evaluator<value_map_type, Cache, decltype(array_map), Accum,
			   false>(
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
	auto r = expr::evaluate<Cache,ValueMap,cache<>,Expr>( c, m, expr );
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
