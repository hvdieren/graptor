#ifndef GRAPTOR_DSL_COMP_FRONTIERSELECT_H
#define GRAPTOR_DSL_COMP_FRONTIERSELECT_H

#include <limits>

#include "graptor/customfp.h"
#include "graptor/bitfield.h"
#include "graptor/utils.h"

namespace expr {

/**********************************************************************
 * Analysing element widths in arrays in order to determine vector
 * lengths and data layouts for adaptable arrays (e.g., frontiers)
 **********************************************************************/
namespace detail {

// TODO: For now, assume there is only one type in use in the code.
//       If there are multiple types (or at least, widths), we fail for now.
//       In general, we should also take into account that user code may
//       want to refer to the frontier, implying we should filter that out
//       (as well as raising the question whether we will be able to influence
//        that frontier, which won't be possible, e.g., in case the user
//        copies the frontier array).
// NOTE: Since introduction of super-length vectors, chose the shortest type
//       when types of different sizes occur.

/*----------------------------------------------------------------------*
 * Type lists: summarise widths and preferred vector lengths of types
 *             used in an AST.
 *----------------------------------------------------------------------*/
template<typename T, typename Enable = void>
struct create_type_list;

template<typename T>
struct create_type_list<T,std::enable_if_t<!is_customfp_v<T>
					   && !is_bitfield_v<T>>> {
    static constexpr unsigned short narrowest_pow2_width =
	(unsigned short)( 8 * sizeof(T) );
    static constexpr unsigned short nonpow2_preferred = 1;
};

template<>
struct create_type_list<void> {
    static constexpr unsigned short narrowest_pow2_width =
	(unsigned short)( 8 * std::numeric_limits<unsigned short>::max() );
    static constexpr unsigned short nonpow2_preferred = 1;
};

template<unsigned short B>
struct create_type_list<bitfield<B>> {
    // Custom-fp with power-of-2 byte width
    static constexpr unsigned short narrowest_pow2_width = B;
    static constexpr unsigned short nonpow2_preferred = 1;
};

/*
template<unsigned short E, unsigned short M>
struct create_type_list<customfp<E,M>,
			std::enable_if_t<((E+M)&(E+M-1))==0 && ((E+M)&7)==0>> {
    // Custom-fp with power-of-2 byte width
    static constexpr unsigned short narrowest_pow2_width = E+M;
    static constexpr unsigned short nonpow2_preferred = 1;
};
*/

template<bool S, unsigned short E, unsigned short M, bool Z, int B>
struct create_type_list<::detail::customfp_em<S,E,M,Z,B>,
			std::enable_if_t<(::detail::customfp_em<S,E,M,Z,B>::bit_size&(::detail::customfp_em<S,E,M,Z,B>::bit_size-1))==0 && (((S?1:0)+E+M)&7)==0>> {
    // Custom-fp with power-of-2 byte width
    static constexpr unsigned short narrowest_pow2_width =
	::detail::customfp_em<S,E,M,Z,B>::bit_size;
    static constexpr unsigned short nonpow2_preferred = 1;
};

template<unsigned short E, unsigned short M>
struct create_type_list<customfp<E,M>,std::enable_if_t<E+M==21>> {
    // A hard-coded exception.
    static constexpr unsigned short narrowest_pow2_width =
	(unsigned short)( 8 * std::numeric_limits<unsigned short>::max() );
    static constexpr unsigned short nonpow2_preferred = 3;
};

/*----------------------------------------------------------------------*
 * Auxiliary type list to characterise the encoding of an array.
 * The power-of-2 byte width of the encoded data is ignored, focussing on
 * creating one full vector worth of data, which is read by a shorter
 * vector in the case that the encoded data is a smaller power-of-2 than
 * the computation is performed on (interface type of array).
 * However, if the encoded data requires an unusual vector length, then
 * we record that.
 *----------------------------------------------------------------------*/
template<typename T>
struct create_type_list_nonpow2 {
    using aux = create_type_list<T>;
    static constexpr unsigned short narrowest_pow2_width =
	(unsigned short)( 8 * std::numeric_limits<unsigned short>::max() );
    static constexpr unsigned short nonpow2_preferred = aux::nonpow2_preferred;
};


/*----------------------------------------------------------------------*
 * Merge type lists: select narrowest native (power-of-2 wide) type and
 *                   calculate least common multiple (or a multiple thereof)
 *                   for other types. Currently, only customfp<E,M> with
 *                   E+M is relevant, so don't bother about the LCM.
 *----------------------------------------------------------------------*/
template<typename TypeList1, typename TypeList2>
struct merge_type_lists {
    static constexpr unsigned short narrowest_pow2_width =
	std::min( TypeList1::narrowest_pow2_width,
		  TypeList2::narrowest_pow2_width );
    static constexpr unsigned short nonpow2_preferred = 
	lcm( TypeList1::nonpow2_preferred, TypeList2::nonpow2_preferred );
    // static_assert( narrowest_pow2_width > 1, "check" );
};

/*----------------------------------------------------------------------*
 * Infer recommendation for bytewidth of frontier masks and for vector
 * length.
 * TODO: We should distinguish which types are accessed through src vs dst
 *       (or vid) and whether these will result in gather/scatter or
 *       sequential access. The code below implicitly assumes that the
 *       non-power-of-2 byte widths are accessed only through vk_src.
 *       As such, Rnd may be interpreted as vk_src accessed randomly (RndRd).
 *----------------------------------------------------------------------*/
template<typename TypeList,
	 unsigned short NativeWidth,
	 unsigned short MaxWidth,
	 bool Rnd,
	 typename Enable = void> // random-access case, ignore non-pow2 fields
struct recommended_vectorization {
    static constexpr unsigned short vlen =
	std::min((unsigned short)(8*NativeWidth/TypeList::narrowest_pow2_width),
		 MaxWidth);
    static constexpr unsigned short width = TypeList::narrowest_pow2_width;
};

template<typename TypeList,
	 unsigned short NativeWidth,
	 unsigned short MaxWidth>
struct recommended_vectorization<TypeList,NativeWidth,MaxWidth,false,
				 std::enable_if_t<MaxWidth!=1>> {
    static constexpr unsigned short vlen =
	TypeList::nonpow2_preferred
	* std::min((unsigned short)(8*NativeWidth/TypeList::narrowest_pow2_width),
		   MaxWidth);
    static constexpr unsigned short width = TypeList::narrowest_pow2_width;
};

template<typename TypeList,
	 unsigned short NativeWidth,
	 bool Rnd>
struct recommended_vectorization<TypeList,NativeWidth,1,Rnd,void> {
    static constexpr unsigned short vlen = 1; // scalar processing
    static constexpr unsigned short width = 1; // bool frontier
};

/*----------------------------------------------------------------------*
 * Build a type list by walking the AST.
 *----------------------------------------------------------------------*/
template<typename E, typename Enable = void>
struct collect_types;

template<>
struct collect_types<noop> : public create_type_list<void> { };

template<typename Tr, value_kind VKind>
struct collect_types<value<Tr, VKind>,
		     std::enable_if_t<simd::detail::is_mask_traits_v<Tr>>>
    : public create_type_list<void> { };

template<typename Tr, value_kind VKind>
struct collect_types<value<Tr, VKind>,
		     std::enable_if_t<!simd::detail::is_mask_traits_v<Tr>>>
    : public create_type_list<typename Tr::element_type> { };

template<typename T, typename U, short AID, typename Enc, bool NT>
struct collect_types<array_intl<T, U, AID, Enc, NT>>
    // Computationally, we will be using type T, however, for load/store
    // purpose we are looking at type Enc::stored_type. If the latter is not
    // an elementary type width (power of 2), then we will need to use multiple
    // vectors at once.
    : public merge_type_lists<create_type_list<T>,
			      create_type_list_nonpow2<typename Enc::stored_type>> { };

template<typename T, typename U, short AID, typename Enc, bool NT>
struct collect_types<array_ro<T, U, AID, Enc, NT>>
    // Computationally, we will be using type T, however, for load/store
    // purpose we are looking at type Enc::stored_type. If the latter is not
    // an elementary type width (power of 2), then we will need to use multiple
    // vectors at once.
    : public merge_type_lists<create_type_list<T>,
			      create_type_list_nonpow2<typename Enc::stored_type>> { };

template<typename T, typename U, short AID>
struct collect_types<bitarray_intl<T, U, AID>>
    // Computationally, we will be using type T, however, for load/store
    // purpose we are looking at type Enc::stored_type. If the latter is not
    // an elementary type width (power of 2), then we will need to use multiple
    // vectors at once.
    : public merge_type_lists<create_type_list<T>,
			      create_type_list_nonpow2<bitfield<1>>> { };

template<typename T, typename U, short AID>
struct collect_types<bitarray_ro<T, U, AID>>
    // Computationally, we will be using type T, however, for load/store
    // purpose we are looking at type Enc::stored_type. If the latter is not
    // an elementary type width (power of 2), then we will need to use multiple
    // vectors at once.
    : public merge_type_lists<create_type_list<T>,
			      create_type_list_nonpow2<bitfield<1>>> { };

template<typename Expr, typename UnOp>
struct collect_types<unop<Expr,UnOp>,
		     std::enable_if_t<!unop_is_cvt<UnOp>::value
		     && !unop_is_cvt_data_type<UnOp>::value>>
    : public collect_types<Expr> { };

template<typename Expr, typename T, unsigned short VL>
struct collect_types<unop<Expr,unop_cvt<T,VL>>>
    : public merge_type_lists<collect_types<Expr>,create_type_list<T>> { };

template<typename Expr, typename Tr>
struct collect_types<unop<Expr,unop_cvt_data_type<Tr>>,
		     std::enable_if_t<simd::detail::is_mask_traits_v<Tr>>>
    : collect_types<Expr> { };

template<typename Expr, typename Tr>
struct collect_types<unop<Expr,unop_cvt_data_type<Tr>>,
		     std::enable_if_t<!simd::detail::is_mask_traits_v<Tr>>>
    : merge_type_lists<collect_types<Expr>,
		       create_type_list<typename Tr::element_type>> { };

template<typename E1, typename E2, typename BinOp>
struct collect_types<binop<E1,E2,BinOp>>
    : merge_type_lists<collect_types<E1>,collect_types<E2>> { };

template<typename E1, typename E2, typename E3, typename TernOp>
struct collect_types<ternop<E1,E2,E3,TernOp>>
    : merge_type_lists<collect_types<E1>,
		       merge_type_lists<collect_types<E2>,
					collect_types<E3>>> { };

template<bool nt, typename A, typename R>
struct collect_types<storeop<nt,A,R>>
    : merge_type_lists<collect_types<A>,collect_types<R>> { };

// TODO: T in refop should be redundant to interface type of array.
template<typename A, typename T, unsigned short VL>
struct collect_types<refop<A,T,VL>>
    : merge_type_lists<collect_types<A>,collect_types<T>> { };

template<typename E1, typename E2, typename RedOp>
struct collect_types<redop<E1,E2,RedOp>>
    : merge_type_lists<collect_types<E1>,collect_types<E2>> { };

template<typename S, typename U, typename C, typename DFSAOp>
struct collect_types<dfsaop<S,U,C,DFSAOp>>
    : merge_type_lists<collect_types<S>,
		       merge_type_lists<collect_types<U>,
					collect_types<C>>> { };

/*----------------------------------------------------------------------*
 * Auxiliary: what frontier type to use for a particular byte width?
 *----------------------------------------------------------------------*/
template<unsigned short W, unsigned short VL>
struct type_for_bitwidth;

template<unsigned short VL>
struct type_for_bitwidth<8*8,VL> {
    typedef logical<8> type;
    static constexpr frontier_type ftype = frontier_type::ft_logical8;
};

template<unsigned short VL>
struct type_for_bitwidth<8*4,VL> {
    typedef logical<4> type;
    static constexpr frontier_type ftype = frontier_type::ft_logical4;
};

template<unsigned short VL>
struct type_for_bitwidth<8*2,VL> {
    typedef logical<2> type; // WARNING - not fully implemented!
    static constexpr frontier_type ftype = frontier_type::ft_logical2;
};

template<unsigned short VL>
struct type_for_bitwidth<8*1,VL> {
    typedef logical<1> type; // WARNING - not fully implemented!
    static constexpr frontier_type ftype = frontier_type::ft_logical1;
};

template<>
struct type_for_bitwidth<8*1,1> {
    typedef bool type;
    static constexpr frontier_type ftype = frontier_type::ft_bool;
};

template<unsigned short VL>
struct type_for_bitwidth<2,VL> {
    typedef bitfield<2> type;
    static constexpr frontier_type ftype = frontier_type::ft_bit2;
};

template<unsigned short VL>
struct type_for_bitwidth<1,VL> {
    typedef void type;
    static constexpr frontier_type ftype = frontier_type::ft_bit;
};

} // namespace detail

/*----------------------------------------------------------------------*
 * Determine frontier properties, specifically for edgemap.
 * This is an auxiliary that will be used in further specialisations.
 *----------------------------------------------------------------------*/
// TODO: for AVX-512BW, as all comparisons results in k-masks, it may be
//       beneficial to represent the frontier as bool and do comparison
//       operations that create the k-mask, as opposed to trying to avoid
//       computation/conversion upon loading the mask for AVX-2.
template<typename VID, typename Operator, unsigned short VLBound, bool RndRd>
struct determine_frontier_bytewidth {
private:
    static Operator & getop();
    using E1 = decltype(getop().relax(
			    expr::value<simd::ty<VID,1>,expr::vk_src>(),
			    expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			    expr::value<simd::ty<EID,1>,expr::vk_edge>() ));
    using E2 = decltype(getop().active(
			    expr::value<simd::ty<VID,1>,expr::vk_dst>() ));
    using type_list =
	detail::merge_type_lists<detail::collect_types<E1>,
				 detail::collect_types<E2>>;
public:
#if __AVX512F__
    using recommendation =
	detail::recommended_vectorization<type_list,64,VLBound,RndRd>;
#elif __AVX2__
    using recommendation =
	detail::recommended_vectorization<type_list,32,VLBound,RndRd>;
#elif __SSE4_2__
    using recommendation =
	detail::recommended_vectorization<type_list,16,VLBound,RndRd>;
#else
    using recommendation =
	detail::recommended_vectorization<type_list,1,1,RndRd>;
#endif

    static constexpr unsigned short VL = recommendation::vlen;

#if FRONTIER_BITMASK
    // Read- and write-frontiers are typically of equal type because
    // the written frontier is often directly fed back as a read frontier
    // TODO: possible exception is when only a frontier is written or only
    //       a frontier is read.
    static constexpr frontier_type rd_ftype = frontier_type::ft_bit;
    static constexpr frontier_type wr_ftype = frontier_type::ft_bit;
#else
    static constexpr unsigned short bit_width = recommendation::width;
    static constexpr frontier_type rd_ftype =
	detail::type_for_bitwidth<bit_width,VL>::ftype;
    static constexpr frontier_type wr_ftype =
	detail::type_for_bitwidth<bit_width,VL>::ftype;
#endif // FRONTIER_BITMASK
};

// Only option is scalar execution. Use boolean frontier
template<typename VID, typename Operator, bool RndRd>
struct determine_frontier_bytewidth<VID,Operator,1,RndRd> {
    static constexpr unsigned short width = 1;
    static constexpr unsigned short VL = 1;
    using type = bool;
    static constexpr frontier_type rd_ftype = frontier_type::ft_bool;
    static constexpr frontier_type wr_ftype = frontier_type::ft_bool;
};

// Trigger errors at compile-time
template<typename VID, typename Operator, bool RndRd>
struct determine_frontier_bytewidth<VID,Operator,0,RndRd>; // no def


/*----------------------------------------------------------------------*
 * Determine frontier properties, specifically for edgemap.
 *----------------------------------------------------------------------*/
#ifndef MAX_VL_V
#ifndef MAX_VL
#define MAX_VL_V (unsigned short)64
#else
#define MAX_VL_V MAX_VL
#endif
#endif // MAX_VL_V

#ifndef MAX_VL
#define MAX_VL (unsigned short)16
#endif // MAX_VL

template<typename VID, typename Operator,
	 bool may_be_unbacked, unsigned short VLBound_, bool RndRd, bool RndWr>
struct determine_frontier {
    static constexpr unsigned short VLBound =
	std::min( VLBound_, (unsigned short)MAX_VL );

    using bytewidth_rd =
	determine_frontier_bytewidth<VID,Operator,VLBound,RndRd>;
    using bytewidth_wr =
	determine_frontier_bytewidth<VID,Operator,VLBound,RndWr>;

    static constexpr frontier_type rd_ftype = bytewidth_rd::rd_ftype;
    static constexpr frontier_type wr_ftype =
#if !DISABLE_UNBACKED
	// Unbacked frontiers possible if allowed algorithmically (may_omit_wr)
	// and we have a means to calculate them differently from the output
	// of the edge map operator (fm_calculate).
	// Operator::new_frontier == fm_calculate && !RndWr
	// && Operator::may_omit_frontier_wr ? frontier_type::ft_unbacked :
	may_be_unbacked && !RndWr ? frontier_type::ft_unbacked :
#endif
	bytewidth_wr::wr_ftype;

    // Vector length
    static constexpr unsigned short VL =
	std::min( (unsigned short)MAX_VL,
		  std::max( bytewidth_rd::VL, bytewidth_wr::VL ) );

    static std::ostream & report( std::ostream & os ) {
	return os << "emap_config { VL: " << VL
		  << ", rd_ftype: " << rd_ftype
		  << ", wr_ftype: " << wr_ftype
		  << " }";
    }
};

template<typename VID, typename Operator, typename GraphType,
	 bool may_be_unbacked>
struct determine_emap_config {
    using pull = determine_frontier<
	VID,Operator,may_be_unbacked,GraphType::getPullVLBound(),true,false>;
    using push = determine_frontier<
	VID,Operator,may_be_unbacked,GraphType::getPushVLBound(),false,true>;
    using ireg = determine_frontier<
	VID,Operator,may_be_unbacked,GraphType::getIRegVLBound(),true,true>;
    using scalar = determine_frontier<VID,Operator,false,1,true,true>;
};

/*----------------------------------------------------------------------*
 * Determine frontier properties, specifically for vertexmap.
 *----------------------------------------------------------------------*/
// For vertex_map. PAlignment is the alignment provided by the graph
// format. Set an optimistic default to allow aggressive vectorization.
// PAlignment is not currently used - try to maximize vectorization.
template<typename VID, typename Operator,
    unsigned short PAlignment = std::min( (unsigned short)MAX_VL_V, VLUpperBound )>
struct determine_frontier_vmap {
private:
    static constexpr unsigned short VLBound =
	std::min( VLUpperBound, (unsigned short)MAX_VL_V );
    static Operator & getop();
    using E = decltype(getop().operator() (
			   expr::value<simd::ty<VID,1>,expr::vk_vid>() ));
    using type_list = detail::collect_types<E>;
public:
#if __AVX512F__
    using recommendation =
	detail::recommended_vectorization<type_list,64,VLBound,false>;
#elif __AVX2__
    using recommendation =
	detail::recommended_vectorization<type_list,32,VLBound,false>;
#elif __SSE4_2__
    using recommendation =
	detail::recommended_vectorization<type_list,16,VLBound,false>;
#else
    using recommendation = detail::recommended_vectorization<type_list,1,1,false>;
#endif

    static constexpr unsigned short VL = recommendation::vlen;
    static constexpr unsigned short bit_width = recommendation::width;

    typedef typename detail::type_for_bitwidth<bit_width,VL>::type type;
    static constexpr frontier_type ftype =
	detail::type_for_bitwidth<bit_width,VL>::ftype;
};

} // namespace expr


#endif // GRAPTOR_DSL_COMP_FRONTIERSELECT_H
