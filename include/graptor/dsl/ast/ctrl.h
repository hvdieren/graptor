// -*- c++ -*-
#ifndef GRAPTOR_DSL_AST_CTRL_H
#define GRAPTOR_DSL_AST_CTRL_H

namespace expr {

/* whileop: while loop
 */
struct loopop_while {
    template<typename C, typename B, typename F>
    struct types {
	using result_type = typename F::data_type;
    };

    static constexpr char const * name = "loopop_while";
};

template<typename E1, typename E2, typename E3>
static constexpr
auto make_loop( E1 e1, E2 e2, E3 e3 ) {
    return make_ternop( e1, e2, e3, loopop_while() );
}

template<unsigned short AIDIdx, unsigned short AIDTmp,
	 unsigned short AIDSwp,
	 typename CountTy, typename PosTy, typename DataTy, typename ValueTy,
	 typename VIDTy>
static auto
unique_insert_sort( const CountTy & count,
		    const PosTy & index,
		    const DataTy & data,
		    const ValueTy & ins,
		    VIDTy vid ) {
    using CTy = typename CountTy::type;
    using DTy = typename DataTy::type;
    using PTy = typename PosTy::type;
    auto idx = make_scalar<AIDIdx,CTy,VIDTy::VL>();
    auto tmp = make_scalar<AIDTmp,DTy,VIDTy::VL>();
    auto swp = make_scalar<AIDSwp,DTy,VIDTy::VL>();
    return make_seq(
	tmp = ins,
	idx = _0,
	make_loop( // while loop
	    idx < count[vid] && tmp != data[index[vid]+cast<PTy>(idx)], // condition
	    make_seq( // loop body and increment
		swp = data[index[vid]+cast<PTy>(idx)],
		data[index[vid]+cast<PTy>(idx)] = iif( swp > tmp, tmp, swp ),
		tmp = iif( swp > tmp, swp, tmp ),
		idx += _1 ),
	    make_seq( // final value
		// store back tmp only if degree > 0
		data[index[vid]+cast<PTy>(idx)] = add_predicate(
		    tmp, index[vid+_1] != index[vid] ),
		count[vid] += iif( idx >= count[vid], _0, _1 ),
		true_val( idx )
		) ) );
}

template<unsigned short AIDIdx, unsigned short AIDTmp,
	 typename CountTy, typename PosTy, typename DataTy,
	 typename VIDTy>
static auto
smallest_absent( const CountTy & count,
		 const PosTy & index,
		 const DataTy & data, VIDTy vid ) {
    using CTy = typename CountTy::type;
    using DTy = typename DataTy::type;
    using PTy = typename PosTy::type;
    auto idx = make_scalar<AIDIdx,CTy,VIDTy::VL>();
    auto col = make_scalar<AIDTmp,DTy,VIDTy::VL>();
    return make_seq(
	idx = _0,
	col = _0,
	make_loop(
	    idx < count[vid] && data[index[vid]+cast<PTy>(idx)] == col, // condition
	    make_seq( // loop body
		idx += _1,
		col += _1 ),
	    // final value
	    col
	    ) );
}

template<unsigned short AIDIdx,
	 typename PosTy, typename DataTy,
	 typename VIDTy>
static auto
smallest_false( const PosTy & index,
		const DataTy & data, VIDTy vid ) {
    using DTy = typename DataTy::type;
    using PTy = typename PosTy::type;
    using MTr = typename simd::ty<PTy,VIDTy::VL>::prefmask_traits;
    auto idx = make_scalar<AIDIdx,PTy,VIDTy::VL>();
    return make_seq(
	idx = index[vid],
	make_loop(
	    idx < index[vid+_1] && make_unop_cvt_data_type<MTr>( data[idx] ), // condition
	    idx += _1, // loop body
	    // final value
	    cast<VID>(
		iif(idx >= index[vid+_1],
		    idx - index[vid],
		    index[vid+_1] - index[vid]) )
	    ) );
}

template<unsigned short AIDIdx, unsigned short AIDTmp, unsigned short AIDLet1,
	 unsigned short AIDLet2,
	 typename PosTy, typename DataTy,
	 typename VIDTy>
static auto
smallest_false_grouped( const PosTy & index,
			const DataTy & data, VIDTy vid ) {
    using DTy = typename DataTy::type;
    using PTy = typename PosTy::type;

    static_assert( is_logical_v<DTy> && sizeof(DTy) == 1,
		   "Assuming byte-sized logical masks" );
    static_assert( std::is_same_v<typename DataTy::encoding,
		   array_encoding<DTy>>, "Assuming baseline encoding" );

    // TODO: try with uint32_t
    using GTy = uint64_t;
    constexpr size_t G = sizeof(GTy)/sizeof(DTy);
    static_assert( G == 8, "assumption" );

    array_ro<GTy,typename DataTy::index_type,DataTy::AID,
	     array_encoding_multi<DTy>> gdata( data.get_ptr() );

    auto tmp = array_intl<GTy,VID,AIDTmp,array_encoding<void>,
			  false>()[zero_val(vid)];

    auto idx = make_scalar<AIDIdx,PTy,VIDTy::VL>();
    auto Gast = constant_val( index[vid], G );
    auto c3 = constant_val( index[vid], 3 );
    auto c8 = _1 << c3;
    auto off = expr::cast<GTy>(
	Gast - ( ( index[vid+_1]-index[vid] ) & ( Gast - _1 ) ) );
    auto finv = _1s >> ( off << c3 );

    return let<AIDLet1>(
	finv,
	[&]( auto fin ) {
	    return make_seq(
		idx = index[vid],
		tmp = iif( idx+c8 <= index[vid+_1],
			   gdata[idx] & fin,
			   gdata[idx] ),
		make_loop(
		    // condition
		    idx < index[vid+_1] && tmp == _1s,
		    // loop body
		    make_seq(
			idx += c8,
			tmp = iif( idx+c8 <= index[vid+_1],
				   gdata[idx] & fin,
				   gdata[idx] )
			),
		    // final value
		    cast<VID>(
			let<AIDLet2>(
			    idx + ( make_unop_tzcnt<PTy>( ~tmp ) >> c3 ),
			    [&]( auto tz ) {
				return iif( tz >= index[vid+_1],
					    tz - index[vid],
					    index[vid+_1] - index[vid] );
			    } )
			) ) ); } );
}

} // namespace expr

#endif // GRAPTOR_DSL_AST_CTRL_H
