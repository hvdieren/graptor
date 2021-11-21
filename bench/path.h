template<typename GraphType,
	 typename T, typename lVID, short VAR, typename Enc>
std::pair<T,T>
find_min_max( const GraphType & GA,
	      const api::vertexprop<T,lVID,VAR,Enc> & prop ) {
    using VID = lVID;
    using ST = typename Enc::stored_type;
    
    VID n = GA.numVertices();
    const partitioner &part = GA.get_partitioner();

    T infty = std::numeric_limits<T>::infinity();
    T tmax = (T)std::numeric_limits<ST>::max();

    T f_min = tmax;
    T f_max = -tmax;
    expr::array_ro<T,VID,VAR+1> a_min( &f_min );
    expr::array_ro<T,VID,VAR+2> a_max( &f_max );

    make_lazy_executor( part )
	.vertex_scan( [&]( auto v ) {
	    auto z =
		expr::value<simd::ty<VID,decltype(v)::VL>,expr::vk_zero>();
	    auto zero =
		expr::value<simd::ty<T,decltype(v)::VL>,expr::vk_zero>();
	    auto inf = expr::constant_val2<T>( v, infty );
	    auto amax = expr::constant_val2<T>( v, tmax );

	    auto w = prop[v];
	    return expr::set_mask( w != zero && w != inf && w != amax,
				   make_seq( a_min[z].min( w ),
					     a_max[z].max( w ) ) );
	} )
	.materialize();

    return std::make_pair( f_min, f_max );
}
