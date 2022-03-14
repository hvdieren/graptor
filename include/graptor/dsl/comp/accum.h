// -*- c++ -*-
#ifndef GRAPTOR_DSL_COMP_ACCUM_H
#define GRAPTOR_DSL_COMP_ACCUM_H

/***********************************************************************
 * Privatizing accumulators (per-partition copies)
 ***********************************************************************/

namespace expr {

namespace detail {

// Construct an expression for a scalar accumulator index
template<typename Tr, value_kind VKind>
static constexpr
auto accum_final_index( value<Tr, VKind> ) {
    using Tr1 = typename Tr::template rebindVL<1>::type;
    return expr::value<Tr1,VKind>();
}

// TODO: this looks wrong, better: template<typename Tr1, unsigned short VL, value_kind VKind>
template<typename Tr, value_kind VKind>
static constexpr
auto accum_final_index( unop<value<Tr, VKind>,unop_incseq<Tr::VL>> ) {
    using Tr1 = typename Tr::template rebindVL<1>::type;
    return expr::value<Tr1,VKind>();
}

template<typename Tr, value_kind VKind, typename Mask>
static constexpr
auto accum_final_index( binop<value<Tr, VKind>,Mask,binop_mask> b ) {
    return accum_final_index( b.data1() );
}

} // namespace detail

#define CACHE_LINE_SIZE 64

template<unsigned CID, typename R> // redop
struct accumulator_info {
    static constexpr unsigned cid = CID;
    static constexpr unsigned next_cid = CID+1;
    static constexpr short id = R::ref_type::array_type::AID;

    using orig_redop_type = R;
    using value_type = typename orig_redop_type::val_type::type;
    using orig_ref_type = typename orig_redop_type::ref_type;

    accumulator_info( R red )
	: m_ref( red.ref() ), m_priv( nullptr ), m_alloc( nullptr ) { }

// TODO: spread out accumulators for different partitions on different cache blocks ???

    static constexpr unsigned stepping() {
	return sizeof(value_type)*orig_redop_type::VL >= CACHE_LINE_SIZE
	    ? orig_redop_type::VL
	    : (CACHE_LINE_SIZE+sizeof(value_type)-1) / sizeof(value_type);
    }
    __attribute__((always_inline))
    value_type * accum_create( const partitioner & part ) {
	constexpr unsigned short VL = orig_redop_type::VL;
	assert( m_alloc == nullptr && "Avoid double-init / memory leak" );
	// One vector per cache block. Normally expect that sizeof(value_type)
	// is power of two
	// constexpr unsigned blksize = stepping();
#define ACCUM_PAD 0
#if ACCUM_PAD
	static_assert( VL == 1, "crude trick for cache padding only works with VL=1" );
	constexpr unsigned blksize = 16;
#else
	constexpr unsigned blksize = VL;
#endif
	m_alloc = new value_type[blksize*part.get_num_partitions()+blksize-1];
	void * addr = reinterpret_cast<void *>( m_alloc );
	size_t space = sizeof(value_type)
	    * (blksize*part.get_num_partitions()+blksize-1);
	if( std::align( sizeof(value_type) * blksize,
			sizeof(value_type) * blksize * part.get_num_partitions(),
			addr, space ) )
	    m_priv = reinterpret_cast<value_type *>( addr );
	else
	    abort(); // alignment failure
			
	// There is some redundancy in this: padding is filled as well (if used)
	std::fill( m_priv, &m_priv[blksize*part.get_num_partitions()], R::unit() );
	return m_priv;
    }

    value_type * get_accum() const { return m_priv; }
    orig_ref_type get_ref() const { return m_ref; }

    template<typename PIDType>
    auto accum_reduce( PIDType pid ) const {
	array_ro<value_type, typename PIDType::type,
		 cid_to_aid(cid), array_encoding<value_type>,
		 false> priv( m_priv );

	// Figure out original accumulator
	// TODO: for vectorization, need to have vector type version at length
	//       PIDType::VL.
	auto lhs = priv[expr::increasing_val( pid )]; // m_ref;
	// Add in partial sum
	auto rhs = priv[pid];
	// constexpr unsigned short VL = orig_redop_type::VL;
	// static_assert( VL == 1, "spacing out accums over cachelines currenlty only works for VL=1" );
	// auto rhs = priv[pid * expr::constant_val(pid, stepping())];
	return make_redop( lhs, rhs, typename orig_redop_type::redop_type() );
    }

    template<typename PIDType>
    auto accum_reduce_final( PIDType pid ) const { // unused
	// reduce 1 vector of length PIDType::VL to a scalar in m_ref
	array_ro<value_type, typename PIDType::type,
		 cid_to_aid(cid), array_encoding<value_type>, false>
	    priv( m_priv );
	auto idx = m_ref.index();
	auto sidx = detail::accum_final_index( idx );
	static_assert( is_value<decltype(sidx)>::value,
		       "require a value expression here" );
	auto lhs = m_ref.replace_index( sidx );
	auto rhs = make_unop_reduce( priv[pid],
				     typename orig_redop_type::redop_type() );
	return make_storeop( lhs, rhs );
    }

    void accum_destroy() {
	if( m_priv ) {
	    delete[] m_alloc;
	    m_alloc = 0;
	    m_priv = 0;
	}
    }

private:
    orig_ref_type m_ref;
    value_type * m_priv;
    value_type * m_alloc;
};

template<unsigned ACID, typename R>
accumulator_info<ACID,R> make_accumulator_info( R r ) {
    return accumulator_info<ACID,R>( r );
}

template<unsigned cid>
static constexpr
auto accumulator_convert_rinfo( cache<> c ) {
    return c;
}

template<typename E>
struct is_accumulator {
    // Assumes E is refop
    static constexpr bool value =
	is_indexed_by_zero<E>::value; // && !is_storeop<E>::value;
};

template<unsigned cid, typename C0, typename... Cs>
static constexpr
auto accumulator_convert_rinfo(
    cache<C0,Cs...> c,
    typename std::enable_if<!is_accumulator<typename C0::ref_type>::value>::type *
    = nullptr ) {
    return accumulator_convert_rinfo<cid>( cdr( c ) );
}

template<unsigned cid, typename C0, typename... Cs>
static constexpr
auto accumulator_convert_rinfo(
    cache<C0,Cs...> c,
    typename std::enable_if<is_accumulator<typename C0::ref_type>::value>::type *
    = nullptr ) {
    return cache_cons( make_accumulator_info<cid>( car( c ).r ),
		       accumulator_convert_rinfo<cid+1>( cdr( c ) ) );
}

template<typename Expr>
static constexpr
auto extract_accumulators( Expr e ) {
    return accumulator_convert_rinfo<0>( 
	cache_dedup( extract_cacheable_refs_helper( e ) ) );
}

namespace detail {
template<short AID, typename Accum>
struct accum_updates;

template<short AID>
struct accum_updates<AID,cache<>> {
    static constexpr bool value = false;
};

template<short AID, typename C, typename... Cs>
struct accum_updates<AID,cache<C,Cs...>> {
    static constexpr bool value
	= C::id == AID || accum_updates<AID,cache<Cs...>>::value;
};

template<size_t pos, typename... Cs>
__attribute__((always_inline))
inline typename std::enable_if<(pos == 0)>::type
accum_create( const partitioner & part, cache<Cs...> & c ) {
    std::get<pos>( c.t ).accum_create( part );
}

template<size_t pos, typename... Cs>
__attribute__((always_inline))
inline typename std::enable_if<(pos > 0)>::type
accum_create( const partitioner & part, cache<Cs...> & c ) {
    std::get<pos>( c.t ).accum_create( part );
    accum_create<pos-1>( part, c );
}


template<size_t pos, typename... Cs>
inline typename std::enable_if<(pos == 0)>::type
accum_destroy( cache<Cs...> & c ) {
    std::get<pos>( c.t ).accum_destroy();
}

template<size_t pos, typename... Cs>
typename std::enable_if<(pos > 0)>::type
accum_destroy( cache<Cs...> & c ) {
    std::get<pos>( c.t ).accum_destroy();
    accum_destroy<pos-1>( c );
}

template<int pos, typename PIDType, typename... Cs>
auto accum_reduce( PIDType pid, const cache<Cs...> & c,
		   typename std::enable_if<(pos==sizeof...(Cs)-1)>::type * = nullptr ) {
    return std::get<pos>( c.t ).accum_reduce( pid );
}

template<int pos, typename PIDType, typename... Cs>
auto accum_reduce( PIDType pid, const cache<Cs...> & c,
		   typename std::enable_if<(pos<sizeof...(Cs)-1)>::type * = nullptr ) {
    return make_seq( std::get<pos>( c.t ).accum_reduce( pid ),
		     accum_reduce<pos+1>( pid, c ) );
}
template<int pos, typename PIDType, typename... Cs>
auto accum_reduce_final( PIDType pid, const cache<Cs...> & c,
			 typename std::enable_if<(pos==sizeof...(Cs)-1)>::type * = nullptr ) {
    return std::get<pos>( c.t ).accum_reduce_final( pid );
}

template<int pos, typename PIDType, typename... Cs>
auto accum_reduce_final( PIDType pid, const cache<Cs...> & c,
			 typename std::enable_if<(pos<sizeof...(Cs)-1)>::type * = nullptr ) {
    return make_seq( std::get<pos>( c.t ).accum_reduce_final( pid ),
		     accum_reduce_final<pos+1>( pid, c ) );
}
}

template<typename... Cs>
__attribute__((always_inline))
inline void accum_create( const partitioner & part, cache<Cs...> & c ) {
    if constexpr ( sizeof...(Cs) > 0 )
	detail::accum_create<(sizeof...(Cs))-1>( part, c );
}

inline void accum_destroy( cache<> & ) { }

template<typename... Cs>
void accum_destroy( cache<Cs...> & c ) {
    detail::accum_destroy<(sizeof...(Cs))-1>( c );
}

template<typename PIDType>
noop accumulate_privatized_accumulators( PIDType pid, const cache<> & c ) {
    return noop();
}

template<typename PIDType, typename... Cs>
auto accumulate_privatized_accumulators( PIDType pid, const cache<Cs...> & c ) {
    return detail::accum_reduce<0>( pid, c );
}

template<typename PIDType>
noop final_accumulate_privatized_accumulators( PIDType pid, const cache<> & c ) {
    return noop();
}

template<typename PIDType, typename... Cs>
auto final_accumulate_privatized_accumulators( PIDType pid, const cache<Cs...> & c ) {
    return detail::accum_reduce_final<0>( pid, c );
}

} // namespace expr

#endif // GRAPTOR_DSL_COMP_ACCUM_H
