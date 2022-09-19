// -*- c++ -*-
#ifndef GRAPTOR_DSL_VALUEMAP_H
#define GRAPTOR_DSL_VALUEMAP_H

namespace expr {
    
// Make this more intelligent, a tuple of pairs kind - type+value
// and do a lookup in this. All resolved at compile-time, but safe in case
// a parameter is undefined, as it can ensure that the lookup fails at
// compile-time.
// Operations:
// - concatenate, push_back
// - check presence (avoid duplicates in construction)
// - lookup (how? vk_src, vk_dst, but how identify caches?)

// TODO: might be handier to encode this as a std::tuple<>, potentially
//       merge with cache

// TODO: consider if we can remove smk and dmk, else specialise to case w/o them
template<unsigned Index, typename ValueTy>
struct map_entry {
    using index_type = unsigned;
    using value_type = ValueTy;

    static constexpr index_type index = Index;

    map_entry() { }
    explicit map_entry( const value_type & val ) : m_val( val ) { }
    explicit map_entry( value_type && val )
	: m_val( std::forward<value_type>( val ) ) { }

    value_type get() const { return m_val; }
    value_type & get() { return m_val; }

    template<unsigned Dist>
    map_entry<Index+Dist,value_type> shift() const {
	return map_entry<Index+Dist,value_type>( m_val );
    }

private:
    value_type m_val;
};


struct value_map_entry { };

template<typename Tr_, layout_t Layout, value_kind vkind_>
struct value_map_entry_vec : value_map_entry {
    using Tr = Tr_;
    using vector_type = simd::vec<Tr,Layout>;
    using index_type = value_kind;

    static constexpr unsigned short VL = Tr::VL_;
    static constexpr value_kind vkind = vkind_;
    static constexpr index_type index = vkind_;

    value_map_entry_vec() { }
    explicit value_map_entry_vec( vector_type val ) : m_val( val ) { }

    vector_type get() const { return m_val; }
private:
    vector_type m_val;
};

template<typename Tr_, value_kind vkind_>
struct value_map_entry_vec<Tr_,simd::lo_variable,vkind_> : value_map_entry {
    using Tr = Tr_;
    using vector_type = simd_vector<typename Tr::member_type,Tr::VL>;
    using index_type = value_kind;

    static constexpr unsigned short VL = Tr::VL_;
    static constexpr value_kind vkind = vkind_;
    static constexpr index_type index = vkind_;

    value_map_entry_vec() { }
    explicit value_map_entry_vec( vector_type val ) : m_val( val ) { }

    vector_type get() const { return m_val; }
private:
    vector_type m_val;
};

template<typename Tr, value_kind vkind_>
struct value_map_entry_mask : value_map_entry {
    using data_type = Tr;
    using vector_type = simd::detail::mask_impl<data_type>;
    using index_type = value_kind;

    static constexpr unsigned short W = data_type::W;
    static constexpr unsigned short VL = data_type::VL;
    static constexpr value_kind vkind = vkind_;
    static constexpr index_type index = vkind_;

    value_map_entry_mask() { }
    explicit value_map_entry_mask( vector_type val ) : m_val( val ) { }

    vector_type get() const { return m_val; }
private:
    vector_type m_val;
};

template<typename Head, typename Tail = void>
struct value_map_cons
{
    using head_type = Head;
    using index_type = typename head_type::index_type;

    constexpr value_map_cons( Head head, Tail tail )
	: m_head( head ), m_tail( tail ) { }

    template<value_kind Index>
    auto get( typename std::enable_if<Head::index == Index>::type * = nullptr ) const {
	return m_head.get();
    }
    template<value_kind Index>
    auto get( typename std::enable_if<Head::index != Index>::type * = nullptr ) const {
	return m_tail.template get<Index>();
    }

private:
    Head m_head;
    Tail m_tail;
};

template<typename Head>
struct value_map_cons<Head,void>
{
    using head_type = Head;
    using index_type = typename head_type::index_type;

    constexpr value_map_cons( head_type head ) : m_head( head ) { }

    template<value_kind Index>
    auto get( typename std::enable_if<Head::index == Index>::type * = nullptr ) const {
	return m_head.get();
    }

private:
    head_type m_head;
};

template<typename Head, typename Tail = void>
struct map_cons
{
    using head_type = Head;
    using index_type = typename head_type::index_type;

    constexpr map_cons( Head head, Tail tail )
	: m_head( head ), m_tail( tail ) { }

    template<unsigned Index>
    auto get( typename std::enable_if<head_type::index == Index>::type * = nullptr ) const {
	return m_head.get();
    }
    template<unsigned Index>
    auto & get( typename std::enable_if<head_type::index == Index>::type * = nullptr ) {
	return m_head.get();
    }
    template<unsigned Index>
    auto get( typename std::enable_if<head_type::index != Index>::type * = nullptr ) const {
	return m_tail.template get<Index>();
    }
    template<unsigned Index>
    auto & get( typename std::enable_if<head_type::index != Index>::type * = nullptr ) {
	return m_tail.template get<Index>();
    }

private:
    Head m_head;
    Tail m_tail;
};

template<typename Head>
struct map_cons<Head,void>
{
    using head_type = Head;
    using index_type = typename head_type::index_type;

    constexpr map_cons( head_type head ) : m_head( head ) { }

    template<unsigned Index>
    auto get( typename std::enable_if<Head::index == Index>::type * = nullptr ) const {
	return m_head.get();
    }
    template<unsigned Index>
    auto & get( typename std::enable_if<Head::index == Index>::type * = nullptr ) {
	return m_head.get();
    }

private:
    head_type m_head;
};


template<typename Head>
struct map_chain
{
    using head_type = Head;
    using index_type = typename head_type::index_type;

    constexpr map_chain( head_type head ) : m_head( head ) { }

    template<index_type Index>
    auto get() const { return m_head.template get<Index>(); }
    template<index_type Index>
    auto & get() { return m_head.template get<Index>(); }

    head_type get_chain() const { return m_head; }

private:
    head_type m_head;
};

// TODO:
// * template arguments VLS and VLD should be inferred from the entries
template<unsigned short VLS_, unsigned short VLD_, typename Head>
struct value_map_chain
{
    static constexpr unsigned short VLS = VLS_;
    static constexpr unsigned short VLD = VLD_;
    using head_type = Head;
    
    constexpr value_map_chain( Head head ) : m_head( head ) { }

    template<value_kind vkind>
    auto get() const { return m_head.template get<vkind>(); }

    auto source() const { return get<vk_src>(); }
    auto source_mask() const { return get<vk_smk>(); }
    auto destination() const { return get<vk_dst>(); }
    auto destination_mask() const { return get<vk_dmk>(); }

private:
    Head m_head;
};

template<typename T>
using is_value_map_entry = std::is_base_of<value_map_entry,T>;

template<value_kind vkind, typename T, unsigned short VL>
__attribute__((always_inline))
static inline constexpr auto create_entry( simd_vector<T,VL> val ) {
    return value_map_entry_vec<simd::ty<T,VL>,simd::lo_variable,vkind>( val );
}

template<value_kind vkind, typename Tr, simd::layout_t Layout>
__attribute__((always_inline))
static inline constexpr auto create_entry( simd::detail::vec<Tr,Layout> val ) {
    return value_map_entry_vec<Tr,Layout,vkind>( val );
}

template<value_kind vkind, typename Tr>
__attribute__((always_inline))
static inline constexpr auto create_entry( simd::detail::mask_impl<Tr> val ) {
    return value_map_entry_mask<Tr,vkind>( val );
}

template<unsigned Index, typename ValueTy>
static inline constexpr auto create_map_entry( ValueTy val ) {
    return map_entry<Index,ValueTy>( val );
}

template<typename Head>
static inline constexpr auto create_map_chain( Head head ) {
    return map_chain<Head>( head );
}

template<typename... Entries>
class map_new {
public:
    static constexpr size_t num_entries = sizeof...(Entries);
    
    map_new( Entries &&... entries )
	: m_entries( std::forward<Entries>( entries )... ) { }

    map_new( const map_new<Entries...> & map )
	: m_entries( map.m_entries ) { }
    map_new( map_new<Entries...> && map )
	: m_entries( std::move( map.m_entries ) ) { }

private:
    template<typename...>
    friend class map_new;

    map_new( const std::tuple<Entries...> & t ) : m_entries( t ) { }
    map_new( std::tuple<Entries...> && t )
	: m_entries( std::forward<std::tuple<Entries...>>( t ) ) { }

public:
    template<unsigned Index>
    auto get() const { return find_helper<Index,0,Entries...>().get(); }
    template<unsigned Index>
    auto & get() { return find_helper<Index,0,Entries...>().get(); }

    auto get_entries() const { return m_entries; }

    template<typename Entry>
    map_new<Entry, Entries...> copy_and_add( Entry && e ) const {
	return map_new<Entry,Entries...>(
	    std::tuple_cat( std::tuple<Entry>( std::forward<Entry>( e ) ),
			    m_entries ) );
    }

    static map_new<Entries...> from_tuple( std::tuple<Entries...> && t ) {
	return map_new<Entries...>( std::forward<std::tuple<Entries...>>( t ) );
    }

private:
    template<unsigned Index, unsigned int Pos,
	     typename FEntry, typename... FEntries>
    auto find_helper() const {
	if constexpr ( FEntry::index == Index ) {
	    return std::get<Pos>( m_entries );
	} else {
	    return find_helper<Index,Pos+1,FEntries...>();
	}
    }

    template<unsigned Index, unsigned int Pos,
	     typename FEntry, typename... FEntries>
    auto & find_helper() {
	if constexpr ( FEntry::index == Index ) {
	    return std::get<Pos>( m_entries );
	} else {
	    return find_helper<Index,Pos+1,FEntries...>();
	}
    }

private:
    std::tuple<Entries...> m_entries;
};

template<typename... Entries>
auto create_map_new( Entries &&... entry ) {
    return map_new<Entries...>( std::forward<Entries>( entry )... );
}

template<typename... Entries>
auto create_map_new_from_tuple( std::tuple<Entries...> && entries ) {
    return map_new<Entries...>::from_tuple(
	std::forward<std::tuple<Entries...>>( entries ) );
}

template<unsigned Dist, std::size_t...Ns, typename... Entries>
auto map_shift_helper( std::index_sequence<Ns...>,
		       std::tuple<Entries...> && entries ) {
    return create_map_new(
	std::get<Ns>( std::forward<std::tuple<Entries...>>( entries )
	    ).template shift<Dist>()... );
}

template<unsigned Dist, typename... Entries>
auto map_shift( const map_new<Entries...> & map ) {
    return map_shift_helper<Dist>( std::index_sequence_for<Entries...>(),
				   map.get_entries() );
}

template<typename Map1, typename Map2>
auto map_merge( Map1 && m1, Map2 && m2 ) {
    return create_map_new_from_tuple(
	std::tuple_cat( m1.get_entries(), m2.get_entries() ) );
}

template<typename... Es>
static inline auto create_map( Es &&... es ) {
    return map_new<Es...>( std::forward<Es>( es )... );
}

template<unsigned... Index, typename... ValueTy>
static inline auto create_map2( ValueTy... args ) {
    return create_map( create_map_entry<Index>( args )... );
}

template<unsigned QIndex, typename Map, typename = void>
struct map_contains : public std::false_type { };

template<unsigned QIndex, typename Entry, typename... Entries>
struct map_contains<QIndex,map_new<Entry,Entries...>> {
    static constexpr bool value = ( Entry::index == QIndex ) ?
	true : map_contains<QIndex,map_new<Entries...>>::value;
};

template<std::size_t... Ns, typename... Ts>
static inline constexpr auto
tuple_cdr_impl( std::index_sequence<Ns...>, const std::tuple<Ts...> & t ) {
   return std::make_tuple( std::get<Ns+1u>(t)... );
}

template<typename T0, typename... Ts>
static inline constexpr auto
tuple_cdr( const std::tuple<T0,Ts...> & t ) {
    return tuple_cdr_impl( std::make_index_sequence<sizeof...(Ts)>(), t );
}

template<unsigned Index, typename Entry, typename... Entries, typename T>
static inline constexpr auto
map_replace_or_set_helper( const tuple<Entry,Entries...> & t, T * p ) {
    if constexpr ( Entry::index == Index ) {
	return std::tuple_cat( std::make_tuple( create_map_entry<Index>( p ) ),
			       tuple_cdr( t ) );
    } else {
	return std::tuple_cat( std::make_tuple( std::get<0u>( t ) ),
			       map_replace_or_set_helper<Index>( tuple_cdr( t ), p ) );
    }
}

template<unsigned Index, typename Map, typename T>
static inline constexpr auto
map_replace_or_set( const Map & m, T * p ) {
    return create_map_new_from_tuple(
	map_replace_or_set_helper<Index>( m.get_entries(), p ) );
}

template<unsigned Index, typename Map, typename T>
constexpr auto
map_set_if_absent( Map && m, T * p,
		   typename std::enable_if<map_contains<Index,Map>::value>::type * = nullptr ) {
    if( p == nullptr || m.template get<Index>() != nullptr )
	return std::forward<Map>( m );
    else
	return map_replace_or_set<Index>( std::forward<Map>( m ), p );
}

template<unsigned Index, typename Map, typename T>
static inline constexpr auto
map_set_if_absent( Map && m, T * p,
		   typename std::enable_if<!map_contains<Index,Map>::value>::type * = nullptr ) {
    return m.copy_and_add( create_map_entry<Index>( p ) );
}

// Case with nothing to override
template<typename... Base>
static inline constexpr auto
map_replace_all_helper( const std::tuple<Base...> & m_base,
			const std::tuple<> & m_override ) {
    return m_base;
}


// Only case with one element implemented for now
template<typename... Base, typename Entry>
static inline constexpr auto
map_replace_all_helper( const std::tuple<Base...> & m_base,
			const std::tuple<Entry> & m_override ) {
    return map_replace_or_set_helper<Entry::index>(
	m_base, std::get<0>( m_override ).get() );
}

template<typename Map1, typename Map2>
static inline constexpr auto
map_replace_all( const Map1 & m_base, const Map2 & m_override ) {
    return create_map_new_from_tuple(
	map_replace_all_helper( m_base.get_entries(), m_override.get_entries() ) );
}

/***********************************************************************
 * Alternative approach
 ***********************************************************************/

template<unsigned short VLD_, typename... Entries>
class value_map_new {
public:
    static constexpr unsigned short VLD = VLD_; // still needd in cache

    value_map_new( Entries &&... entries )
	: m_entries( std::forward<Entries>( entries )... ) { }

    template<value_kind vkind>
    auto get() const { return find_helper<vkind,0,Entries...>().get(); }

private:
    template<value_kind VKind, unsigned int Pos,
	     typename FEntry, typename... FEntries>
    auto find_helper() const {
	if constexpr ( FEntry::vkind == VKind ) {
	    return std::get<Pos>( m_entries );
	} else {
	    return find_helper<VKind,Pos+1,FEntries...>();
	}
    }

private:
    std::tuple<Entries...> m_entries;
};

template<unsigned short VLD, typename... Entries>
auto create_value_map_new( Entries &&... entry ) {
    return value_map_new<VLD,Entries...>( std::forward<Entries>( entry )... );
}

template<unsigned short VLS, unsigned short VLD,
	 value_kind... vkind, typename... VecMask>
__attribute__((always_inline))
static inline auto create_value_map2( VecMask &&... args ) {
    return create_value_map_new<VLD>( create_entry<vkind>( args )... );
}

template<unsigned short VLD,
	 value_kind... vkind, typename... VecMask>
__attribute__((always_inline))
static inline auto create_value_map_new2( VecMask &&... args ) {
    return create_value_map_new<VLD>( create_entry<vkind>( args )... );
}

/***********************************************************************
 * Deprecated definitions.
 ***********************************************************************/

template<typename S_, unsigned short VLS_,
	 typename D_, unsigned short VLD_>
struct value_map_full {
    static constexpr unsigned short VLS = VLS_;
    static constexpr unsigned short VLD = VLD_;
    using S = S_;
    using D = D_;

    // using _S_type_sel = detail::type_sel<S,VLS>;
    // using _D_type_sel = detail::type_sel<D,VLD>;

    using src_vector_type = simd_vector<S,VLS>;
    using src_mask_type = typename src_vector_type::simd_mask_type; // simd_mask<sizeof(S),VLS>;
    using dst_vector_type = simd_vector<D,VLD>;
    using dst_mask_type = typename dst_vector_type::simd_mask_type; // simd_mask<sizeof(D),VLD>;
    using self_type = value_map_full<S, VLS, D, VLD>;
    
    value_map_full() { }
    explicit value_map_full( src_vector_type src, src_mask_type smk,
			     dst_vector_type dst, dst_mask_type dmk )
	: m_src( src ), m_smk( smk ), m_dst( dst ), m_dmk( dmk ) { }

    auto source() const { return m_src; }
    auto source_mask() const { return m_smk; }
    auto destination() const { return m_dst; }
    auto destination_mask() const { return m_dmk; }

private:
    src_vector_type m_src;
    src_mask_type   m_smk;
    dst_vector_type m_dst;
    dst_mask_type   m_dmk;
};

} // namespace expr

#endif // GRAPTOR_DSL_VALUEMAP_H
