// -*- c++ -*-
#ifndef GRAPTOR_CONTAINER_DUAL_SET_H
#define GRAPTOR_CONTAINER_DUAL_SET_H

namespace graptor {

template<typename S>
struct is_hash_set {
    static constexpr bool value = requires( const S & s ) {
	s.contains( typename S::type(0) );
    };
};

template<typename S>
constexpr bool is_hash_set_v = is_hash_set<S>::value;

template<typename S>
struct is_hash_table {
    static constexpr bool value = requires( const S & s ) {
	s.lookup( typename S::type(0) );
    };
};

template<typename S>
constexpr bool is_hash_table_v = is_hash_table<S>::value;

template<typename S>
struct is_multi_hash_set {
    static constexpr bool value = requires( const S & s ) {
	s.template multi_contains<typename S::type,8,target::mt_vmask>(
	    vector_type_traits_vl<typename S::type,8>::setzero(),
	    target::mt_vmask() );
    };
};

template<typename S>
constexpr bool is_multi_hash_set_v = is_multi_hash_set<S>::value;

template<typename S>
struct is_multi_hash_table {
    static constexpr bool value = requires( const S & s ) {
	s.template multi_lookup<typename S::type,8>(
	    vector_type_traits_vl<typename S::type,8>::setzero() );
    };
};

template<typename S>
constexpr bool is_multi_hash_table_v = is_multi_hash_table<S>::value;

// A dual representation of a set as a sequential collection Seq
// and a hash set Hash.
// Note: the sequence is copied (by value) as it is small and assumed not to
//       own its content (e.g., array_slice). The hash set is assumed to
//       own its content, so a reference is kept.
template<typename Seq, typename Hash>
struct dual_set {
    using seq_type = Seq;
    using hash_type = Hash;
    using type = std::remove_cv_t<typename seq_type::type>;

    dual_set( seq_type seq, const hash_type & hash )
	: m_seq( seq ), m_hash( hash ) { }

    auto size() const { return m_seq.size(); }
    
    auto begin() { return m_seq.begin(); }
    const auto begin() const { return m_seq.begin(); }
    auto end() { return m_seq.end(); }
    const auto end() const { return m_seq.end(); }

    seq_type get_seq() const { return m_seq; }
    const hash_type & get_hash() const { return m_hash; }

    bool contains( type value ) const { return m_hash.contains( value ); }

    type lookup( type value ) const { return m_hash.lookup( value ); }

    template<typename U, unsigned short VL, typename MT>
    auto
    multi_contains( typename vector_type_traits_vl<U,VL>::type index,
		    MT mt ) const {
	return m_hash.template multi_contains<U,VL,MT>( index, mt );
    }

    template<typename U, unsigned short VL>
    auto
    multi_lookup( typename vector_type_traits_vl<U,VL>::type index ) const {
	return m_hash.template multi_lookup<U,VL>( index );
    }

    dual_set<seq_type,hash_type> trim_r( const type * r ) const {
	return dual_set<seq_type,hash_type>( m_seq.trim_r( r ), m_hash );
    }

    dual_set<seq_type,hash_type> trim_range( type lo, type hi ) const {
	return dual_set<seq_type,hash_type>( m_seq.trim_range( lo, hi ),
					     m_hash );
    }

private:
    seq_type m_seq;
    const hash_type & m_hash;
};

template<typename Seq, typename Hash>
auto make_dual_set( Seq && seq, const Hash & hash ) {
    return dual_set<Seq,Hash>( seq, hash );
}

} // namespace graptor

#endif // GRAPTOR_CONTAINER_DUAL_SET_H
