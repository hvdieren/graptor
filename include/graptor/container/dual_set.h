// -*- c++ -*-
#ifndef GRAPTOR_CONTAINER_DUAL_SET_H
#define GRAPTOR_CONTAINER_DUAL_SET_H

namespace graptor {

// A dual representation of a set as a sequential collection Seq
// and a hash set Hash.
// Note: the sequence is copied (by value) as it is small and assumed not to
//       own its content (e.g., array_slice). The hash set is assumed to
//       own its content, so a reference is kept.
template<typename Seq, typename Hash>
struct dual_set {
    using seq_type = Seq;
    using hash_type = Hash;

    dual_set( seq_type seq, hash_type & hash )
	: m_seq( seq ), m_hash( hash ) { }

    auto size() const { return m_seq.size(); }
    
    auto begin() { return m_seq.begin(); }
    const auto begin() const { return m_seq.begin(); }
    auto end() { return m_seq.end(); }
    const auto end() const { return m_seq.end(); }

    seq_type get_seq() const { return m_seq; }
    hash_type & get_hash() { return m_hash; }
    const hash_type & get_hash() const { return m_hash; }

    template<typename T>
    bool contains( T value ) const { return m_hash.contains( value ); }

private:
    seq_type m_seq;
    hash_type & m_hash;
};

} // namespace graptor

#endif // GRAPTOR_CONTAINER_DUAL_SET_H
