// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_CONTRACT_VERTEX_SET_H
#define GRAPTOR_GRAPH_CONTRACT_VERTEX_SET_H

#include <vector>
#include <algorithm>
#include <ostream>

namespace contract {
    
namespace detail {

template<bool complete, typename It, typename Inserter>
void intersect_merge( It ls, It le, It rs, It re, Inserter & insert ) {
    It l = ls;
    It r = rs;

    while( l != le && r != re ) {
	if( *l == *r ) {
	    if( !insert( *l++ ) )
		return;
	    ++r;
	} else if( *l < *r ) {
	    // if( !insert.ignore_left( *l ) ) return;
	    ++l;
	} else {
	    // if( !insert.ignore_right( *r ) ) return;
	    ++r;
	}
    }
/*
    if constexpr ( complete ) {
	while( l != le ) {
	    if( !insert.ignore_left( *l ) )
		return;
	    ++l;
	}
	while( r != re ) {
	    if( !insert.ignore_right( *r ) )
		return;
	    ++r;
	}
    }
*/
}

template<typename It, typename Inserter>
void intersect_search( It ls, It le, It rs, It re, Inserter & insert ) {
    It l = ls;
    It r = rs;

    while( r != re ) {
	const VID * f = std::lower_bound( l, le, *r );
	if( f != le && *f == *r ) {
	    if( !insert( *f ) )
		return;
	    // the next element we will search for is larger than *r
	    // and *f == *r, so f need not be considered
	    l = f+1;
	    // } else {
	    // if( !insert.ignore_right( *r ) ) return;
	}
	r++;
    }
}

template<typename It, typename Inserter>
void intersect_tmpl( It ls, It le, It rs, It re, Inserter && insert ) {
    auto nl = std::distance( ls, le );
    auto nr = std::distance( rs, re );
    if( nr * nr < nl )
	detail::intersect_search( ls, le, rs, re, insert );
    else if( nl * nl < nr )
	detail::intersect_search( rs, re, ls, le, insert );
    else
	detail::intersect_merge<false>( ls, le, rs, re, insert );
}

template<typename VID>
struct append_list {
    append_list( VID * p ) : m_ptr( p ) { }

    bool operator() ( VID v ) {
	*m_ptr++ = v;
	return true;
    }

    bool ignore_left( VID ) const { return true; }
    bool ignore_right( VID ) const { return true; }

    VID * get_pointer() const  { return m_ptr; }
    
private:
    VID * m_ptr;
};

template<typename VID>
struct append_list_3way {
    append_list_3way( VID * p, VID * l, VID * r )
	: m_ptr( p ), m_left( l ), m_right( r ) { }

    bool operator() ( VID v ) {
	*m_ptr++ = v;
	return true;
    }

    bool ignore_left( VID v ) {
	*m_left++ = v;
	return true;
    }

    bool ignore_right( VID v ) {
	*m_right++ = v;
	return true;
    }

    VID * get_pointer() const  { return m_ptr; }
    VID * get_left() const { return m_left; }
    VID * get_right() const { return m_right; }
    
private:
    VID * m_ptr, * m_left, * m_right;
};

template<typename VID>
struct count_size {
    count_size() : m_size( 0 ) { }

    bool operator() ( VID v ) {
	++m_size;
	return true;
    }

    bool ignore_left( VID ) const { return true; }
    bool ignore_right( VID ) const { return true; }

    VID size() const { return m_size; }
    
private:
    VID m_size;
};

template<typename VID, typename Fn>
struct count_size_cond {
    count_size_cond( Fn fn ) : m_size( 0 ), m_fn( fn ) { }

    bool operator() ( VID v ) {
	if( m_fn( v ) )
	    ++m_size;
	return true;
    }

    bool ignore_left( VID ) const { return true; }
    bool ignore_right( VID ) const { return true; }

    VID size() const { return m_size; }
    
private:
    VID m_size;
    Fn m_fn;
};

template<typename VID>
struct check_empty {
    check_empty() : m_empty( true ) { }

    bool operator() ( VID v ) {
	m_empty = false;
	return false;
    }

    bool ignore_left( VID ) const { return true; }
    bool ignore_right( VID ) const { return true; }

    VID is_empty() const { return m_empty; }
    
private:
    VID m_empty;
};

// This method assumes both lists are sorted
template<typename VID>
VID intersect( const VID * const candidates, VID num_candidates,
	       const VID * const vertices, VID num_vertices,
	       VID * new_candidates ) {
    append_list<VID> fn( new_candidates );
    intersect_tmpl( candidates, &candidates[num_candidates],
		    vertices, &vertices[num_vertices], fn );
    return fn.get_pointer() - new_candidates;
}

template<typename VID>
auto intersect_3way( const VID * const ls, VID nl,
		     const VID * const rs, VID nr,
		     VID * new_candidates, VID * dropped_left,
		     VID * dropped_right ) {
    append_list_3way<VID> fn( new_candidates, dropped_left, dropped_right );
    detail::intersect_merge<true>( ls, ls+nl, rs, rs+nr, fn );
    return std::make_tuple( fn.get_pointer() - new_candidates,
			    fn.get_left() - dropped_left,
			    fn.get_right() - dropped_right );
}

template<typename VID>
VID intersection_size(
    const VID * candidates, VID num_candidates,
    const VID * const vertices, VID num_vertices ) {
    count_size<VID> fn;
    intersect_tmpl( candidates, &candidates[num_candidates],
		    vertices, &vertices[num_vertices], fn );
    return fn.size();
}

template<typename VID, typename Fn>
VID intersection_size(
    const VID * candidates, VID num_candidates,
    const VID * const vertices, VID num_vertices, Fn ffn ) {
    count_size_cond<VID,Fn> fn( ffn );
    intersect_tmpl( candidates, &candidates[num_candidates],
		    vertices, &vertices[num_vertices], fn );
    return fn.size();
}

template<typename VID>
bool intersection_empty(
    const VID * candidates, VID num_candidates,
    const VID * const vertices, VID num_vertices ) {
    check_empty<VID> fn;
    intersect_tmpl( candidates, &candidates[num_candidates],
		    vertices, &vertices[num_vertices], fn );
    return fn.is_empty();
}

size_t set_union(
    const VID * const candidates, VID num_candidates,
    const VID * const vertices, VID num_vertices,
    VID * out ) {
    const VID * bu = candidates;
    const VID * eu = candidates + num_candidates;
    const VID * bv = vertices;
    const VID * ev = vertices + num_vertices;
    VID * p = out;
    while( bu != eu && bv != ev ) {
	if( *bu == *bv ) {
	    *p++ = *bu;
	    ++bu;
	    ++bv;
	} else if( *bu < *bv )
	    *p++ = *bu++;
	else
	    *p++ = *bv++;
    }
    while( bu != eu )
	*p++ = *bu++;
    while( bv != ev )
	*p++ = *bv++;

    return p - out;
}

} // namespace detail

template<typename VID>
class vertex_set {
public:
    vertex_set() { }
    vertex_set( VID upper_bound ) { reserve( upper_bound ); }
    void reserve( VID size ) { m_vertices.reserve( size ); }
    void resize( VID size ) { m_vertices.resize( size ); }
    VID size() const {
	return m_vertices.size();
    }
    VID get( VID idx ) const { return m_vertices[idx]; }
    const VID * begin() const { return &*m_vertices.begin(); }
    VID * begin() { return &*m_vertices.begin(); }
    const VID * end() const { return &*m_vertices.end(); }
    VID * end() { return &*m_vertices.end(); }

    void swap( vertex_set<VID> & s ) {
	std::swap( m_vertices, s.m_vertices );
    }
    void swap( vertex_set<VID> && s ) {
	std::swap( m_vertices, s.m_vertices );
    }

    // It is assumed that the vertex set is maintained in sort order.
    // The push  method should only be used if the caller can ensure this.
    void push( VID v ) { m_vertices.push_back( v ); }
    template<typename Iter>
    void push( const Iter start, const Iter end ) {
	reserve( size() + std::distance( start, end ) );
	for( Iter I=start; I != end; ++I )
	    push( *I );
    }
    template<typename Iter, typename Filter>
    void push( const Iter start, const Iter end, const Filter & filter ) {
	reserve( size() + std::distance( start, end ) );
	for( Iter I=start; I != end; ++I )
	    if( filter( *I ) )
		push( *I );
    }
    void pop() { m_vertices.pop_back(); }

    // Add does not know whether the vertex set will remain sorted
    // should we insert the vertex at the end like push.
    void add( VID v ) {
	auto p = std::lower_bound( m_vertices.begin(), m_vertices.end(), v );
	m_vertices.insert( p, v );
    }
    template<typename Iter>
    void add( const Iter start, const Iter end ) {
/*
	// Could do better in linear time using merge
	reserve( size() + std::distance( start, end ) );
	for( auto I=start; I != end; ++I )
	    add( *I );
*/
	std::vector<VID> u;
	u.resize( std::distance( start, end ) + m_vertices.size() );
	size_t n
	    = detail::set_union( &*m_vertices.begin(), m_vertices.size(),
				 start, std::distance( start, end ),
				 &*u.begin() );
	u.resize( n );
	m_vertices.swap( u );
    }
    void add( const vertex_set<VID> & S ) {
	add( S.begin(), S.end() );
    }

    bool empty() const {
	return m_vertices.empty();
    }

    bool contains( VID v ) const {
	auto p = std::lower_bound( m_vertices.begin(), m_vertices.end(), v );
	return p != m_vertices.end() && *p == v;
    }
    
    // Assumes set is sorted
    void remove( VID elm ) {
	auto p = std::lower_bound( m_vertices.begin(), m_vertices.end(), elm );
	if( p != m_vertices.end() && *p == elm )
	    m_vertices.erase( p );
    }

    template<typename Iter>
    void remove( const Iter start, const Iter end ) {
	for( auto I=start; I != end; ++I )
	    remove( *I );
    }

    void remove( const vertex_set<VID> & s ) {
	remove( s.begin(), s.end() );
    }

    // Specifically for sorting cliques by size
    // TODO: move to derived class
    bool operator < ( const vertex_set<VID> & c ) const {
	return size() > c.size();
    }

    // Intersection operations
    VID intersect( const VID * p, VID n, VID * out ) const {
	return detail::intersect( begin(), size(), p, n, out );
    }
    vertex_set<VID> intersect( const VID * p, VID n ) const {
	VID sz = std::min( n, size() );
	vertex_set<VID> result( sz );
	result.resize( sz );
	result.resize( intersect( p, n, result.begin() ) );
	return result;
    }
#if 0
    void intersect( const vertex_set<VID> & s,
		    vertex_set<VID> & i,
		    vertex_set<VID> & l,
		    vertex_set<VID> & r ) {
	VID sz = std::min( s.size(), size() );
	i.resize( sz );
	l.resize( size() );
	r.resize( s.size() );
	size_t si, sl, sr;
	std::tie( si, sl, sr )
	    = detail::intersect_3way( begin(), size(), s.begin(), s.size(),
				      i.begin(), l.begin(), r.begin() );
	i.resize( si );
	l.resize( sl );
	r.resize( sr );
    }
#endif
		    
    VID intersection_size( const VID * p, VID n ) const {
	return detail::intersection_size( begin(), size(), p, n );
    }
    VID intersection_size( const vertex_set<VID> & s ) const {
	return intersection_size( s.begin(), s.size() );
    }
    bool intersection_empty( const VID * p, VID n) const {
	return detail::intersection_empty( begin(), size(), p, n );
    }
    bool intersection_empty( const vertex_set<VID> & S ) const {
	return intersection_empty( S.begin(), S.size() );
    }

    void clear() {
	m_vertices.clear();
    }
    
private:
    std::vector<VID> m_vertices;
};

template<typename VID>
ostream & operator << ( ostream & os, const vertex_set<VID> & s ) {
    os << "{ #" << s.size() << ": ";
    for( auto I=s.begin(), E=s.end(); I != E; ++I )
	os << ' ' << *I;
    os << " }";
    return os;
}

} // namespace contract

#endif // GRAPTOR_GRAPH_CONTRACT_VERTEX_SET_H
