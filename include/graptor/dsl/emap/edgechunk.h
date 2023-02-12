// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_EDGE_CHUNK_H
#define GRAPTOR_DSL_EMAP_EDGE_CHUNK_H

template<bool atomic,
	 typename Cache, typename Environment, typename Expr>
static VID
process_csr_append( const VID *out, EID be, VID deg, VID srcv,
		    VID * frontier, bool *zf,
		    Cache & c, const Environment & env, const Expr & e );

template<bool atomic, update_method um,
	 typename Cache, typename Environment, typename Expr>
static void
process_csr_sparse( const VID *out, EID be, VID deg, VID srcv,
		    VID *frontier, bool *zf,
		    Cache & c, const Environment & env, const Expr & e );

template<typename lVID, typename lEID>
struct basic_edge_iterator {
    using VID = lVID;
    using EID = lEID;
    using iterator_category = std::input_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = std::pair<VID,EID>;
    using pointer           = const value_type *;
    using reference         = const value_type &;

    basic_edge_iterator( VID vstart, const EID * const index,
			 const VID * const vertices, VID num_vertices )
	: m_index( index ),
	  m_vertices( vertices ), m_num_vertices( num_vertices ) {
	if( m_num_vertices == 0 ) {
	    m_i = 0;
	    m_j = 0;
	} else {
	    m_i = vstart;
	    m_j = index[m_vertices[vstart]];
	    next();
	}
    }
    basic_edge_iterator( const EID * const index,
			 const VID * const vertices, VID num_vertices )
	: m_index( index ),
	  m_vertices( vertices ), m_num_vertices( num_vertices ),
	  m_i( m_num_vertices ), m_j( 0 ) { }

    value_type operator * () const {
	return std::make_pair( m_vertices[m_i], m_j );
    }

    basic_edge_iterator & operator ++ () { advance(); return *this; } // pre-inc
    basic_edge_iterator operator ++ ( int ) { // post-inc
	basic_edge_iterator it = *this; advance(); return it;
    }

    friend bool operator == (
	const basic_edge_iterator & a, const basic_edge_iterator & b ) {
	return a.m_i == b.m_i && a.m_j == b.m_j;
    };
    friend bool operator != (
	const basic_edge_iterator & a, const basic_edge_iterator & b ) {
	return !( a == b );
    }

private:
    void advance() {
	++m_j;
	next();
    }
    void next() {
	while( m_j == m_index[m_vertices[m_i]+1] ) {
	    if( ++m_i == m_num_vertices ) {
		m_j = 0;
		break;
	    } else {
		m_j = m_index[m_vertices[m_i]];
	    }
	}
    }

private:
    const EID * const m_index;
    const VID * const m_vertices;
    const VID m_num_vertices;
    VID m_i;
    EID m_j;
};

template<typename lVID, typename lEID>
struct trimmed_edge_iterator {
    using VID = lVID;
    using EID = lEID;
    using iterator_category = std::input_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = std::pair<VID,EID>;
    using pointer           = const value_type *;
    using reference         = const value_type &;

    // Begin iterator
    trimmed_edge_iterator( VID vstart, const EID * const index,
			   const VID * const vertices, VID num_vertices,
			   EID fstart, EID lend )
	: m_index( index ),
	  m_vertices( vertices ), m_num_vertices( num_vertices ),
	  m_fstart( fstart ), m_lend( lend ) {
	// assert( lend != 0 );
	if( m_num_vertices == 0 ) {
	    m_i = 0;
	    m_j = m_j_end = 0;
	} else {
	    // assert( vstart <= m_num_vertices );
	    m_i = vstart;
	    m_j = m_i == 0 ? m_fstart : m_index[m_vertices[m_i]];
	    m_j_end = get_j_end( m_i );
	    next();
	    // assert( m_j <= m_j_end );
	}
    }
    // End iterator
    trimmed_edge_iterator( const EID * index,
			   const VID * const vertices, VID num_vertices )
	: m_index( index ),
	  m_vertices( vertices ), m_num_vertices( num_vertices ) {
	if( m_num_vertices == 0 ) {
	    m_i = 0;
	    m_j = m_j_end = 0;
	} else {
	    m_i = m_num_vertices;
	    m_j = m_j_end = 0;
	}
	// assert( m_j <= m_j_end );
    }

    value_type operator * () const {
	return std::make_pair( m_vertices[m_i], m_j );
    }

    trimmed_edge_iterator & operator ++ () { // pre-inc
	advance();
	return *this;
    }
    trimmed_edge_iterator operator ++ ( int ) { // post-inc
	trimmed_edge_iterator it = *this;
	advance();
	return it;
    }

    friend bool operator == (
	const trimmed_edge_iterator & a, const trimmed_edge_iterator & b ) {
	return a.m_i == b.m_i && a.m_j == b.m_j;
    };
    friend bool operator != (
	const trimmed_edge_iterator & a, const trimmed_edge_iterator & b ) {
	return !( a == b );
    }

private:
    void advance() {
	++m_j;
	next();
    }
    void next() {
	while( m_j == m_j_end ) {
	    if( ++m_i == m_num_vertices ) {
		m_j = m_j_end = 0;
		break;
	    } else {
		m_j = m_index[m_vertices[m_i]];
		m_j_end = get_j_end( m_i );
	    }
	}
	// assert( m_j < m_j_end
	// || ( m_i == m_num_vertices && m_j == m_j_end ) );
    }

    EID get_j_end( EID i ) const {
	return i == m_num_vertices-1 ? m_lend : m_index[m_vertices[i]+1];
    }

private:
    const EID * const m_index;
    const VID * const m_vertices;
    const VID m_num_vertices;
    VID m_i;
    EID m_j;
    EID m_fstart;
    EID m_lend;
    EID m_j_end;
};

/************************************************************************
 * @brief A descriptor for a subset of the edges to be processed, given a set
 * of vertices and their degrees.
 *
 * The content of the class is relative to a particular graph, represented
 * in a CSx format, in particular it stores locations in the edges list
 * to identify which subset of the edges incident to a vertex are processed.
 *
 * This class does not store the list of vertices.
 *
 * @param <lVID> the vertex ID type
 * @param <lEID> the edge ID type
 ************************************************************************/
template<typename lVID, typename lEID>
class vertex_partition {
public:
    using VID = lVID;
    using EID = lEID;

    vertex_partition() : m_from( 0 ), m_to( 0 ) { }
    vertex_partition( VID from, VID to ) : m_from( from ), m_to( to ) {
	// assert( from <= to );
    }

    VID get_from() const { return m_from; }
    VID get_to() const { return m_to; }
    VID num_vertices() const { return m_to - m_from; }

    void set_from( VID from ) { m_from = from; }
    void set_to( VID to ) { m_to = to; }

    bool is_empty() const { return m_to == m_from; }

    template<typename Cache, typename Environment, typename Expr>
    void process_pull( const VID * const vertices,
		       const EID * const idx,
		       const VID * const edges,
		       Cache & c,
		       const Environment & env, 
		       const Expr & expr ) const {
	static_assert( Expr::VL == 1, "Sparse traversal requires VL == 1" );

	for( VID js=get_from(), je=get_to(), j=js; j < je; ++j ) {
	    VID v = vertices[j];
	    EID x = idx[v];
	    EID y = idx[v+1];
	    auto dst = simd::template create_scalar<simd::ty<VID,1>>( v );

	    using output_type = simd::container<typename Expr::data_type>;
	    auto output = output_type::false_mask();

	    for( EID e=x; e < y; ++e ) {
		auto src
		    = simd::template load_from<simd::ty<VID,1>>( &edges[e] );
		auto edg = simd::template create_constant<simd::ty<EID,1>>( e );
		auto m = expr::create_value_map_new<1>(
		    expr::create_entry<expr::vk_edge>( edg ),
		    expr::create_entry<expr::vk_dst>( dst ),
		    expr::create_entry<expr::vk_src>( src ) );
		// Note: CSC, sequential per vertex, no atomics required
		auto ret = env.template evaluate<false>( c, m, expr );
	    }
	}
    }

    template<typename AllCacheDesc, typename CacheDesc,
	     typename Environment, typename Expr,
	     typename AExpr>
    void process_pull( const VID * const vertices,
		       const EID * const idx,
		       const VID * const edges,
		       const AllCacheDesc & cdesc_all,
		       const CacheDesc & cdesc,
		       const Environment & env, 
		       const Expr & expr,
		       const AExpr & aexpr ) const {
	static_assert( Expr::VL == 1, "Sparse traversal requires VL == 1" );

	auto c = expr::cache_create_no_init(
	    cdesc_all, expr::create_value_map_new<1>() );

	for( VID js=get_from(), je=get_to(), j=js; j < je; ++j ) {
	    VID v = vertices[j];
	    EID x = idx[v];
	    EID y = idx[v+1];
	    auto dst = simd::template create_scalar<simd::ty<VID,1>>( v );

	    auto mdst = expr::create_value_map_new<1>(
		expr::create_entry<expr::vk_dst>( dst ) );
	    cache_init( env, c, cdesc, mdst ); // partial init

	    using output_type = simd::container<typename Expr::data_type>;
	    auto output = output_type::false_mask();

	    for( EID e=x; e < y; ++e ) {
		auto src
		    = simd::template load_from<simd::ty<VID,1>>( &edges[e] );
		auto edg = simd::template create_constant<simd::ty<EID,1>>( e );
		auto m = expr::create_value_map_new<1>(
		    expr::create_entry<expr::vk_edge>( edg ),
		    expr::create_entry<expr::vk_dst>( dst ),
		    expr::create_entry<expr::vk_src>( src ) );

		// Check if active
		auto act = env.template evaluate<false>( c, m, aexpr );
		if( !act.value().data() )
		    break;

		// Note: CSC, sequential per vertex, no atomics required
		env.template evaluate<false>( c, m, expr );
	    }

	    cache_commit( env, cdesc, c, mdst );
	}
    }


private:
    VID m_from, m_to;
};

template<typename lVID, typename lEID>
class edge_partition : public vertex_partition<lVID,lEID> {
public:
    using VID = lVID;
    using EID = lEID;
    using super = vertex_partition<lVID,lEID>;

    edge_partition() : super(), m_fstart( 0 ), m_lend( 0 ), m_offset( 0 ) { }
    edge_partition( VID from, VID to, EID fstart, EID lend, EID offset )
	: super( from, to ), m_fstart( fstart ), m_lend( lend ),
	  m_offset( offset ) {
	// assert( from <= to );
    }

    EID get_fstart() const { return m_fstart; }
    EID get_lend() const { return m_lend; }
    EID get_offset() const { return m_offset; }
    EID num_edges( const VID * vertices, const EID * idx ) const {
	EID ne = 0;
	for( VID js=super::get_from(), je=super::get_to(), j=js; j < je; ++j ) {
	    VID v = vertices[j];
	    EID x = j == js ? m_fstart : idx[v];
	    EID y = j == je-1 ? m_lend : idx[v+1];
	    VID d = y-x;
	    ne += d;
	}
	return ne;
    }

    void set_fstart( EID fstart ) { m_fstart = fstart; }
    void set_lend( EID lend ) { m_lend = lend; }
    void set_offset( EID offset ) { m_offset = offset; }

    template<bool need_atomic,
	     typename Cache, typename Environment, typename Expr>
    VID process_push( const VID * vertices,
		      VID * out_edges,
		      bool * zf,
		      const EID * idx,
		      const VID * edges,
		      Cache & c,
		      const Environment & env, 
		      const Expr & expr ) const {
	static_assert( Expr::VL == 1, "Sparse traversal requires VL == 1" );

	VID *f_out = &out_edges[m_offset];
	for( VID js=super::get_from(), je=super::get_to(), j=js; j < je; ++j ) {
	    VID v = vertices[j];
	    EID x = j == js ? m_fstart : idx[v];
	    EID y = j+1 == je ? m_lend : idx[v+1];
	    // assert( y >= x );
	    /*
	    n_out += process_csr_append<need_atomic>(
		&edges[x], x, d, v, &f_out[n_out], zf, c, env, expr );
	    */
	    auto src = simd::template create_scalar<simd::ty<VID,1>>( v );

	    for( EID e=x; e < y; ++e ) {
		auto dst = simd::template load_from<simd::ty<VID,1>>( &edges[e] );
		auto eid = simd::template create_scalar<simd::ty<EID,1>>( e );
		auto m = expr::create_value_map_new<1>(
		    expr::create_entry<expr::vk_dst>( dst ),
		    expr::create_entry<expr::vk_edge>( eid ),
		    expr::create_entry<expr::vk_src>( src ) );
		// Note: set evaluator to use atomics
		auto ret = env.template evaluate<need_atomic>( c, m, expr );
		if( ret.value().data() ) {
		    // for particular cases, no need to use zerof
		    // (e.g. count_down).
		    if constexpr ( !expr::is_single_trigger<Expr>::value ) {
			// Set frontier, once.
			if( zf[dst.data()] == 0
			    && __sync_fetch_and_or(
				(unsigned char *)&zf[dst.data()],
				(unsigned char)1 ) == 0 )
			    // first time being set
			    *f_out++ = dst.data();
		    } else {
			// first time and only time being set
			*f_out++ = dst.data();
		    }
		}
	    }
	}
	return f_out - &out_edges[m_offset];
    }

    template<bool need_atomic, update_method um,
	     typename Cache, typename Environment, typename Expr>
    void
    process_push_many( const VID * vertices,
		       const EID * idx,
		       const VID * edges,
		       bool *zf,
		       Cache & c, const Environment & env,
		       const Expr & expr ) const {
	for( VID js=super::get_from(), je=super::get_to(), j=js; j < je; ++j ) {
	    VID v = vertices[j];
	    EID x = j == js ? m_fstart : idx[v];
	    EID y = j+1 == je ? m_lend : idx[v+1];
	    // assert( y >= x );
	    VID d = y-x;
	    process_csr_sparse<need_atomic,um>(
		&edges[x], x, d, v, nullptr, zf, c, env, expr );
	}
    }

    template<typename Cache, typename Environment, typename Expr>
    void process_pull( const VID * const vertices,
		       const EID * const idx,
		       const VID * const edges,
		       Cache & c,
		       const Environment & env, 
		       const Expr & expr ) const {
	static_assert( Expr::VL == 1, "Sparse traversal requires VL == 1" );

	for( VID js=super::get_from(), je=super::get_to(), j=js; j < je; ++j ) {
	    VID v = vertices[j];
	    EID x = j == js ? m_fstart : idx[v];
	    EID y = j+1 == je ? m_lend : idx[v+1];
	    auto dst = simd::template create_scalar<simd::ty<VID,1>>( v );

	    using output_type = simd::container<typename Expr::data_type>;
	    auto output = output_type::false_mask();

	    for( EID e=x; e < y; ++e ) {
		auto src
		    = simd::template load_from<simd::ty<VID,1>>( &edges[e] );
		auto edg = simd::template create_constant<simd::ty<EID,1>>( e );
		auto m = expr::create_value_map_new<1>(
		    expr::create_entry<expr::vk_edge>( edg ),
		    expr::create_entry<expr::vk_dst>( dst ),
		    expr::create_entry<expr::vk_src>( src ) );
		// Note: CSC, sequential per vertex, no atomics required
		auto ret = env.template evaluate<false>( c, m, expr );
	    }
	}
    }

private:
    EID m_fstart, m_lend, m_offset;
};


/************************************************************************
 * @brief Calculate a partition of the set of vertices
 *
 * The method aims to edge-balance the partitions as well as possible
 * without spreading edges incident to a vertex across partitions.
 *
 * @param <mm_block> target number of edges per part
 * @param <mm_threshold> tolerance on the number of edges per part
 * @param <lVID> the vertex ID type
 * @param <lEID> the edge ID type
 * @param <RecordFn> how to record each partition
 *
 * @param[in] s an array of vertices
 * @param[in] m number of vertices in array @see s
 * @param[in] cdegree an array with cumulative sum of degrees of the vertices
 * @param[in] idx index array of a CSx representation
 * @param[in] mm total number of edges incident to the vertices in @see s
 * @param[in] mm_parts number of partitions of the edge set to construct
 * @return a freshly allocated array of edge partition descriptors
 ************************************************************************/
template<size_t mm_block=2048, size_t mm_threshold=2048,
	 typename lVID, typename lEID, typename RecordFn>
void partition_vertex_list(
    const lVID * s, // sparse frontier
    lVID m, // number of vertices in sparse frontier
    const lEID * cdegree, // cumulative degrees of vertices in s
    const lEID * idx, // index array CSx
    lEID mm, // total number of edges
    lVID mm_parts,
    const RecordFn & record ) {
    lEID e_done = 0;
    lVID v_done = 0;

    for( lVID p=0; p < mm_parts; ++p ) {
	lEID avgdeg = ( mm - e_done ) / lEID( mm_parts - p );
	const lEID * bnd = std::upper_bound( &cdegree[v_done], &cdegree[m],
					     e_done + avgdeg );
	lVID bnd_pos = bnd - cdegree;
	record( p, v_done, bnd_pos );
	v_done = bnd_pos;
	e_done = *bnd;
    }
    assert( e_done == mm );
    assert( v_done == m );
}

/************************************************************************
 * @brief Calculate a partition of the edge set of the edges incident to the
 * vertices in a list.
 *
 * The function may call record fewer than mm_parts times, i.e., create
 * fewer than mm_parts partitions.
 *
 * @param <mm_block> target number of edges per part
 * @param <mm_threshold> tolerance on the number of edges per part
 * @param <lVID> the vertex ID type
 * @param <lEID> the edge ID type
 * @param <RecordFn> how to record each partition
 *
 * @param[in] s an array of vertices
 * @param[in] m number of vertices in array @see s
 * @param[in] cdegree an array with cumulative sum of degrees of the vertices
 * @param[in] idx index array of a CSx representation
 * @param[in] mm total number of edges incident to the vertices in @see s
 * @param[in] mm_parts number of partitions of the edge set to construct
 * @return a freshly allocated array of edge partition descriptors
 ************************************************************************/
template<size_t mm_block=2048, size_t mm_threshold=2048,
	 typename lVID, typename lEID, typename RecordFn>
lVID partition_edge_list(
    const lVID * s, // sparse frontier
    lVID m, // number of vertices in sparse frontier
    const lEID * cdegree, // cumulative degrees of vertices in s
    const lEID * idx, // index array CSx
    lEID mm, // total number of edges
    lVID mm_parts,
    const RecordFn & record ) {
    lEID e_done = 0;
    lVID v_done = 0;

    lEID next_start = idx[s[0]];
    lVID p = 0;
    for( ; p < mm_parts && v_done < m; ++p ) {
	lEID avgdeg = ( mm - e_done ) / lEID(mm_parts - p);
	const lEID * bnd = std::upper_bound( &cdegree[v_done], &cdegree[m],
					     e_done + avgdeg );
	lEID slice = *bnd - e_done;
	if( slice > avgdeg + mm_threshold ) {
	    // Peel of a set of the edges of a vertex, and redo the vertex
	    // in the next iteration, considering its remaining edges
	    lEID excess = slice - avgdeg;
	    // assert( bnd == cdegree || excess <= ( *bnd - *(bnd-1) ) );
	    lVID repeated = bnd - cdegree - 1;
	    lEID pos = idx[s[repeated]+1] - excess;
	    record( p, v_done, repeated+1, next_start, pos, e_done );
	    next_start = pos;
	    v_done = repeated;
	    e_done = *bnd - excess;
	} else if( slice == 0 ) {
	    // A partition with vertices with zero degree
	    // Skip it, under the assumption that we care to iterate edges.
	    // The loop below ensures the assertion at the end (v_done == m)
	    // succeeds.
	    assert( e_done == mm );
	    while( v_done < m && idx[s[v_done]] == idx[s[v_done]+1] )
		++v_done;
	    break;
	} else {
	    lVID bnd_pos = bnd - cdegree;
	    record( p, v_done, bnd_pos, next_start, idx[s[bnd_pos-1]+1], e_done );
	    v_done = bnd_pos;
	    e_done = *bnd;
	    next_start = v_done < m ? idx[s[v_done]] : idx[s[v_done-1]+1];
	}

/*
	std::cerr << "part " << p << ": is=" << parts[p].get_from()
		  << " ie=" << parts[p].get_to()
		  << " fs=" << parts[p].get_fstart()
		  << '/' << ( parts[p].get_from() < m ? idx[s[parts[p].get_from()]] : -1 )
		  << '/' << ( parts[p].get_from() < m ? idx[s[parts[p].get_from()]+1] : -1 )
		  << " le=" << parts[p].get_lend()
		  << '/' << idx[s[parts[p].get_to()-1]]
		  << '/' << idx[s[parts[p].get_to()-1]+1]
		  << " slice=" << slice
		  << " v_done=" << v_done
		  << " e_done=" << e_done
		  << "\n";
*/
    }
    assert( e_done == mm );
    assert( v_done == m );

    return p;
}


/************************************************************************
 ************************************************************************/
template<typename lVID, typename lEID>
class vertex_buffer {
public:
    using VID = lVID;
    using EID = lEID;
    using vertex_iterator = const VID *;
    using edge_iterator = basic_edge_iterator<VID,EID>;

    vertex_buffer( VID * buf, VID bufsize )
	: m_buf( buf ), m_bufsize( bufsize ), m_fill( 0 ) { }
    vertex_buffer( VID * buf, VID bufsize, VID fill, const EID * const idx )
	: m_buf( buf ), m_bufsize( bufsize ), m_fill( fill ) { }

    vertex_iterator vertex_begin() const { return m_buf; }
    vertex_iterator vertex_end() const { return &m_buf[m_fill]; }
    edge_iterator edge_begin( const EID * idx ) const {
	return edge_iterator( 0, idx, m_buf, m_fill );
    }
    edge_iterator edge_end( const EID * idx ) const {
	return edge_iterator( idx, m_buf, m_fill );
    }

    void push_back( VID value, const EID * const idx ) {
	m_buf[m_fill++] = value;
    }

    void close( const EID * idx ) { }

    VID size() const { return m_fill; }
    bool has_space( VID v, EID e, EID target_edges ) const {
	return m_fill + v <= m_bufsize;
    }
    bool is_empty() const { return m_fill == 0; }
    const VID * get_vertices() const { return m_buf; }

    void set( const edge_partition<VID,EID> & ep,
	      const VID * const v = nullptr ) {
	assert( 0 && "Should not call this" );
    }

private:
    VID * m_buf;
    VID m_bufsize;
    VID m_fill;
};


/************************************************************************
 * @brief A work list describing an edge partition.
 *
 * @param <lVID> the vertex ID type
 * @param <lEID> the edge ID type
 ************************************************************************/
template<typename lVID, typename lEID>
class edge_buffer {
public:
    using VID = lVID;
    using EID = lEID;
    using vertex_iterator = const VID *;
    using edge_iterator = trimmed_edge_iterator<lVID,lEID>;

    edge_buffer( VID * vertices, VID bufsize, VID fill, const EID * idx )
	: m_partition( 0, fill, idx[vertices[0]],
		       fill > 0 ? idx[vertices[fill-1]+1] : 0, 0 ),
	  m_vertices( vertices ), m_bufsize( bufsize ),
	  m_current_edges( 0 ) { }

    edge_buffer( VID * vertices, const edge_partition<VID,EID> & ep )
	: m_partition( ep ),
	  m_vertices( vertices ), m_bufsize( ep.get_to() - ep.get_from() ),
	  m_current_edges( 0 ) { }

    edge_buffer( VID * vertices, VID bufsize )
	: m_partition(), m_vertices( vertices ), m_bufsize( bufsize ),
	  m_current_edges( 0 ) { }

    vertex_iterator vertex_begin() const {
	return &m_vertices[m_partition.get_from()];
    }
    vertex_iterator vertex_end() const {
	return &m_vertices[m_partition.get_to()];
    }
    edge_iterator edge_begin( const EID * m_index ) const {
	return edge_iterator( 0, m_index, &m_vertices[get_from()],
			      get_to() - get_from(), 
			      get_fstart(), get_lend() );
    }
    edge_iterator edge_end( const EID * m_index ) const {
	return edge_iterator( m_index, &m_vertices[get_from()],
			      get_to() - get_from() );
    }

    void push_back( VID v, const EID * const idx ) {
	assert( size() < m_bufsize );
	
	EID deg = idx[v+1] - idx[v];
	if( get_to() == 0 )
	    set_fstart( idx[v] );
	m_vertices[get_to()] = v;
	set_to( get_to()+1 );
	m_current_edges += deg;

	assert( m_current_edges != 0 || deg == 0 );

	assert( ( idx[m_vertices[get_from()]] <= get_fstart()
		  && get_fstart() < idx[m_vertices[get_from()]+1] )
		|| ( get_fstart() == idx[m_vertices[get_from()]]
		     && get_fstart() == idx[m_vertices[get_from()]+1] ) );
    }

    template<bool need_atomic,
	     typename Cache, typename Environment, typename Expr>
    VID process_push( VID * out_edges,
		      bool * zf,
		      const EID * idx,
		      const VID * edges,
		      Cache & c,
		      const Environment & env, 
		      const Expr & expr ) const {
	return m_partition.template process_push<need_atomic>(
	    m_vertices, out_edges, zf, idx, edges, c, env, expr );
    }

    template<bool need_atomic, update_method um,
	     typename Cache, typename Environment, typename Expr>
    void
    process_push_many( const VID * vertices,
		       const EID * idx,
		       const VID * edges,
		       bool *zf,
		       Cache & c, const Environment & env,
		       const Expr & expr ) const {
	return m_partition.template process_push_many<need_atomic,um>(
	    vertices, idx, edges, zf, c, env, expr );
    }

    VID size() const { return get_to() - get_from(); }
    bool has_space( VID v, EID e, EID target_edges ) const {
	if( get_to() - get_from() + v > m_bufsize )
	    return false;
	if( m_current_edges + e >= target_edges )
	    return false;
	return true;
    }
    bool is_empty() const { return m_partition.is_empty(); }
    const VID * get_vertices() const { return m_vertices; }

    void set( const edge_partition<VID,EID> & ep,
	      const VID * const v = nullptr ) {
	VID k = ep.get_to() - ep.get_from();
if( k > m_bufsize && v == nullptr )
std::cerr << "ooops: k=" << k << " m_bufsize=" << m_bufsize << "\n";
	assert( k <= m_bufsize );
	if( v )
	    std::copy( v+ep.get_from(), v+ep.get_to(), m_vertices );
	m_partition.set_to( k );
	m_partition.set_from( 0 );
	m_partition.set_fstart( ep.get_fstart() );
	m_partition.set_lend( ep.get_lend() );
	m_partition.set_offset( 0 );
    }

    VID get_from() const { return m_partition.get_from(); }
    VID get_to() const { return m_partition.get_to(); }
    EID get_fstart() const { return m_partition.get_fstart(); }
    EID get_lend() const { return m_partition.get_lend(); }

    void set_to( VID to ) { m_partition.set_to( to ); }
    void set_fstart( EID fstart ) { m_partition.set_fstart( fstart ); }

    void close( const EID * idx ) {
	assert( ( idx[m_vertices[get_from()]] <= get_fstart()
		  && get_fstart() < idx[m_vertices[get_from()]+1] )
		|| ( get_fstart() == idx[m_vertices[get_from()]]
		     && get_fstart() == idx[m_vertices[get_from()]+1] )
	    );

	if( m_partition.get_lend() == 0 ) {
	    if( get_to() - get_from() == 1 ) {
		m_partition.set_lend( get_fstart() + m_current_edges );
		// assert( get_lend() == idx[m_vertices[get_to()-1]+1] );
	    } else if( get_to() - get_from() > 1 ) {
		m_partition.set_lend( idx[m_vertices[get_to()-1]+1] );
	    }
	}
    }

private:
    edge_partition<VID,EID> m_partition;
    VID * m_vertices;
    VID m_bufsize;
    EID m_current_edges;
};

#endif // GRAPTOR_DSL_EMAP_EDGE_CHUNK_H
