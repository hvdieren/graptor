template<typename ID_>
class bucket {
public:
    using ID = ID_;

    bucket()
	: m_entries( nullptr ),
	  m_capacity( 0 ),
	  m_count( 0 ) { }
    bucket( ID capacity_ )
	: m_entries( new ID[capacity_] ),
	  m_capacity( capacity_ ),
	  m_count( 0 ) { }
    ~bucket() {
	if( m_entries != nullptr )
	    delete[] m_entries;
    }

    const bucket<ID> & operator = ( bucket<ID> && b ) {
	if( m_entries != nullptr )
	    delete[] m_entries;
	m_entries = b.m_entries;
	m_capacity = b.m_capacity;
	m_count = b.m_count;
	b.m_entries = nullptr;
	b.m_capacity = 0;
	b.m_count = 0;
	return *this;
    }

    void insert( ID xpos, ID val ) {
	m_entries[m_count+xpos] = val;
    }
    void insert( ID val ) {
	grow( 1 );
	m_entries[m_count++] = val;
    }

    void grow( ID inc ) {
	if( m_count + inc > m_capacity ) {
	    m_capacity = m_capacity == 0 ? 128 : 2*m_capacity;
	    while( m_count + inc > m_capacity )
		m_capacity *= 2;

	    ID * new_entries = new ID[m_capacity];
	    // TODO: in parallel?
	    std::copy( &m_entries[0], &m_entries[m_count], &new_entries[0] );
	    if( m_entries )
		delete[] m_entries;
	    m_entries = new_entries;
	}
    }
    void resize( ID inc ) {
	m_count += inc;
    }
    void clear() { m_count = 0; }

    frontier as_frontier( ID n ) {
	frontier F = frontier::sparse( n, m_count, m_entries );
	m_entries = nullptr;
	m_count = m_capacity = 0;
	return F;
    }

    bool empty() const { return m_count == 0; }
    ID size() const { return m_count; }
    const ID * get_ptr() const { return m_entries; }

    friend void swap( bucket<ID> & l, bucket<ID> & r ) {
	using namespace std;
	swap( l.m_entries, r.m_entries );
	swap( l.m_capacity, r.m_capacity );
	swap( l.m_count, r.m_count );
    }

private:
    ID * m_entries;
    ID m_capacity;
    ID m_count;
};

template<typename ID_, typename BktFn>
struct bucket_updater {
    using ID = ID_;
    using BucketFn = BktFn;

    bucket_updater( const ID * lst, ID num, BucketFn fn )
	: m_lst( lst ), m_num( num ), m_fn( fn ) { }

    VID size() const { return m_num; }

    std::pair<ID,ID> operator() ( ID nth ) const {
	ID id = m_lst[nth];
	ID bkt = m_fn( id );
	return std::make_pair( id, bkt );
    }
    
private:
    const ID * m_lst;
    ID m_num;
    BucketFn m_fn;
};

// This assumes that the number of active vertices and edges may not have
// been calculated.
template<frontier_type ftype, typename ID_, typename BktFn>
struct bucket_updater_dense {
    using ID = ID_;
    using BucketFn = BktFn;
    using L = typename frontier_params<ftype,0>::type;

    bucket_updater_dense( const frontier & f, BucketFn fn )
	: m_mask( f.template getDense<ftype>() ),
	  m_num( f.nVertices() ), m_fn( fn ) { }

    VID size() const { return m_num; }

    std::pair<ID,ID> operator() ( ID nth ) const {
	if( !m_mask[nth] )
	    return std::make_pair( ~(VID)0, ~(VID)0 );
	else
	    return std::make_pair( nth, m_fn( nth ) );
    }
    
private:
    const L * m_mask;
    ID m_num;
    BucketFn m_fn;
};

template<typename ID_, typename BktFn_>
class buckets {
public:
    using ID = ID_;
    using BucketFn = BktFn_;

    /* buckets
     * @param n_: the number of identifiers
     * @param bfn_: function mapping identifier to bucket
     * @param open_buckets_: the number of open buckets to keep
     */
    buckets( ID n,
	     ID open_buckets,
	     BucketFn fn )
	: m_range( n ), m_open_buckets( open_buckets ), m_fn( fn ),
	  m_cur_bkt( 0 ), m_cur_range( 0 ), m_elems( 0 ) {
	// Create a set of buckets
	m_buckets = new bucket<ID>[m_open_buckets+1];
    }

    ~buckets() {
	delete[] m_buckets;
    }

    bool empty() const { return m_elems == 0; }

    ID get_current_bucket() const { return m_cur_range + m_cur_bkt + 1; }

    frontier __attribute__((noinline)) next_bucket() {
	while( true ) {
	    // Re-do current bucket if not empty (some elements changed, and
	    // remained in current bucket)
	    if( !m_buckets[m_cur_bkt].empty() ) {
		ID b = m_cur_bkt;
		m_elems -= m_buckets[b].size();
		// std::cerr << "BUCKETS: take bucket " << b
			  // << " with " << m_buckets[b].size() << " items "
			  // << " count now " << m_elems << "\n";
		return m_buckets[b].as_frontier( m_range );
	    }

	    // Progress to next bucket
	    ++m_cur_bkt;

	    // If we run out of buckets, refill
	    if( m_cur_bkt == m_open_buckets ) {
		using namespace std;
		// Assumption: all buckets are empty except for the final one
		bucket<ID> reassign;
		swap( reassign, m_buckets[m_open_buckets] );
		assert( m_elems == reassign.size() );
		m_elems = 0;
		m_cur_range += m_open_buckets;
		m_cur_bkt = 0;
		update_buckets( reassign );
	    }
	}
    }

    void insert( ID id, ID b ) {
	update_buckets_seq(
	    [&]( auto i ) { return std::make_pair( id, b ); }, 1 );
    }

    void update_buckets( const ID * elements, ID num_elements ) {
	bucket_updater upd( elements, num_elements, m_fn );
	return update_buckets_sparse( upd );
    }

    void update_buckets( const partitioner & part, frontier & f ) {
	switch( f.getType() ) {
	case frontier_type::ft_true:
	case frontier_type::ft_bit:
	case frontier_type::ft_bit2:
	case frontier_type::ft_bool:
	case frontier_type::ft_logical1:
	case frontier_type::ft_logical2:
	case frontier_type::ft_logical8:
	    assert( 0 && "NYI" );
	case frontier_type::ft_logical4:
	{
	    bucket_updater_dense<frontier_type::ft_logical4, ID, BucketFn>
		upd( f, m_fn );
	    update_buckets_dense( part, upd, upd.size() );
	    break;
	}
	case frontier_type::ft_sparse:
	{
	    bucket_updater upd( f.getSparse(), f.nActiveVertices(), m_fn );
	    update_buckets_sparse( upd );
	    break;
	}
	default: UNREACHABLE_CASE_STATEMENT;
	}
    }
    
private:
    ID slot( ID bkt ) const {
	if( bkt < m_cur_range + m_cur_bkt )
	    return m_cur_bkt;
	else if( bkt >= m_cur_range + m_open_buckets )
	    return m_open_buckets;
	else
	    return bkt - m_cur_range;
    }
    
    void update_buckets( const bucket<ID> & b ) {
	bucket_updater upd( b.get_ptr(), b.size(), m_fn );
	update_buckets_sparse( upd );
    }
    
    template<typename IterFn>
    void update_buckets_sparse( IterFn fn ) {
	update_buckets_sparse( fn, fn.size() );
    }

    template<typename IterFn>
    void update_buckets_dense( const partitioner & part,
			       IterFn fn, ID num_elements ) {
	static constexpr ID BLOCK = 64;

	// 0. Allocate memory
	unsigned np = part.get_num_partitions();
	ID hsize = BLOCK;
	while( hsize < m_open_buckets+1 )
	    hsize *= 2;
	ID * hist = new ID[(np+1) * hsize];

	std::fill( &hist[0], &hist[(np+1)*hsize], ID(0) );

	// 1. Calculate number of elements moving to each bucket
	//    There are m_open_buckets+1 buckets (final one is overflow)
	map_partition( part, [&]( unsigned p ) {
	    VID s = part.start_of( p );
	    VID e = part.end_of( p );
	    ID * lhist = &hist[p*hsize];
	    for( VID v=s; v < e; ++v ) {
		ID id, bkt;
		std::tie( id, bkt ) = fn( v );

		if( id == ~(ID)0 )
		    continue;

		ID b = slot( bkt );
		lhist[b]++;
	    }
	} );
	
	// 2. Aggregate histograms and compute insertion points for
	//    each partition / bucket
	ID * thist = &hist[np * hsize];
	parallel_for( ID b=0; b < m_open_buckets+1; ++b ) {
	    ID t = 0;
	    for( unsigned p=0; p < np; ++p ) {
		ID u = hist[p*hsize+b];
		hist[p*hsize+b] = t;
		t += u;
	    }
	    thist[b] = t;
	}

	// 3. Resize buckets to accommodate new elements
	for( ID b=0; b < m_open_buckets+1; ++b )
	    m_buckets[b].grow( thist[b] );

	// 4. Insert elements into buckets
	map_partition( part, [&]( unsigned p ) {
	    VID s = part.start_of( p );
	    VID e = part.end_of( p );
	    ID * lhist = &hist[p*hsize];
	    for( VID v=s; v < e; ++v ) {
		ID id, bkt;
		std::tie( id, bkt ) = fn( v );

		if( id == ~(ID)0 )
		    continue;

		ID b = slot( bkt );
		m_buckets[b].insert( lhist[b]++, id );
	    }
	} );

	// 5. Update bucket sizes
	for( ID b=0; b < m_open_buckets+1; ++b ) {
	    m_buckets[b].resize( thist[b] );
	    m_elems += thist[b];
	}

	// 6. Sanity check (debugging)
	ID k = 0;
	for( ID i=0; i < m_open_buckets+1; ++i )
	    k += m_buckets[i].size();
	assert( k == m_elems );

	// Cleanup
	delete[] hist;
    }

    template<typename IterFn>
    void update_buckets_sparse( IterFn fn, ID num_elements ) {
	static constexpr ID CHUNK = 4096;
	static constexpr ID BLOCK = 64;

	ID np = ( num_elements + CHUNK - 1 ) / CHUNK;
	if( np < 2 ) {
	    update_buckets_seq( fn, num_elements );
	    return;
	}

	// 0. Allocate memory
	ID hsize = BLOCK;
	while( hsize < m_open_buckets+1 )
	    hsize *= 2;
	ID * hist = new ID[(np+1) * hsize];

	std::fill( &hist[0], &hist[(np+1)*hsize], ID(0) );

	// 1. Calculate number of elements moving to each bucket
	//    There are m_open_buckets+1 buckets (final one is overflow)
	parallel_for( ID p=0; p < np; ++p ) {
	    VID s = p * CHUNK;
	    VID e = std::min( (p+1)*CHUNK, num_elements );
	    ID * lhist = &hist[p*hsize];
	    for( VID v=s; v < e; ++v ) {
		ID id, bkt;
		std::tie( id, bkt ) = fn( v );

		if( id == ~(ID)0 )
		    continue;

		ID b = slot( bkt );
		lhist[b]++;
	    }
	};
	
	// 2. Aggregate histograms and compute insertion points for
	//    each chunk / bucket
	ID * thist = &hist[np * hsize];
	parallel_for( ID b=0; b < m_open_buckets+1; ++b ) {
	    ID t = 0;
	    for( unsigned p=0; p < np; ++p ) {
		ID u = hist[p*hsize+b];
		hist[p*hsize+b] = t;
		t += u;
	    }
	    thist[b] = t;
	}

	// 3. Resize buckets to accommodate new elements
	for( ID b=0; b < m_open_buckets+1; ++b )
	    m_buckets[b].grow( thist[b] );

	// 4. Insert elements into buckets
	parallel_for( ID p=0; p < np; ++p ) {
	    VID s = p * CHUNK;
	    VID e = std::min( (p+1)*CHUNK, num_elements );
	    ID * lhist = &hist[p*hsize];
	    for( VID v=s; v < e; ++v ) {
		ID id, bkt;
		std::tie( id, bkt ) = fn( v );

		if( id == ~(ID)0 )
		    continue;

		ID b = slot( bkt );
		m_buckets[b].insert( lhist[b]++, id );
	    }
	};

	// 5. Update bucket sizes
	for( ID b=0; b < m_open_buckets+1; ++b ) {
	    m_buckets[b].resize( thist[b] );
	    m_elems += thist[b];
	}

	// 6. Sanity check (debugging)
	ID k = 0;
	for( ID i=0; i < m_open_buckets+1; ++i )
	    k += m_buckets[i].size();
	assert( k == m_elems );

	// Cleanup
	delete[] hist;
    }
	
    template<typename IterFn>
    void update_buckets_seq( IterFn fn, ID num_elements ) {
	ID num_inserted = 0;
	for( ID i=0; i < num_elements; ++i ) {
	    ID id, bkt;
	    std::tie( id, bkt ) = fn( i );

	    if( id == ~(ID)0 )
		continue;

	    ID b = slot( bkt );
	    m_buckets[b].insert( id );
	    ++num_inserted;

	    // std::cerr << "BUCKETS: insert " << id << " into " << b
	    // << " contains now " << m_buckets[b].size() << "\n";
	}
	m_elems += num_inserted;
	// std::cerr << "BUCKETS: insert " << k
		  // << " elements, count now " << m_elems << "\n";

	ID k = 0;
	for( ID i=0; i <= m_open_buckets; ++i )
	    k += m_buckets[i].size();
	assert( k == m_elems );
    }

private:
    ID m_range;
    ID m_open_buckets;
    BucketFn m_fn;
    ID m_cur_bkt;
    ID m_cur_range;
    ID m_elems;
    bucket<ID> * m_buckets;
};
