#include <limits>

/************************************************************************
 * Representation of a single bucket.
 * Counts of objects are size_t due to potential replication of objects.
 *
 * @param <ID_> the type of the objects stored in the bucket
 ************************************************************************/
template<typename ID_>
class bucket {
public:
    using ID = ID_;

    bucket()
	: m_entries( nullptr ),
	  m_capacity( 0 ),
	  m_count( 0 ) { }
    bucket( size_t capacity_ )
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

    void insert( size_t xpos, ID val ) {
	m_entries[m_count+xpos] = val;
    }
    void insert( ID val ) {
	grow( 1 );
	m_entries[m_count++] = val;
    }

    // Note: requires &b != this
    void take( bucket<ID> & b ) {
	grow( b.m_count );
	std::copy( &b.m_entries[0], &b.m_entries[b.m_count],
		   &m_entries[m_count] );
	m_count += b.m_count;
	b.m_count = 0;
    }

    size_t filter( size_t inc ) {
	size_t l = 0;
	for( size_t k=0; k < inc; ++k ) {
	    if( m_entries[m_count+k] != std::numeric_limits<ID>::max() ) {
		m_entries[m_count+l] = m_entries[m_count+k];
		++l;
	    }
	}
	return l;
    }

    void grow( size_t inc ) {
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
    void resize( size_t inc ) {
	m_count += inc;
    }
    void clear() { m_count = 0; }

    //! Convert bucket to frontier and transfer ownership over contents.
    frontier as_frontier( size_t n ) {
	// By default, don't worry about how many edges there are on F since we
	// likely use it only in a vertex-filter operation. Simply record
	// non-zero.
	frontier F = frontier::sparse( n, m_count, m_entries );
	m_entries = nullptr;
	m_count = m_capacity = 0;
	F.setActiveCounts( F.nActiveVertices(), F.nActiveVertices() );
	return F;
    }

    bool empty() const { return m_count == 0; }
    size_t size() const { return m_count; }
    const ID * get_ptr() const { return m_entries; }

    friend void swap( bucket<ID> & l, bucket<ID> & r ) {
	using namespace std;
	swap( l.m_entries, r.m_entries );
	swap( l.m_capacity, r.m_capacity );
	swap( l.m_count, r.m_count );
    }

private:
    ID * m_entries;
    size_t m_capacity;
    size_t m_count;
};

/************************************************************************
 * Functor to help assigning objects to buckets.
 *
 * @param <ID_> the type of the objects stored in the bucket
 * @param <BID_> the identifier type for bucket numbers
 ************************************************************************/
template<typename ID_, typename BID_, typename BktFn>
struct bucket_updater {
    using ID = ID_;
    using BID = BID_;
    using BucketFn = BktFn;

    bucket_updater( const ID * lst, size_t num, BID cur, BID oflow,
		    BucketFn fn )
	: m_lst( lst ), m_num( num ), m_cur( cur ), m_oflow( oflow ),
	  m_fn( fn ) { }

    VID size() const { return m_num; }

    std::pair<ID,BID> operator() ( size_t nth ) const {
	ID id = m_lst[nth];
	BID bkt = m_fn( id, m_cur, m_oflow );
	return std::make_pair( id, bkt );
    }

    ID get( size_t nth ) const {
	return m_lst[nth];
    }
    
private:
    const ID * m_lst;
    size_t m_num;
    BID m_cur;
    BID m_oflow;
    BucketFn m_fn;
};

template<typename ID_, typename BID_, typename BktFn>
struct bucket_updater_all {
    using ID = ID_;
    using BID = BID_;
    using BucketFn = BktFn;

    bucket_updater_all( size_t num, BID cur, BID oflow, BucketFn fn )
	: m_num( num ), m_cur( cur ), m_oflow( oflow ), m_fn( fn ) { }

    size_t size() const { return m_num; }

    std::pair<ID,BID> operator() ( size_t nth ) const {
	return std::make_pair( ID(nth), m_fn( ID(nth), m_cur, m_oflow ) );
    }

    ID get( size_t nth ) const { return ID(nth); }
    
private:
    size_t m_num;
    BID m_cur;
    BID m_oflow;
    BucketFn m_fn;
};

// This assumes that the number of active vertices and edges may not have
// been calculated.
template<frontier_type ftype, typename ID_, typename BID_, typename BktFn>
struct bucket_updater_dense {
    using ID = ID_;
    using BID = BID_;
    using BucketFn = BktFn;
    using L = typename frontier_params<ftype,0>::type;

    bucket_updater_dense( const frontier & f, BID cur, BID oflow, BucketFn fn )
	: m_mask( f.template getDense<ftype>() ),
	  m_num( f.nVertices() ), m_cur( cur ), m_oflow( oflow ), m_fn( fn ) { }

    VID size() const { return m_num; }

    std::pair<ID,BID> operator() ( size_t nth ) const {
	if( !m_mask[nth] )
	    return std::make_pair( std::numeric_limits<ID>::max(),
				   std::numeric_limits<BID>::max() );
	else
	    return std::make_pair( nth, m_fn( nth, m_cur, m_oflow ) );
    }
    
/*
    ID get( ID nth ) const {
	if( !m_mask[nth] )
	    return ~(VID)0;
	else
	    return nth;
    }
*/
    
private:
    const L * m_mask;
    size_t m_num;
    BID m_cur;
    BID m_oflow;
    BucketFn m_fn;
};

template<typename ID_, typename BID_, typename BktFn>
struct bucket_updater_dense<frontier_type::ft_bit, ID_, BID_, BktFn> {
    using ID = ID_;
    using BID = BID_;
    using BucketFn = BktFn;
    using L = typename frontier_params<frontier_type::ft_bit,0>::type;

    bucket_updater_dense( const frontier & f, BID cur, BID oflow, BucketFn fn )
	: m_mask( f.template getDense<frontier_type::ft_bit>() ),
	  m_num( f.nVertices() ), m_cur( cur ), m_oflow( oflow ), m_fn( fn ) { }

    VID size() const { return m_num; }

    std::pair<ID,BID> operator() ( size_t nth ) const {
	constexpr L mod = 8 * sizeof(L);
	constexpr L mask = mod - 1;
	if( ( ( m_mask[nth / mod] >> ( nth & mask ) ) & 1 ) == 0 )
	    return std::make_pair( std::numeric_limits<ID>::max(),
				   std::numeric_limits<BID>::max() );
	else
	    return std::make_pair( nth, m_fn( nth, m_cur, m_oflow ) );
    }
    
private:
    const L * m_mask;
    size_t m_num;
    BID m_cur;
    BID m_oflow;
    BucketFn m_fn;
};


/*
template<typename BktFn>
struct half_bucket_fn {
    using ID = typename BktFn::ID;
    
    half_bucket_fn( BktFn fn ) : m_fn( fn ) { }

    ID operator() ( ID v ) const {
	return m_fn.template get_scaled<2>( v );
    }
    
private:
    BktFn m_fn;
};
*/

template<typename ID_, typename BktFn_>
class buckets {
public:
    using ID = ID_;
    using BID = std::make_unsigned_t<ID>;
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
	m_buckets = new bucket<ID>[m_open_buckets+1]();
    }

    ~buckets() {
	delete[] m_buckets;
    }

    template<typename ID__, typename BktFn__>
    friend class buckets;

    bool empty() const { return m_elems == 0; }

    BID get_current_bucket() const { return m_cur_range + m_cur_bkt; }
    BID get_overflow_bucket() const { return m_cur_range + m_open_buckets; }

/*
    need to consider reinsertion cost -> especially if in dense traversal
    we visit all vertices, and may count all as updated, even if they
	do not move bucket (we will re-insert, doubly -> toDense should catch duplicates in sparse frontier or not?)
	Seems like number of removable vertices is really small
*/

    frontier __attribute__((noinline)) next_bucket() {
	while( true ) {
	    // If we run out of buckets, refill
	    if( m_cur_bkt == m_open_buckets ) {
		using namespace std;
		// Assumption: all buckets are empty except for the final one
		bucket<ID> reassign;

		swap( reassign, m_buckets[m_open_buckets] );
		m_cur_range += m_open_buckets;
		m_cur_bkt = 0; // actual bucket is m_cur_range + m_cur_bkt
		
		assert( m_elems == reassign.size() );
		m_elems = 0;
		reassign_buckets( reassign );

		// TODO: at this pointm we may desire to remove duplicates
		//       to reduce numbers.
		// TODO: initially, only need to insert degree-1 vertices,
		//       others will be added as we go along (wavefront).
		// std::cerr << "split open bucket; current=" << m_cur_bkt
		//   << " range=" << m_cur_range
		//   << " m_elems=" << m_elems << "\n";

		// In case we dropped all elements during re-assignment
		if( m_elems == 0 )
		    return frontier::empty();
	    }

	    // Re-do current bucket if not empty (some elements changed, and
	    // remained in current bucket)
	    if( !m_buckets[m_cur_bkt].empty() ) {
/* merge buckets
		for( ID b=m_cur_bkt+1; b < m_open_buckets; ++b ) {
		    if( m_buckets[m_cur_bkt].size()
			+ m_buckets[b].size() > m_range/8 )
			break;

		    m_buckets[m_cur_bkt].take( m_buckets[b] );
		}
		m_elems -= m_buckets[m_cur_bkt].size();
		return m_buckets[m_cur_bkt].as_frontier( m_range );
*/
/* split buckets
		if( m_buckets[m_cur_bkt].size() > (1<<18) ) {
		    using namespace std;
		    using HFn = half_bucket_fn<BucketFn>;

		    HFn hfn( m_fn );
		    buckets<ID,HFn> bkts2( m_range, 2, hfn );
		    bkts2.m_cur_range = ( m_cur_range + m_cur_bkt ) * 2;
		    bkts2.m_buckets[0].grow( m_buckets[m_cur_bkt].size() );
		    bkts2.m_buckets[1].grow( m_buckets[m_cur_bkt].size() );
		    bkts2.update_buckets( m_buckets[m_cur_bkt] );

		    assert( bkts2.m_buckets[0].size()
			    + bkts2.m_buckets[1].size()
			    == m_buckets[m_cur_bkt].size() );
		    
		    frontier F = bkts2.m_buckets[0].as_frontier( m_range );
		    swap( m_buckets[m_cur_bkt], bkts2.m_buckets[1] );
		    if( F.nActiveVertices() == 0 )
			F.del();
		    else {
			std::cerr << "split " << bkts2.m_elems
				  << " into " << F.nActiveVertices() << "\n";
			m_elems -= F.nActiveVertices();
			return F;
		    } 
		}
 */

/*
*/
		m_elems -= m_buckets[m_cur_bkt].size();
		return m_buckets[m_cur_bkt].as_frontier( m_range );
	    }

	    // Progress to next bucket
	    ++m_cur_bkt;
	}
    }

    void insert( ID id, BID b ) {
	update_buckets_seq(
	    [&]( auto i ) { return std::make_pair( id, b ); }, 1 );
    }

    void update_buckets( const ID * elements, size_t num_elements ) {
	bucket_updater upd( elements, num_elements,
			    get_current_bucket(),
			    get_overflow_bucket(), m_fn );
	return update_buckets_sparse( upd );
    }

    void initialize_buckets( const partitioner & part, frontier & f ) {
	update_buckets<true>( part, f );
    }

    template<bool initialize=false>
    void update_buckets( const partitioner & part, frontier & f ) {
	BID cur = get_current_bucket();
	BID oflow = initialize
	    ? std::numeric_limits<ID>::max() : get_overflow_bucket();
	switch( f.getType() ) {
	case frontier_type::ft_true:
	{
	    bucket_updater_all<ID, BID, BucketFn>
		upd( m_range, cur, oflow, m_fn );
	    update_buckets_dense( part, upd, upd.size() );
	    break;
	}
	case frontier_type::ft_bit2:
	case frontier_type::ft_logical1:
	case frontier_type::ft_logical2:
	case frontier_type::ft_logical8:
	{
	    assert( 0 && "NYI" );
	    break;
	}
	case frontier_type::ft_bit:
	{
	    bucket_updater_dense<frontier_type::ft_bit, ID, BID, BucketFn>
		upd( f, cur, oflow, m_fn );
	    update_buckets_dense( part, upd, upd.size() );
	    break;
	}
	case frontier_type::ft_bool:
	{
	    bucket_updater_dense<frontier_type::ft_bool, ID, BID, BucketFn>
		upd( f, cur, oflow, m_fn );
	    update_buckets_dense( part, upd, upd.size() );
	    break;
	}
	case frontier_type::ft_logical4:
	{
	    bucket_updater_dense<frontier_type::ft_logical4, ID, BID, BucketFn>
		upd( f, cur, oflow, m_fn );
	    update_buckets_dense( part, upd, upd.size() );
	    break;
	}
	case frontier_type::ft_sparse:
	{
	    bucket_updater
		upd( f.getSparse(), f.nActiveVertices(), cur, oflow, m_fn );
	    update_buckets_sparse( upd );
	    break;
	}
	default: UNREACHABLE_CASE_STATEMENT;
	}
	// std::cerr << "m_cur_bkt=" << m_cur_bkt << "\n";
	// std::cerr << "m_elems=" << m_elems << "\n";
    }
    
private:
    BID slot( BID bkt ) const {
	if( bkt < m_cur_range + m_cur_bkt )
	    return m_cur_bkt;
	else if( bkt >= m_cur_range + m_open_buckets )
	    return m_open_buckets;
	else
	    return bkt - m_cur_range;
    }
    
    void reassign_buckets( const bucket<ID> & b ) {
	// This is a special version where elements have to be re-inserted
	// in the overflow bucket, which is otherwise avoided.
	bucket_updater upd( b.get_ptr(), b.size(),
			    get_current_bucket(),
			    std::numeric_limits<BID>::max(), m_fn );
	update_buckets_sparse( upd );
    }
    
    template<typename IterFn>
    void update_buckets_sparse( IterFn fn ) {
	update_buckets_sparse( fn, fn.size() );
    }

    template<typename IterFn>
    void update_buckets_dense( const partitioner & part,
			       IterFn fn, size_t num_elements ) {
	static constexpr ID BLOCK = 64;

	if( num_elements == 0 )
	    return;

	// 0. Allocate memory
	size_t np = part.get_num_partitions();
	size_t hsize = ( BLOCK + sizeof(ID) - 1 ) / sizeof(ID);
	while( hsize < m_open_buckets+1 )
	    hsize *= 2;
	size_t * hist = new size_t[(np+1) * hsize](); // zero init
	uint8_t * idb = new uint8_t[num_elements];

	assert( sizeof(*idb)*256 >= m_open_buckets+1 );
	static_assert( std::is_same_v<ID,VID>, "implicit assumption" );

	// 1. Calculate number of elements moving to each bucket
	//    There are m_open_buckets+1 buckets (final one is overflow)
	map_partition( part, [&]( unsigned p ) {
	    VID s = part.start_of( p );
	    VID e = part.end_of( p );
	    size_t * lhist = &hist[p*hsize];
	    for( VID v=s; v < e; ++v ) {
		ID id;
		BID bkt;
		std::tie( id, bkt ) = fn( v );

		if( id == std::numeric_limits<ID>::max() )
		    continue;

		if( bkt == std::numeric_limits<BID>::max() )
		    continue;

		BID b = slot( bkt );
		lhist[b]++;
		idb[v] = b;
	    }
	} );
	
	// 2. Aggregate histograms and compute insertion points for
	//    each partition / bucket
	size_t * thist = &hist[np * hsize];
	parallel_for( BID b=0; b < m_open_buckets+1; ++b ) {
	    size_t t = 0;
	    for( unsigned p=0; p < np; ++p ) {
		size_t u = hist[p*hsize+b];
		hist[p*hsize+b] = t;
		t += u;
	    }
	    thist[b] = t;
	}

	// 3. Resize buckets to accommodate new elements
	for( BID b=0; b < m_open_buckets+1; ++b )
	    m_buckets[b].grow( thist[b] );

	// 4. Insert elements into buckets
	map_partition( part, [&]( unsigned p ) {
	    VID s = part.start_of( p );
	    VID e = part.end_of( p );
	    size_t * lhist = &hist[p*hsize];
	    for( VID v=s; v < e; ++v ) {
		ID id;
		BID bkt;
		std::tie( id, bkt ) = fn( v );
		// ID id = fn.get( v );

		if( id == std::numeric_limits<ID>::max() )
		    continue;

		if( bkt == std::numeric_limits<BID>::max() )
		    continue;

		// ID b = slot( bkt );
		BID b = idb[v];
		m_buckets[b].insert( lhist[b]++, id );
/*
		if( m_fn.set_slot( id, b ) )
		    m_buckets[b].insert( lhist[b]++, id );
		else
		    __sync_fetch_and_add( &thist[b], -1 );
*/
	    }
	} );

	// 5. Update bucket sizes
	for( BID b=0; b < m_open_buckets+1; ++b ) {
	    m_buckets[b].resize( thist[b] );
	    m_elems += thist[b];
	}

	// 6. Sanity check (debugging)
	size_t k = 0;
	for( BID i=0; i < m_open_buckets+1; ++i )
	    k += m_buckets[i].size();
	assert( k == m_elems );

	// Cleanup
	delete[] idb;
	delete[] hist;
    }

    template<typename IterFn>
    void update_buckets_sparse( IterFn fn, size_t num_elements ) {
	static constexpr size_t CHUNK = 4096;
	static constexpr size_t BLOCK = 64;

	if( num_elements == 0 )
	    return;

	size_t np = ( num_elements + CHUNK - 1 ) / CHUNK;
	if( np < 2 ) {
	    update_buckets_seq( fn, num_elements );
	    return;
	}

	// 0. Allocate memory
	size_t hsize = BLOCK;
	while( hsize < m_open_buckets+1 )
	    hsize *= 2;
	size_t * hist = new size_t[(np+1) * hsize]();
	uint8_t * idb = new uint8_t[num_elements];

	assert( sizeof(*idb)*256 >= m_open_buckets+1 );

	// 1. Calculate number of elements moving to each bucket
	//    There are m_open_buckets+1 buckets (final one is overflow)
	parallel_for( size_t p=0; p < np; ++p ) {
	    size_t s = p * CHUNK;
	    size_t e = std::min( (p+1)*CHUNK, num_elements );
	    size_t * lhist = &hist[p*hsize];
	    for( size_t v=s; v < e; ++v ) {
		ID id;
		BID bkt;
		std::tie( id, bkt ) = fn( v );

		if( id == std::numeric_limits<ID>::max() )
		    continue;

		if( bkt == std::numeric_limits<BID>::max() )
		    continue;

		BID b = slot( bkt );
		lhist[b]++;
		idb[v] = b;
	    }
	};
	
	// 2. Aggregate histograms and compute insertion points for
	//    each chunk / bucket
	size_t * thist = &hist[np * hsize];
	parallel_for( BID b=0; b < m_open_buckets+1; ++b ) {
	    size_t t = 0;
	    for( size_t p=0; p < np; ++p ) {
		size_t u = hist[p*hsize+b];
		hist[p*hsize+b] = t;
		t += u;
	    }
	    thist[b] = t;
	}

	// 3. Resize buckets to accommodate new elements
	for( BID b=0; b < m_open_buckets+1; ++b )
	    m_buckets[b].grow( thist[b] );

	// 4. Insert elements into buckets
	parallel_for( size_t p=0; p < np; ++p ) {
	    VID s = p * CHUNK;
	    VID e = std::min( (p+1)*CHUNK, num_elements );
	    size_t * lhist = &hist[p*hsize];
	    for( VID v=s; v < e; ++v ) {
		ID id;
		BID bkt;
		std::tie( id, bkt ) = fn( v );
		// ID id = fn.get( v );

		if( id == std::numeric_limits<ID>::max() )
		    continue;

		if( bkt == std::numeric_limits<BID>::max() )
		    continue;

		// ID b = slot( bkt );
		BID b = idb[v];
		m_buckets[b].insert( lhist[b]++, id );
/*
		if( m_fn.set_slot( id, b ) )
		    m_buckets[b].insert( lhist[b]++, id );
		else
		    m_buckets[b].insert( lhist[b]++, ~(ID)0 );
*/
	    }
	};

	// 4b. filter empty slots
/*
	parallel_for( ID b=0; b < m_open_buckets+1; ++b ) {
	    thist[b] = m_buckets[b].filter( thist[b] );
	}
*/

	// 5. Update bucket sizes
	for( BID b=0; b < m_open_buckets+1; ++b ) {
	    m_buckets[b].resize( thist[b] );
	    m_elems += thist[b];
	}

	// 6. Sanity check (debugging)
	size_t k = 0;
	for( BID i=0; i < m_open_buckets+1; ++i )
	    k += m_buckets[i].size();
	assert( k == m_elems );

	// Cleanup
	delete[] idb;
	delete[] hist;
    }
	
    template<typename IterFn>
    void update_buckets_seq( IterFn fn, size_t num_elements ) {
	size_t num_inserted = 0;
	for( size_t i=0; i < num_elements; ++i ) {
	    ID id;
	    BID bkt;
	    std::tie( id, bkt ) = fn( i );

	    if( id == std::numeric_limits<ID>::max() )
		continue;

	    if( bkt == std::numeric_limits<BID>::max() )
		continue;

	    BID b = slot( bkt );
	    m_buckets[b].insert( id );
	    ++num_inserted;

/*
	    if( m_fn.set_slot( id, b ) ) {
		m_buckets[b].insert( id );
		++num_inserted;
	    }
*/
	}
	m_elems += num_inserted;

	// Sanity check (debugging)
	size_t k = 0;
	for( BID i=0; i <= m_open_buckets; ++i )
	    k += m_buckets[i].size();
	assert( k == m_elems );
    }

private:
    BID m_range;
    BID m_open_buckets;
    BucketFn m_fn;
    BID m_cur_bkt;
    BID m_cur_range;
    size_t m_elems;
    bucket<ID> * m_buckets;
};
