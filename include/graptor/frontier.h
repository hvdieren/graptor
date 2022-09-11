// -*- c++ -*-
#ifndef GRAPTOR_FRONTIER_H
#define GRAPTOR_FRONTIER_H

#include "graptor/partitioner.h"
#include "simd/simd.h"

// Pre-declarations
template<typename vertex>
class graph;

class GraphCSx;

enum class frontier_type {
    ft_true = 0,
    ft_unbacked = 1,
    ft_bool = 2,
    ft_bit = 3,
    ft_logical1 = 4,
    ft_logical2 = 5,
    ft_logical4 = 6,
    ft_logical8 = 7,
    ft_sparse = 8,
    ft_bit2 = 9,
    ft_msb4 = 10,
    ft_N = 11
};

extern const char * frontier_type_names[static_cast<std::underlying_type_t<frontier_type>>( frontier_type::ft_N )+1];

inline std::ostream & operator << ( std::ostream & os, frontier_type fr ) {
    int ifr = (int) fr;
    if( ifr >= 0 && ifr < (int)frontier_type::ft_N )
	return os << frontier_type_names[ifr];
    else
	return os << frontier_type_names[(int)frontier_type::ft_N];
}

// Default information
template<frontier_type ftype_, unsigned short VL, typename Enable = void>
struct frontier_params; /* {
    static constexpr frontier_type ftype = ftype_;
    using type = void;
    }; */

template<unsigned short VL_>
struct frontier_params<frontier_type::ft_bool,VL_> {
    static constexpr unsigned short W = 1;
    static constexpr unsigned short VL = VL_;
    static constexpr frontier_type ftype = frontier_type::ft_bool;
    using type = bool;
    using data_type = simd::ty<bool,VL>;
    using mask_type = simd::detail::mask_bool_traits;
    using encoding = array_encoding<type>;
};

template<>
struct frontier_params<frontier_type::ft_bit,0> {
    static constexpr unsigned short W = 0;
    static constexpr unsigned short VL = 0;
    static constexpr frontier_type ftype = frontier_type::ft_bit;
    using type = unsigned char;
    using data_type = simd::ty<void,VL>;
    using mask_type = simd::detail::mask_bit_traits<VL>;
    using encoding = array_encoding<type>;
};

template<unsigned short VL_>
struct frontier_params<frontier_type::ft_bit2,VL_> {
    static constexpr unsigned short W = 0;
    static constexpr unsigned short VL = VL_;
    static constexpr frontier_type ftype = frontier_type::ft_bit2;
    using type = unsigned char;
    using data_type = simd::ty<bitfield<2>,VL>;
    using mask_type = simd::detail::mask_bit_logical_traits<2,VL>;
    using encoding = array_encoding_bit<1>;
};

template<unsigned short VL_>
struct frontier_params<frontier_type::ft_bit,VL_,std::enable_if_t<VL_ != 0>> {
    static constexpr unsigned short W = 0;
    static constexpr unsigned short VL = VL_;
    static constexpr frontier_type ftype = frontier_type::ft_bit;
    using type = typename mask_type_traits<1>::type;
    using data_type = simd::ty<void,VL>;
    using mask_type = simd::detail::mask_bit_traits<VL>;
    using encoding = array_encoding_bit<2>;
};

template<unsigned short VL_>
struct frontier_params<frontier_type::ft_logical1,VL_> {
    static constexpr unsigned short W = 1;
    static constexpr unsigned short VL = VL_;
    static constexpr frontier_type ftype = frontier_type::ft_logical1;
    using type = logical<1>;
    using data_type = simd::detail::mask_logical_traits<W,VL>;
    using mask_type = data_type;
    using encoding = array_encoding<type>;
};

template<unsigned short VL_>
struct frontier_params<frontier_type::ft_logical2,VL_> {
    static constexpr unsigned short W = 2;
    static constexpr unsigned short VL = VL_;
    static constexpr frontier_type ftype = frontier_type::ft_logical2;
    using type = logical<2>;
    using data_type = simd::detail::mask_logical_traits<W,VL>;
    using mask_type = data_type;
    using encoding = array_encoding<type>;
};

template<unsigned short VL_>
struct frontier_params<frontier_type::ft_logical4,VL_> {
    static constexpr unsigned short W = 4;
    static constexpr unsigned short VL = VL_;
    static constexpr frontier_type ftype = frontier_type::ft_logical4;
    using type = logical<4>;
    using data_type = simd::detail::mask_logical_traits<W,VL>;
    using mask_type = data_type;
    using encoding = array_encoding<type>;
};

template<unsigned short VL_>
struct frontier_params<frontier_type::ft_logical8,VL_> {
    static constexpr unsigned short W = 8;
    static constexpr unsigned short VL = VL_;
    static constexpr frontier_type ftype = frontier_type::ft_logical8;
    using type = logical<8>;
    using data_type = simd::detail::mask_logical_traits<W,VL>;
    using mask_type = data_type;
    using encoding = array_encoding<type>;
};

template<unsigned short VL_>
struct frontier_params<frontier_type::ft_msb4,VL_> {
    static constexpr unsigned short W = 4;
    static constexpr unsigned short VL = VL_;
    static constexpr frontier_type ftype = frontier_type::ft_msb4;
    using type = logical<4>;
    using data_type = simd::detail::mask_logical_traits<W,VL>;
    using mask_type = data_type;
    using encoding = array_encoding_msb<W,true>;
};


template<typename lVID>
struct BitReader {
    using VID = lVID;
    
    BitReader( const unsigned char * flags, VID off = 0 )
	: m_flags( flags ), m_off( off ) { }

    bool operator [] ( VID i ) const {
	i += m_off;
	return ( ( m_flags[i/8] >> (i%8) ) & ((unsigned char)1) ) != 0;
    }

    const unsigned char * m_flags;
    VID m_off;
};

template<typename lVID>
BitReader<lVID> operator + ( BitReader<lVID> b, lVID off ) {
    return BitReader<lVID>( b.m_flags, b.m_off + off );
}

template<typename lVID>
struct BitReader2 {
    using VID = lVID;
    
    BitReader2( const unsigned char * flags, VID off = 0 )
	: m_flags( flags ), m_off( off ) { }

    bool operator [] ( VID i ) const {
	i += m_off;
	// Only check the top of the two bits (logical mask)
	return ( ( m_flags[i/4] >> (2*(i%4)+1) ) & ((unsigned char)1) ) != 0;
    }

    const unsigned char * m_flags;
    VID m_off;
};

template<typename lVID>
BitReader2<lVID> operator + ( BitReader2<lVID> b, lVID off ) {
    return BitReader<lVID>( b.m_flags, b.m_off + off );
}

class frontier {
    /*************************************************************
     * Creation and destruction
     *************************************************************/
public:
    // Create functions (no constructors used)
    [[deprecated("redundant")]]
    static frontier all_true( const partitioner & part,
			      VID num_vertices, EID num_edges ) {
	return all_true( num_vertices, num_edges );
    }
    static frontier all_true( VID num_vertices, EID num_edges ) {
	// std::cerr << "FRONTIER create all true\n";
	frontier f;
	f.nv = num_vertices;
	f.nactv = num_vertices;
	f.nacte = num_edges;
	f.ftype = frontier_type::ft_true;
        return f;
    }
    static frontier empty() {
	// std::cerr << "FRONTIER create empty (sparse)\n";
	frontier f;
	f.nv = 0;
	f.nactv = 0;
	f.nacte = 0;
	f.get_s() = nullptr;
	f.ftype = frontier_type::ft_sparse;
	return f;
    }
    static frontier unbacked( const partitioner & part ) {
	frontier f;
	f.nv = part.get_num_elements();
	f.nactv = 0;
	f.nacte = 0;
	f.get_s() = nullptr; // any/all of the pointers should be null
	f.ftype = frontier_type::ft_unbacked;
	return f;
    }
    static frontier bit( const partitioner & part ) {
	frontier f;
	f.nv = part.get_num_elements();
	f.nactv = 0;
	f.nacte = 0;

	partitioner cpart = part.contract( sizeof(unsigned char) );
	mmap_ptr<unsigned char> & fb = f.get_bit();
	new ( &fb ) mmap_ptr<unsigned char>();
	fb.allocate( numa_allocation_partitioned( cpart ) );

	// TODO: is it necessary to zero out? avoidable (mmap)?
	unsigned char * d = f.get_bit().get();
        map_partitionL( part, [&]( unsigned int p ) {
		std::fill( &d[cpart.start_of(p)], &d[cpart.start_of(p+1)], 0 );
	    } );

	f.ftype = frontier_type::ft_bit;
	return f;
    }
    static frontier bit2( const partitioner & part ) {
	frontier f;
	f.nv = part.get_num_elements();
	f.nactv = 0;
	f.nacte = 0;

	partitioner cpart = part.contract( 4 ); // 8 bits / 2 bits per elm
	mmap_ptr<unsigned char> & fb = f.get_bit2();
	new ( &fb ) mmap_ptr<unsigned char>();
	fb.allocate( numa_allocation_partitioned( cpart ) );

	// TODO: is it necessary to zero out? avoidable (mmap)?
	unsigned char * d = f.get_bit2().get();
        map_partitionL( part, [&]( unsigned int p ) {
		std::fill( &d[cpart.start_of(p)], &d[cpart.start_of(p+1)], 0 );
	    } );

	f.ftype = frontier_type::ft_bit2;
	return f;
    }
    
    // Create method short-cut, only applicable to dense frontiers.
    // Sparse and true frontiers require additional information.
    template<frontier_type ftype>
    static frontier create( const partitioner & part );

    template<unsigned short W>
    static frontier dense( const partitioner & part ) {
	// std::cerr << "FRONTIER create dense logical " << W << "\n";
	static_assert( (W & (W-1)) == 0 && W > 0 && W <= 8,
		       "W must be one of 1, 2, 4 or 8" );

	frontier f;
	f.nv = part.get_num_elements();
	f.nactv = 0;
	f.nacte = 0;
	mmap_ptr<logical<W>> & fb = f.get_l<W>();
	new ( &fb ) mmap_ptr<logical<W>>();
	fb.allocate( numa_allocation_partitioned( part ) );

	// TODO: is it necessary to zero out? avoidable (mmap)?
	logical<W> * d = fb.get();
	map_vertexL( part, [&]( VID v ) { d[v]=logical<W>::false_val(); } );

	switch( W ) {
	case 1: f.ftype = frontier_type::ft_logical1; break;
	case 2: f.ftype = frontier_type::ft_logical2; break;
	case 4: f.ftype = frontier_type::ft_logical4; break;
	case 8: f.ftype = frontier_type::ft_logical8; break;
	default: UNREACHABLE_CASE_STATEMENT;
	}
	return f;
    }
    // template<typename VertexProp>
    // static frontier msb( const partitioner & part, VertexProp & p ) {
	// static constexpr unsigned short W = sizeof(typename VertexProp::type);
    template<typename type>
    static frontier msb( const partitioner & part, type * p ) {
	static constexpr unsigned short W = sizeof(type);
	static_assert( (W & (W-1)) == 0 && W > 0 && W <= 8,
		       "W must be one of 1, 2, 4 or 8" );

	frontier f;
	f.nv = part.get_num_elements();
	f.nactv = 0;
	f.nacte = 0;
	mmap_ptr<logical<W>> & fb = f.get_l<W>();
	new ( &fb ) mmap_ptr<logical<W>>();
	fb.set_allocation( part.get_num_vertices() * W, p /*.get_ptr()*/ );

	// TODO: Should use VertexProp::encoding to access data
	// TODO: how to put together constructor? for now, not clear...
	logical<W> * d = fb.get();
	map_vertexL( part, [&]( VID v ) { d[v].clear_msb(); } );

	switch( W ) {
	case 4: f.ftype = frontier_type::ft_msb4; break;
	default: UNREACHABLE_CASE_STATEMENT;
	}
	return f;
    }
    static frontier dense( const partitioner & part, VID num_vertices ) {
	// std::cerr << "FRONTIER create dense bool\n";
	// TODO: get num_vertices argument from part argument
	frontier f;
	f.nv = num_vertices;
	f.nactv = 0;
	f.nacte = 0;
	mmap_ptr<bool> & fb = f.get_b();
	new ( &fb ) mmap_ptr<bool>();
	fb.allocate( numa_allocation_partitioned( part ) );

	// TODO: is it necessary to zero out? avoidable (mmap)?
	bool * d = f.get_b().get();
        map_vertexL( part, [&]( VID v ) { d[v]=false; } );

	f.ftype = frontier_type::ft_bool;
	return f;
    }
    static frontier dense( VID num_vertices, const partitioner & part ) {
	return dense( part, num_vertices ); // TODO: ease migration
    }
    static frontier sparse( VID nv, VID nactv_max ) {
	// std::cerr << "FRONTIER create sparse\n";
        frontier f;
	f.nv = nv;
	f.nactv = 0;
	f.nacte = 0;
	f.get_s() = new VID[nactv_max];
        f.ftype = frontier_type::ft_sparse;
        return f;
    }
    static frontier sparse( VID nv, VID nactv, VID * ptr ) {
	// std::cerr << "FRONTIER create sparse\n";
        frontier f;
	f.nv = nv;
	f.nactv = nactv;
	f.nacte = 0;
	f.get_s() = ptr;
        f.ftype = frontier_type::ft_sparse;
        return f;
    }
    static frontier create( VID nv, VID v, VID odeg ) {
	// std::cerr << "FRONTIER create one\n";
        frontier f;
	f.nv = nv;
	f.nactv = 1;
	f.nacte = odeg;
	f.get_s() = new VID[1];
	f.get_s()[0] = v;
        f.ftype = frontier_type::ft_sparse;
        return f;
    }

    // Legacy
    [[deprecated("replaced by all_true()")]]
    static frontier bits( const partitioner & part,
			  VID num_vertices, EID num_edges ) {
	return all_true( num_vertices, num_edges );
    }
    //Used for vertexFilter to create the array with number of vertices and boolean array
public:
    // Cleanup of the frontier (no destructor to allow easy but unsafe
    // sharing - remnant of Ligra)
    void del()
    {
	// std::cerr << "FRONTIER del\n";
	switch( ftype ) {
	case frontier_type::ft_true:
	case frontier_type::ft_unbacked:
	    break;
	case frontier_type::ft_bool:
	    get_b().del();
	    break;
	case frontier_type::ft_bit:
	    get_bit().del();
	    break;
	case frontier_type::ft_bit2:
	    get_bit2().del();
	    break;
	case frontier_type::ft_logical1: get_l<1>().del(); break;
	case frontier_type::ft_logical2: get_l<2>().del(); break;
	case frontier_type::ft_logical4: get_l<4>().del(); break;
	case frontier_type::ft_logical8: get_l<8>().del(); break;
	case frontier_type::ft_msb4: break; // frontier does not own data
	case frontier_type::ft_sparse:
	    if( VID * s = get_s() ) // s is null in case of ::empty()
		delete[] s;
	    break;
	default: UNREACHABLE_CASE_STATEMENT;
	}
	ftype = frontier_type::ft_true; // for safety
    }
    frontier copy( const partitioner & part, EID m ) {
	frontier f;
        // Calculate statistics on active vertices and their out-degree
	switch( ftype ) {
	case frontier_type::ft_true:
	    return all_true( nv, m );
	case frontier_type::ft_bool:
	    assert( 0 && "NYI" );
	    break;
	case frontier_type::ft_bit:
	case frontier_type::ft_bit2:
	    assert( 0 && "NYI" );
	    break;
	case frontier_type::ft_logical1:
	{
	    f.nv = part.get_num_elements();
	    f.nactv = 0;
	    f.nacte = 0;
	    f.ftype = frontier_type::ft_logical1;
	    mmap_ptr<logical<1>> & fb = f.get_l<1>();
	    new ( &fb ) mmap_ptr<logical<1>>();
	    fb.allocate( numa_allocation_partitioned( part ) );
	    logical<1> * d = fb.get();
	    logical<1> * dd = get_l<1>();
	    map_vertexL(part, [&](VID j) { d[j]=dd[j]; });
	    break;
	}
	case frontier_type::ft_logical2:
	{
	    f.nv = part.get_num_elements();
	    f.nactv = 0;
	    f.nacte = 0;
	    f.ftype = frontier_type::ft_logical2;
	    mmap_ptr<logical<2>> & fb = f.get_l<2>();
	    new ( &fb ) mmap_ptr<logical<2>>();
	    fb.allocate( numa_allocation_partitioned( part ) );
	    logical<2> * d = fb.get();
	    logical<2> * dd = get_l<2>();
	    map_vertexL(part, [&](VID j){d[j]=dd[j];});
	    break;
	}
	case frontier_type::ft_logical4:
	{
	    f.nv = part.get_num_elements();
	    f.nactv = 0;
	    f.nacte = 0;
	    f.ftype = frontier_type::ft_logical4;
	    mmap_ptr<logical<4>> & fb = f.get_l<4>();
	    new ( &fb ) mmap_ptr<logical<4>>();
	    fb.allocate( numa_allocation_partitioned( part ) );
	    logical<4> * d = fb.get();
	    logical<4> * dd = get_l<4>();
	    map_vertexL(part,[&](VID j){d[j]=dd[j];});
	    break;
	}
	case frontier_type::ft_logical8:
	{
	    f.nv = part.get_num_elements();
	    f.nactv = 0;
	    f.nacte = 0;
	    f.ftype = frontier_type::ft_logical8;
	    mmap_ptr<logical<8>> & fb = f.get_l<8>();
	    new ( &fb ) mmap_ptr<logical<8>>();
	    fb.allocate( numa_allocation_partitioned( part ) );
	    logical<8> * d = fb.get();
	    logical<8> * dd = get_l<8>();
	    map_vertexL(part,[&](VID j){d[j]=dd[j];});
	    break;
	}
	case frontier_type::ft_sparse:
	    assert( 0 && "NYI" );
	    break;
	default: UNREACHABLE_CASE_STATEMENT;
	}
	return f;
    }

    /*************************************************************
     * Update active counts after external update of frontier
     *************************************************************/
public:
    // Note: these functions must be passed the CSR graph
    template<class vertex>
    inline void calculateActiveCounts( graph<vertex> G, VID n = ~(VID)0 ); 
    template<typename GraphType>
    inline void calculateActiveCounts( const GraphType & G );
    inline void calculateActiveCounts( GraphCSx G, VID from, VID to );
    inline void calculateActiveCounts( const GraphCSx & G,
				       const partitioner & part,
				       VID n );
private:
    template<typename FlagsTy>
    void calculateActiveCounts_tmpl( const partitioner & part,
				     const VID * outdeg,
				     const FlagsTy * flags );

public:
    void setActiveCounts( VID v, EID e ) {
	nactv = v;
	nacte = e;
    }
    void resetActiveCounts() {
	nactv = 0;
	nacte = 0;
    }

    /*************************************************************
     * Query functions
     *************************************************************/
public:
    template<typename T>
    void toDense(const partitioner & part);

    template<frontier_type ftype>
    void toDense(const partitioner & part);

    void toDense(const partitioner & part) {
	toBool( part ); // TODO: temporary
    }

    void toBool( const partitioner & part ) {
	switch( ftype ) {
	case frontier_type::ft_true: break;
	case frontier_type::ft_unbacked: break;
	case frontier_type::ft_bool: break;
	case frontier_type::ft_bit:
	case frontier_type::ft_bit2:
	    assert( 0 && "NYI" );
	    break;
	case frontier_type::ft_logical1:
	{
	    // std::cerr << "FRONTIER CONVERT frontier from logical<1> to bool\n";
	    // No need to reallocate memory. Just reinterpret the values,
	    // and remap 0->0 and ~0 -> 1. One may argue whether any action
	    // is required at all as usually ~0 is interpreted as true.
	    // However, ~0 != true, so it depends on coding style.
	    logical<1> * p = get_l<1>().get();
	    char * cp = reinterpret_cast<char *>( p );
	    // TODO: Use Cilk array extensions to facilitate auto-vectorisation.
	    // cp[0:nv] = ( cp[0:nv] != (char)0 );
	    // TODO: parallelise as well as vectorise
	    for( VID v=0; v < nv; ++v )
		cp[v] = ( cp[v] != (char)0 );
	    ftype = frontier_type::ft_bool;
	    break;
	}
	case frontier_type::ft_logical2:
	{
	    assert( 0 && "NYI" );
	    break;
	}
	case frontier_type::ft_logical4:
	{
	    // std::cerr << "FRONTIER CONVERT frontier from logical<4> to bool\n";
	    mmap_ptr<logical<4>> l4( std::move( get_l<4>() ) );
	    mmap_ptr<bool> & b = get_b();
	    new ( &b ) mmap_ptr<bool>();
	    b.allocate( numa_allocation_partitioned( part ) );
	    // TODO: Use Cilk array extensions to facilitate auto-vectorisation.
	    // TODO: parallelise as well as vectorise
	    // b.get()[0:nv] = l4.get()[0:nv];
	    for( VID v=0; v < nv; ++v )
		b.get()[v] = (bool)l4.get()[v];

	    l4.del();
	    ftype = frontier_type::ft_bool;
	    break;
	}
	case frontier_type::ft_logical8:
	{
	    // std::cerr << "FRONTIER CONVERT frontier from logical<8> to bool\n";
	    mmap_ptr<logical<8>> l8( std::move( get_l<8>() ) );
	    mmap_ptr<bool> & b = get_b();
	    new ( &b ) mmap_ptr<bool>();
	    b.allocate( numa_allocation_partitioned( part ) );
	    // TODO: Use Cilk array extensions to facilitate auto-vectorisation.
	    // TODO: parallelise as well as vectorise
	    // b.get()[0:nv] = l8.get()[0:nv];
	    for( VID v=0; v < nv; ++v )
		b.get()[v] = (bool)l8.get()[v];

	    l8.del();
	    ftype = frontier_type::ft_bool;
	    break;
	}
	case frontier_type::ft_sparse:
	{
	    // std::cerr << "FRONTIER CONVERT frontier from sparse to bool\n";
	    VID * s = get_s();
	    mmap_ptr<bool> & b = get_b();
	    new ( &b ) mmap_ptr<bool>();
	    b.allocate( numa_allocation_partitioned( part ) );
	    bool *bp = b.get();
	    map_vertexL( part, [&]( VID v ) { bp[v]=0; } );
	    parallel_loop( (VID)0, nactv, [&]( VID i ) { bp[s[i]] = 1; } );
	    delete[] s;
	    ftype = frontier_type::ft_bool;
	    break;
	}
	default: UNREACHABLE_CASE_STATEMENT;
	}
    }

    void toBit( const partitioner & part ) {
	switch( ftype ) {
	case frontier_type::ft_true: break;
	case frontier_type::ft_unbacked: break;
	case frontier_type::ft_bool:
	{
	    partitioner cpart = part.contract( sizeof(unsigned char) );
	    mmap_ptr<bool> bo( std::move( get_b() ) );
	    mmap_ptr<unsigned char> & b = get_bit();
	    new ( &b ) mmap_ptr<unsigned char>();
	    b.allocate( numa_allocation_partitioned( cpart ) );
	    for( VID v=0; v < nv; v += 8 ) {
		unsigned char m = 0;
		for( VID vs=0; vs < 8; ++vs ) {
		    bool val = bo.get()[v+8-vs];
		    m <<= 1;
		    m |= (val != 0);
		}
		b.get()[v/8] = m;
	    }

	    bo.del();
	    ftype = frontier_type::ft_bit;
	    break;
	}
	case frontier_type::ft_bit: break;
	case frontier_type::ft_bit2: break;
	case frontier_type::ft_logical1:
	case frontier_type::ft_logical2:
	    assert( 0 && "NYI" );
	    break;
	case frontier_type::ft_logical4:
	{
	    partitioner cpart = part.contract( sizeof(unsigned char) );
	    mmap_ptr<logical<4>> l4( std::move( get_l<4>() ) );
	    mmap_ptr<unsigned char> & b = get_bit();
	    new ( &b ) mmap_ptr<unsigned char>();
	    b.allocate( numa_allocation_partitioned( part ) );

#if __AVX512F__
	    unsigned short * bp = reinterpret_cast<unsigned short *>( b.get() );
	    for( VID v=0; v < nv; v += 16 ) {
		auto vv = simd::vector<logical<4>,16>::load_from( &l4.get()[v] );
		simd::mask<0,16> m = vv.template asmask<simd::detail::mask_bit_traits<16>>();
		bp[v/16] = m.get(); // div 16 because unsigned short *
	    }
#elif __AVX2__
	    unsigned char * bp = reinterpret_cast<unsigned char *>( b.get() );
	    for( VID v=0; v < nv; v += 8 ) {
		auto vv = simd::vector<logical<4>,8>::load_from( &l4.get()[v] );
		simd::mask<0,8> m = vv.template asmask<simd::detail::mask_bit_traits<8>>();
		bp[v/8] = m.get(); // div 8 because unsigned char *
	    }
#else
	    assert( 0 && "Not supported" );
#endif

	    l4.del();
	    ftype = frontier_type::ft_bit;
	    break;
	}
	case frontier_type::ft_logical8:
	{
	    assert( 0 && "NYI" );
	    mmap_ptr<logical<8>> l8( std::move( get_l<8>() ) );
	    mmap_ptr<bool> & b = get_b();
	    new ( &b ) mmap_ptr<bool>();
	    b.allocate( numa_allocation_partitioned( part ) );
	    // TODO: Use Cilk array extensions to facilitate auto-vectorisation.
	    // TODO: parallelise as well as vectorise
	    // b.get()[0:nv] = l8.get()[0:nv];
	    for( VID v=0; v < nv; ++v )
		b.get()[v] = (bool)l8.get()[v];

	    l8.del();
	    ftype = frontier_type::ft_bool;
	    break;
	}
	case frontier_type::ft_sparse:
	{
	    partitioner cpart = part.contract( sizeof(unsigned char) );
	    VID * s = get_s();
	    mmap_ptr<unsigned char> & b = get_bit();
	    new ( &b ) mmap_ptr<unsigned char>();
	    b.allocate( numa_allocation_partitioned( cpart ) );
	    unsigned char *bp = b.get();
	    map_vertexL( cpart, [&]( VID v ) { bp[v]=0; } );
	    /*parallel_*/for( VID i=0; i < nactv; i++ ) {
		VID idx = s[i] / 8;
		bp[idx] |= ((unsigned char)1) << ( s[i] % 8 );
	    }
	    delete[] s;
	    ftype = frontier_type::ft_bit;
	    break;
	}
	default: UNREACHABLE_CASE_STATEMENT;
	}
    }
    void toBit2( const partitioner & part ) {
	switch( ftype ) {
	case frontier_type::ft_true: break;
	case frontier_type::ft_unbacked: break;
	case frontier_type::ft_bool:
	{
	    partitioner cpart = part.contract( 4 );
	    mmap_ptr<bool> bo( std::move( get_b() ) );
	    mmap_ptr<unsigned char> & b = get_bit2();
	    new ( &b ) mmap_ptr<unsigned char>();
	    b.allocate( numa_allocation_partitioned( cpart ) );
	    for( VID v=0; v < nv; v += 4 ) {
		unsigned char m = 0;
		for( VID vs=0; vs < 8; ++vs ) {
		    bool val = bo.get()[v+8-vs];
		    m <<= 2;
		    m |= (val == 0 ? 0 : 3);
		}
		b.get()[v/8] = m;
	    }

	    bo.del();
	    ftype = frontier_type::ft_bit;
	    break;
	}
	case frontier_type::ft_bit:
	{
	    assert( 0 && "NYI" );
	    break;
	}
	case frontier_type::ft_bit2: break;
	case frontier_type::ft_logical1:
	case frontier_type::ft_logical2:
	    assert( 0 && "NYI" );
	    break;
	case frontier_type::ft_logical4:
	{
	    assert( 0 && "NYI" );
	    partitioner cpart = part.contract( sizeof(unsigned char) );
	    mmap_ptr<logical<4>> l4( std::move( get_l<4>() ) );
	    mmap_ptr<unsigned char> & b = get_bit2();
	    new ( &b ) mmap_ptr<unsigned char>();
	    b.allocate( numa_allocation_partitioned( part ) );
	    unsigned short * bp = reinterpret_cast<unsigned short *>( b.get() );

#if __AVX512F__
	    for( VID v=0; v < nv; v += 16 ) {
		auto vv = simd::vector<logical<4>,16>::load_from( &l4.get()[v] );
		simd::mask<0,16> m = vv.template asmask<simd::detail::mask_bit_traits<16>>();
		bp[v/16] = m.get(); // div 8 because unsigned char *
	    }
#else
	    assert( 0 && "Not supported" );
#endif

	    l4.del();
	    ftype = frontier_type::ft_bit;
	    break;
	}
	case frontier_type::ft_logical8:
	{
	    assert( 0 && "NYI" );
	    mmap_ptr<logical<8>> l8( std::move( get_l<8>() ) );
	    mmap_ptr<bool> & b = get_b();
	    new ( &b ) mmap_ptr<bool>();
	    b.allocate( numa_allocation_partitioned( part ) );
	    // TODO: Use Cilk array extensions to facilitate auto-vectorisation.
	    // TODO: parallelise as well as vectorise
	    // b.get()[0:nv] = l8.get()[0:nv];
	    for( VID v=0; v < nv; ++v )
		b.get()[v] = (bool)l8.get()[v];

	    l8.del();
	    ftype = frontier_type::ft_bool;
	    break;
	}
	case frontier_type::ft_sparse:
	{
	    VID * s = get_s();
	    mmap_ptr<unsigned char> & b = get_bit2();
	    new ( &b ) mmap_ptr<unsigned char>();
	    b.allocate( numa_allocation_partitioned( part ) );
	    unsigned char *bp = b.get();

	    partitioner cpart = part.contract( 4 ); // 8 bits / 2 bits per elm
	    map_vertexL( cpart, [&]( VID v ) { bp[v]=0; } );

	    parallel_loop( (VID)0, nactv, [&]( VID i ) {
		bp[s[i]/4] |= 3 << ((s[i]%4)*2);
	    } );
	    delete[] s;
	    ftype = frontier_type::ft_bit2;
	    break;
	}
	default: UNREACHABLE_CASE_STATEMENT;
	}
    }


    template<unsigned short B>
	void toLogical( const partitioner & part ) {
	if( B == 8 )
	    toLogical8( part );
	else if( B == 4 )
	    toLogical4( part );
	else if( B ==  2 )
	    toLogical2( part );
	else if( B ==  1 )
	    toLogical1( part );
	else
	    assert( 0 && "NYI" );
    }

    void toLogical1( const partitioner & part ) {
	switch( ftype ) {
	case frontier_type::ft_true: break;
	case frontier_type::ft_unbacked: break;
	case frontier_type::ft_bool:
	case frontier_type::ft_bit:
	    assert( 0 && "NYI" );
	    break;
	case frontier_type::ft_logical2:
	{
	    convert_logical<2,1>( part );
	    ftype = frontier_type::ft_logical1;
	    break;
	}
	case frontier_type::ft_logical4:
	{
	    convert_logical<4,1>( part );
	    ftype = frontier_type::ft_logical1;
	    break;
	}
	case frontier_type::ft_logical8:
	{
	    convert_logical<8,1>( part );
	    ftype = frontier_type::ft_logical1;
	    break;
	}
	case frontier_type::ft_logical1:
	    break; // nothing to do
	case frontier_type::ft_sparse:
	{
	    VID * s = get_s();
	    mmap_ptr<logical<1>> & l1 = get_l<1>();
	    new ( &l1 ) mmap_ptr<logical<1>>();
	    l1.allocate( numa_allocation_partitioned( part ) );
	    logical<1> *l1p = l1.get();
	    map_vertexL( part, [&](VID v) { l1p[v]=logical<1>::false_val(); } );
	    parallel_loop( (VID)0, nactv, [&]( VID i ) {
		assert( !l1p[s[i]] );
		l1p[s[i]] = logical<1>::true_val();
	    } );

	    delete[] s;
	    ftype = frontier_type::ft_logical1;
	    break;
	}
	default: UNREACHABLE_CASE_STATEMENT;
	}
    }
    template<typename From, typename To>
    static void convert_logical( const partitioner & part,
				 From * fromp, To * top );

    template<unsigned short FromW, unsigned short ToW>
    void convert_logical( const partitioner & part ) {
	mmap_ptr<logical<FromW>> l1( std::move( get_l<FromW>() ) );
	logical<FromW> *l1p = l1.get();

	mmap_ptr<logical<ToW>> & l2 = get_l<ToW>();
	new ( &l2 ) mmap_ptr<logical<ToW>>();
	l2.allocate( numa_allocation_partitioned( part ) );
	logical<ToW> *l2p = l2.get();

	convert_logical<logical<FromW>, logical<ToW>>( part, l1p, l2p );

	l1.del();
    }
    
    void toLogical2( const partitioner & part ) {
	switch( ftype ) {
	case frontier_type::ft_true: break;
	case frontier_type::ft_unbacked: break;
	case frontier_type::ft_bool:
	case frontier_type::ft_bit:
	case frontier_type::ft_logical8:
	    assert( 0 && "NYI" );
	    break;
	case frontier_type::ft_logical1:
	{
	    convert_logical<1,2>( part );
	    ftype = frontier_type::ft_logical2;
	    break;
	}
	case frontier_type::ft_logical4:
	{
	    convert_logical<4,2>( part );
	    ftype = frontier_type::ft_logical2;
	    break;
	}
	case frontier_type::ft_logical2:
	    break; // nothing to do
	case frontier_type::ft_sparse:
	{
	    VID * s = get_s();
	    mmap_ptr<logical<2>> & l2 = get_l<2>();
	    new ( &l2 ) mmap_ptr<logical<2>>();
	    l2.allocate( numa_allocation_partitioned( part ) );
	    logical<2> *l2p = l2.get();
	    map_vertexL( part, [&](VID v) { l2p[v]=logical<2>::false_val(); } );
	    /*parallel_*/for( VID i=0; i < nactv; i++ ) {
		assert( !l2p[s[i]] );
		l2p[s[i]] = logical<2>::true_val();
	    }

	    delete[] s;
	    ftype = frontier_type::ft_logical2;
	    break;
	}
	default: UNREACHABLE_CASE_STATEMENT;
	}
    }
    void toLogical4( const partitioner & part ) {
	switch( ftype ) {
	case frontier_type::ft_true: break;
	case frontier_type::ft_unbacked: break;
	case frontier_type::ft_bool:
	{
	    // std::cerr << "FRONTIER CONVERT frontier from bool to logical<4>\n";
	    mmap_ptr<bool> b( std::move( get_b() ) );
	    mmap_ptr<logical<4>> & l4 = get_l<4>();
	    new ( &l4 ) mmap_ptr<logical<4>>();
	    l4.allocate( numa_allocation_partitioned( part ) );
	    logical<4> * l4p = l4.get();
	    bool * bp = b.get();
	    // TODO: Use Cilk array extensions to facilitate auto-vectorisation.
	    // TODO: parallelise as well as vectorise
	    // b.get()[0:nv] = l4.get()[0:nv];
	    for( VID v=0; v < nv; ++v )
		l4p[v] = logical<4>::get_val( bp[v] );

	    b.del();
	    ftype = frontier_type::ft_logical4;
	    break;
	}
	case frontier_type::ft_bit:
	{
	    partitioner cpart = part.contract( sizeof(unsigned char) );
	    mmap_ptr<unsigned char> b( std::move( get_bit() ) );
	    mmap_ptr<logical<4>> & l4 = get_l<4>();
	    new ( &l4 ) mmap_ptr<logical<4>>();
	    l4.allocate( numa_allocation_partitioned( part ) );
	    logical<4> * l4p = l4.get();
	    unsigned char * bp = b.get();
#if __AVX512F__
	    using vtraits = vector_type_traits_vl<logical<4>,8>;
	    map_partitionL( cpart, [&]( VID v ) {
		    unsigned char m = bp[v];
		    typename vtraits::type vm = vtraits::asvector( m );
		    vtraits::store( &l4p[v*8], vm );
		} );
#else
	    assert( 0 && "Not supported" );
#endif
	    b.del();
	    ftype = frontier_type::ft_logical4;
	    break;
	}
	case frontier_type::ft_logical4:
	    break; // nothing to do
	case frontier_type::ft_msb4:
	    // Leave as is; works nearly same way as logical<4>, so why change?
	    break;
	case frontier_type::ft_logical1:
	{
	    convert_logical<1,4>( part );
	    ftype = frontier_type::ft_logical4;
	    break;
	}
	case frontier_type::ft_logical8:
	{
	    convert_logical<8,4>( part );
	    ftype = frontier_type::ft_logical4;
	    break;
	}
	case frontier_type::ft_logical2:
	{
	    convert_logical<2,4>( part );
	    ftype = frontier_type::ft_logical4;
	    break;
	}
	case frontier_type::ft_sparse:
	{
	    // std::cerr << "FRONTIER CONVERT frontier from sparse to logical<4>\n";
	    VID * s = get_s();
	    mmap_ptr<logical<4>> & l4 = get_l<4>();
	    new ( &l4 ) mmap_ptr<logical<4>>();
	    l4.allocate( numa_allocation_partitioned( part ) );
	    logical<4> *l4p = l4.get();
	    map_vertexL( part, [&](VID v) { l4p[v]=logical<4>::false_val(); } );
	    parallel_loop( (VID)0, nactv, [&]( VID i ) {
		// Assertion meaningful only when prior sparse edgemaps
		// are guaranteed to remove duplicates
		// assert( !l4p[s[i]] );
		l4p[s[i]] = logical<4>::true_val();
	    } );

	    delete[] s;
	    ftype = frontier_type::ft_logical4;
	    break;
	}
	default: UNREACHABLE_CASE_STATEMENT;
	}
    }
    void toLogical8( const partitioner & part ) {
	switch( ftype ) {
	case frontier_type::ft_true: break;
	case frontier_type::ft_unbacked: break;
	case frontier_type::ft_bool: break;
	{
	    // std::cerr << "FRONTIER CONVERT frontier from bool to logical<8>\n";
	    mmap_ptr<bool> b( std::move( get_b() ) );
	    mmap_ptr<logical<8>> & l8 = get_l<8>();
	    new ( &l8 ) mmap_ptr<logical<8>>();
	    l8.allocate( numa_allocation_partitioned( part ) );
	    logical<8> * l8p = l8.get();
	    bool * bp = b.get();
	    // TODO: Use Cilk array extensions to facilitate auto-vectorisation.
	    // TODO: parallelise as well as vectorise
	    // b.get()[0:nv] = l8.get()[0:nv];
	    for( VID v=0; v < nv; ++v )
		l8p[v] = logical<8>::get_val( bp[v] );

	    b.del();
	    ftype = frontier_type::ft_logical8;
	    break;
	}
	case frontier_type::ft_bit:
	case frontier_type::ft_logical1:
	case frontier_type::ft_logical2:
	case frontier_type::ft_logical4:
	    assert( 0 && "NYI" );
	    break;
	case frontier_type::ft_sparse:
	{
	    // std::cerr << "FRONTIER CONVERT frontier from sparse to logical<8>\n";
	    VID * s = get_s();
	    mmap_ptr<logical<8>> & l8 = get_l<8>();
	    new ( &l8 ) mmap_ptr<bool>();
	    l8.allocate( numa_allocation_partitioned( part ) );
	    logical<8> *l8p = l8.get();
	    map_vertexL( part, [&](VID v) { l8p[v]=logical<8>::false_val(); } );
	    parallel_loop( (VID)0, nactv, [&]( VID i ) {
		l8p[s[i]] = logical<8>::true_val();
	    } );

	    delete[] s;
	    ftype = frontier_type::ft_logical8;
	    break;
	}
	default: UNREACHABLE_CASE_STATEMENT;
	}
    }

    void toSparse( const partitioner & part ) {
	switch( ftype ) {
	case frontier_type::ft_true: break;
	case frontier_type::ft_sparse: break;
	case frontier_type::ft_unbacked:
	{
	    // Assume nactv is correct; simply allocate backing storage.
	    get_s() = new VID[nactv];
	    ftype = frontier_type::ft_sparse;
	    break;
	}
	case frontier_type::ft_bool:
	{
	    _seq<VID> R = packDense( part, get_b().get() );
	    replace_storage( R );
	    break;
	}
	case frontier_type::ft_bit:
	{
	    _seq<VID> R = packDenseBit( part, get_bit().get() );
	    replace_storage( R );
	    break;
	}
	case frontier_type::ft_bit2:
	{
	    _seq<VID> R = packDenseBit2( part, get_bit2().get() );
	    replace_storage( R );
	    break;
	}
	case frontier_type::ft_logical1:
	{
	    _seq<VID> R = packDense( part, get_l<1>().get() );
	    replace_storage( R );
	    break;
	}
	case frontier_type::ft_logical2:
	{
	    _seq<VID> R = packDense( part, get_l<2>().get() );
	    replace_storage( R );
	    break;
	}
	case frontier_type::ft_logical4:
	case frontier_type::ft_msb4:
	{
	    // Same code applies to ft_logical4 and ft_msb4 as we should only
	    // consult the MSB to decide on true/false in both cases
	    _seq<VID> R = packDense( part, get_l<4>().get() );
	    replace_storage( R );
	    break;
	}
	case frontier_type::ft_logical8:
	{
	    _seq<VID> R = packDense( part, get_l<8>().get() );
	    replace_storage( R );
	    break;
	}
	default: UNREACHABLE_CASE_STATEMENT;
	}
    }

    /*************************************************************
     * Query functions
     *************************************************************/
    // Note: this function decides based on the type only. It may return
    //       false when in fact all vertices are set active. It will never
    //       return true if not all vertices are active.
    frontier_type getType() const { return ftype; }
    bool allTrue() const { return ftype == frontier_type::ft_true; }
    bool isDense() const {
	return ftype == frontier_type::ft_bit || ftype == frontier_type::ft_bool
	    || ftype == frontier_type::ft_logical1 || ftype == frontier_type::ft_logical2
	    || ftype == frontier_type::ft_logical4 || ftype == frontier_type::ft_logical8;
    }
    // [[deprecated("name change")]]
    bool hasDense() const { return isDense(); }
    bool isSparse() const { return ftype == frontier_type::ft_sparse; }
    // Returns the dense bool array, if it exists
    bool * getDenseB() const {
	return ftype == frontier_type::ft_bool ? get_b().get() : nullptr;
    }
    unsigned char * getDenseBit() const {
	return ftype == frontier_type::ft_bit ? get_bit().get() : nullptr;
    }
    unsigned char * getDenseBit2() const {
	return ftype == frontier_type::ft_bit2 ? get_bit2().get() : nullptr;
    }
    // Returns the dense logical<W> array, if it exists
    template<unsigned short W>
    logical<W> * getDenseL() const {
	switch( ftype ) {
	case frontier_type::ft_true:
	case frontier_type::ft_bool:
	case frontier_type::ft_bit:
	case frontier_type::ft_bit2:
	case frontier_type::ft_sparse:
	case frontier_type::ft_unbacked:
	    return nullptr;
	case frontier_type::ft_logical1: return W == 1 ? get_l<W>().get() : nullptr;
	case frontier_type::ft_logical2: return W == 2 ? get_l<W>().get() : nullptr;
	case frontier_type::ft_logical4: return W == 4 ? get_l<W>().get() : nullptr;
	case frontier_type::ft_logical8: return W == 8 ? get_l<W>().get() : nullptr;
	case frontier_type::ft_msb4: return W == 4 ? get_l<W>().get() : nullptr;
	default: UNREACHABLE_CASE_STATEMENT;
	}
    }
    VID * getSparse() const {
	return ftype == frontier_type::ft_sparse ? get_s() : nullptr;
    }

    // Helper
    template<typename T>
    T * getDense() const;
    template<frontier_type ft>
    typename frontier_params<ft,0>::type * getDense() const;

    [[deprecated("Replaced nVertices")]]
    VID numRows() const { return nv; } // TODO: migration
    VID nVertices() const { return nv; }
    VID nActiveVertices() const { return nactv; }
    [[deprecated("Replaced by nActiveVertices which has better name")]]
    VID numNonzeros() const { return nactv; } // TODO: migration
    EID nActiveEdges() const { return nacte; }

    VID * nActiveVerticesPtr() { return &nactv; }
    EID * nActiveEdgesPtr() { return &nacte; }

    bool isEmpty() const { return nActiveEdges() == 0; }

    double density( EID m ) const {
/*
	double d = ( double( nactv ) + double( nacte ) ) / double( m );
	std::cerr << "density: nactv=" << nactv
		  << " nacte=" << nacte << " m=" << m
		  << " d=" << d << "\n";
	return d;
*/
	return ( double( nactv ) + double( nacte ) ) / double( m );
    }

    template<typename GraphType>
    void merge_or( const GraphType &, frontier & );

private:
    inline void merge_or_sparse( const GraphCSx & G, frontier & f );
    
    template<typename GraphType, typename LHSTy>
    void merge_or_ds( const GraphType &, LHSTy *, frontier & );

    template<typename GraphType, typename LHSTy, typename RHSTy>
    void merge_or_tmpl( const GraphType &, LHSTy *, RHSTy * );

    /*************************************************************
     * Vertex map and filter
     *************************************************************/
public:
    template<class F>
    frontier filter( const partitioner & part, F fn ) {
	switch( ftype ) {
	case frontier_type::ft_true:
	{
	    // To select the size, vertexFilter should consult a
	    // method in the Graph that provides the preferred frontier type.
	    // somewhat similar to determine_frontier_bytewidth...
	    // frontier fnew = dense<sizeof(VID)>( part ); // all false
	    // filter_to( part, fnew.getDenseL<sizeof(VID)>(), fn );
	    frontier fnew = create<frontier_type::ft_bool>( part ); // all false
	    filter_to( part, fnew.getDense<frontier_type::ft_bool>(), fn );
	    return fnew;
	}
	case frontier_type::ft_bool:
	{
	    frontier fnew = dense( part, nv ); // all false
	    filter_to( part, fnew.getDenseB(), fn );
	    return fnew;
	}
	case frontier_type::ft_bit:
	    assert( 0 && "NYI" );
	    break;
	case frontier_type::ft_logical1: return filter_helper<1>( part, fn );
	case frontier_type::ft_logical2: return filter_helper<2>( part, fn );
	case frontier_type::ft_logical4: return filter_helper<4>( part, fn );
	case frontier_type::ft_logical8: return filter_helper<8>( part, fn );
	case frontier_type::ft_sparse:
	{
	    frontier fnew = sparse( nv, nactv );
	    VID * sold = getSparse();
	    VID * snew = fnew.getSparse();
	    VID ne = 0;
	    for( VID i=0; i < nactv; ++i ) {
		VID v = sold[i];
		if( fn( v ) )
		    snew[ne++] = v;
	    }
	    fnew.nactv = ne;
	    return fnew;
	}
	default: UNREACHABLE_CASE_STATEMENT;
	}
    }
private:
    template<typename DNTy, class F>
	void filter_to( const partitioner & part, DNTy * dnew, F fn ) {
	switch( ftype ) {
	case frontier_type::ft_true: map_vertexL(
	    part, [&]( VID v ) { dnew[v] = as_type<DNTy>( fn(v) ); } );
	    break;
	case frontier_type::ft_bool: filter_dense( part, getDenseB(), dnew, fn ); break;
	case frontier_type::ft_bit:
	    assert( 0 && "NYI" );
	    break;
	case frontier_type::ft_logical1: filter_dense( part, getDenseL<1>(), dnew, fn ); break;
	case frontier_type::ft_logical2: filter_dense( part, getDenseL<2>(), dnew, fn ); break;
	case frontier_type::ft_logical4: filter_dense( part, getDenseL<4>(), dnew, fn ); break;
	case frontier_type::ft_logical8: filter_dense( part, getDenseL<8>(), dnew, fn ); break;
	case frontier_type::ft_sparse:
	{
	    VID * sold = getSparse();
	    for( VID i=0; i < nactv; ++i ) {
		VID v = sold[i];
		if( fn( v ) )
		    dnew[v] = as_type<DNTy>( true );
	    }
	    break;
	}
	default: UNREACHABLE_CASE_STATEMENT;
	}
    }
    template<unsigned short W, class F>
	frontier filter_helper( const partitioner & part, F fn ) {
	frontier fnew = dense<W>( part ); // all false
	filter_dense( part, getDenseL<W>(), fnew.getDenseL<W>(), fn );
	return fnew;
    }
    template<typename DOTy, typename DNTy, class F>
	void filter_dense( const partitioner & part, const DOTy * dold,
			   DNTy * dnew, F fn ) {
	map_vertexL( part, [&]( VID v ) {
		if( dold[v] ) dnew[v] = as_type<DNTy>( fn(v) ); } );
    }

    /*************************************************************
     * Output functions
     *************************************************************/
private:
    template<typename T>
    ostream & dump( ostream & os, const T * b ) {
	for( VID i=0; i < nv; ++i )
	    os << ( b[i] ? 'T' : '.' );
	os << " #e:" << nacte << " #v:" << nactv << '\n';
	return os;
    }
public:
    ostream & dump( ostream & os ) {
	switch( ftype ) {
	case frontier_type::ft_true:
	    os << "Frontier<true>";
	    break;
	case frontier_type::ft_bool:
	{
	    os << "Frontier<bool>: ";
	    dump( os, get_b().get() );
	    break;
	}
	case frontier_type::ft_bit:
	{
	    os << "Frontier<bit>: ";
	    dump( os, get_bit().get() );
	    break;
	}
	case frontier_type::ft_logical1:
	    os << "Frontier<logical1>: ";
	    dump( os, get_l<1>().get() );
	    break;
	case frontier_type::ft_logical2:
	    os << "Frontier<logical2>: ";
	    dump( os, get_l<2>().get() );
	    break;
	case frontier_type::ft_logical4:
	    os << "Frontier<logical4>: ";
	    dump( os, get_l<4>().get() );
	    break;
	case frontier_type::ft_logical8:
	    os << "Frontier<logical8>: ";
	    dump( os, get_l<8>().get() );
	    break;
	case frontier_type::ft_sparse:
	{
	    os << "Frontier<sparse>: ";
	    const VID * f = get_s();
	    for( VID i=0; i < nactv; ++i )
		os << ' ' << f[i];
	    os << '\n';
	    break;
	}
	default: UNREACHABLE_CASE_STATEMENT;
	}
	return os;
    }

    /*************************************************************
     * Query vertices - may be slow
     *************************************************************/
public:
    bool is_set( VID v ) const {
	switch( ftype ) {
	case frontier_type::ft_true: return true;
	case frontier_type::ft_bool: return get_b().get()[v];
	case frontier_type::ft_bit:
	    assert( 0 && "NYI" );
	    return true;
	case frontier_type::ft_logical1:
	    return get_l<1>().get()[v];
	case frontier_type::ft_logical2:
	    return get_l<2>().get()[v];
	case frontier_type::ft_logical4:
	    return get_l<4>().get()[v];
	case frontier_type::ft_logical8:
	    return get_l<8>().get()[v];
	case frontier_type::ft_sparse:
	{
	    const VID * f = get_s();
	    for( VID i=0; i < nactv; ++i )
		if( f[i] == v )
		    return true;
	    return false;
	}
	default: UNREACHABLE_CASE_STATEMENT;
	}
    }

    /*************************************************************
     * Internal functions
     *************************************************************/
private:
    template<typename FlagTy>
    _seq<VID> packDense( const partitioner & part, FlagTy *flags ) {
	// Note: if the sparse representation has duplicate entries (which
	//       is tolerable for certain algorithms), then the conversion of
	//       a sparse to dense frontier and back may observe that the number
	//       of active vertices and edges have been calculated "wrongly".
	//       A general solution would be to recalculate those numbers,
	//       however, this would result in needless overhead in most
	//       cases. Well-optimised code will not do such a sequence of
	//       conversions.
	// Note: Due to the presence of padding vertices, the general Ligra
	//       utilities in namespace sequence (e.g., pack) are not applicable
	//       as they do not know about the presence of padding vertices.
	int npart = part.get_num_partitions();
	VID * sums = new VID[npart+1];
	VID * ids = new VID[nactv]; // already know how many

	map_partition(
	    part, [&]( int p ) {
		      VID s = part.start_of( p );
		      VID e = part.end_of( p );
		      VID r = 0;
		      for( VID v=s; v < e; ++v )
			  if( !!flags[v] ) // uses logical<>::is_true()
			      ++r;
		      sums[p] = r;
		  } );

	VID check = 0;
	for( int p=0; p < npart; ++p ) {
	    VID tmp = sums[p];
	    sums[p] = check;
	    check += tmp;
	}
	sums[npart] = check;

	if( nactv != check )
	    std::cerr << "new packDense check=" << check << " nactv=" << nactv << "\n";
	assert( nactv == check && "nactv value mismatch with array" );

	map_partition(
	    part, [&]( int p ) {
		      VID s = part.start_of( p );
		      VID e = part.end_of( p );
		      VID pos = sums[p];
		      for( VID v=s; v < e; ++v )
			  if( !!flags[v] ) // uses logical<>::is_true()
			      ids[pos++] = v;
		      assert( pos == sums[p+1] );
		  } );
	
	delete[] sums;
	return _seq<VID>( ids, nactv );

	// _seq<VID> packed = sequence::pack( (VID*)nullptr, flags, (VID)0,
	// nv, identityF<VID>() );
	// assert( nactv == packed.n && "nactv value mismatch with array" );
	// return packed;
    }
    _seq<VID> packDenseBit( const partitioner & part,
			    unsigned char * raw_flags ) /*{
	assert( 0 && "Need update like packDense() wrt padding vertices" );
	_seq<VID> packed = sequence::pack( (VID*)nullptr, BitReader<VID>(flags),
					   (VID)0, nv, identityF<VID>() );
	assert( nactv == packed.n && "nactv value mismatch with array" );
	return packed;
	}*/{
	// Note: if the sparse representation has duplicate entries (which
	//       is tolerable for certain algorithms), then the conversion of
	//       a sparse to dense frontier and back may observe that the number
	//       of active vertices and edges have been calculated "wrongly".
	//       A general solution would be to recalculate those numbers,
	//       however, this would result in needless overhead in most
	//       cases. Well-optimised code will not do such a sequence of
	//       conversions.
	// Note: Due to the presence of padding vertices, the general Ligra
	//       utilities in namespace sequence (e.g., pack) are not applicable
	//       as they do not know about the presence of padding vertices.
	int npart = part.get_num_partitions();
	VID * sums = new VID[npart+1];
	VID * ids = new VID[nactv]; // already know how many

	BitReader<VID> flags( raw_flags );

	map_partition(
	    part, [&]( int p ) {
		      VID s = part.start_of( p );
		      VID e = part.end_of( p );
		      VID r = 0;
		      for( VID v=s; v < e; ++v )
			  if( !!flags[v] )
			      ++r;
		      sums[p] = r;
		  } );

	VID check = 0;
	for( int p=0; p < npart; ++p ) {
	    VID tmp = sums[p];
	    sums[p] = check;
	    check += tmp;
	}
	sums[npart] = check;

	if( nactv != check )
	    std::cerr << "new packDenseBit check=" << check << " nactv=" << nactv << "\n";
	assert( nactv == check && "nactv value mismatch with array" );

	map_partition(
	    part, [&]( int p ) {
		      VID s = part.start_of( p );
		      VID e = part.end_of( p );
		      VID pos = sums[p];
		      for( VID v=s; v < e; ++v )
			  if( !!flags[v] )
			      ids[pos++] = v;
		      assert( pos == sums[p+1] );
		  } );
	
	delete[] sums;
	return _seq<VID>( ids, nactv );

	// _seq<VID> packed = sequence::pack( (VID*)nullptr, flags, (VID)0,
	// nv, identityF<VID>() );
	// assert( nactv == packed.n && "nactv value mismatch with array" );
	// return packed;
    }
    _seq<VID> packDenseBit2( const partitioner & part,
			     unsigned char *raw_flags ) {
	// Note: if the sparse representation has duplicate entries (which
	//       is tolerable for certain algorithms), then the conversion of
	//       a sparse to dense frontier and back may observe that the number
	//       of active vertices and edges have been calculated "wrongly".
	//       A general solution would be to recalculate those numbers,
	//       however, this would result in needless overhead in most
	//       cases. Well-optimised code will not do such a sequence of
	//       conversions.
	// Note: Due to the presence of padding vertices, the general Ligra
	//       utilities in namespace sequence (e.g., pack) are not applicable
	//       as they do not know about the presence of padding vertices.
	int npart = part.get_num_partitions();
	VID * sums = new VID[npart+1];
	VID * ids = new VID[nactv]; // already know how many

	BitReader2<VID> flags( raw_flags );

	map_partition(
	    part, [&]( int p ) {
		      VID s = part.start_of( p );
		      VID e = part.end_of( p );
		      VID r = 0;
		      for( VID v=s; v < e; ++v )
			  if( !!flags[v] )
			      ++r;
		      sums[p] = r;
		  } );

	VID check = 0;
	for( int p=0; p < npart; ++p ) {
	    VID tmp = sums[p];
	    sums[p] = check;
	    check += tmp;
	}
	sums[npart] = check;

	if( nactv != check )
	    std::cerr << "new packDenseBit2 check=" << check << " nactv=" << nactv << "\n";
	assert( nactv == check && "nactv value mismatch with array" );

	map_partition(
	    part, [&]( int p ) {
		      VID s = part.start_of( p );
		      VID e = part.end_of( p );
		      VID pos = sums[p];
		      for( VID v=s; v < e; ++v )
			  if( !!flags[v] )
			      ids[pos++] = v;
		      assert( pos == sums[p+1] );
		  } );
	
	delete[] sums;
	return _seq<VID>( ids, nactv );

	// _seq<VID> packed = sequence::pack( (VID*)nullptr, flags, (VID)0,
	// nv, identityF<VID>() );
	// assert( nactv == packed.n && "nactv value mismatch with array" );
	// return packed;
    }

    void replace_storage( _seq<VID> sparse ) {
	clear_storage();
	get_s() = sparse.A;
	ftype = frontier_type::ft_sparse;
    }

    void clear_storage() {
	switch( ftype ) {
	case frontier_type::ft_true: break;
	case frontier_type::ft_sparse: delete[] get_s(); break;
	case frontier_type::ft_bool: get_b().del(); break;
	case frontier_type::ft_bit: get_bit().del(); break;
	case frontier_type::ft_bit2: get_bit2().del(); break;
	case frontier_type::ft_logical1: get_l<1>().del(); break;
	case frontier_type::ft_logical2: get_l<2>().del(); break;
	case frontier_type::ft_logical4: get_l<4>().del(); break;
	case frontier_type::ft_logical8: get_l<8>().del(); break;
	case frontier_type::ft_msb4: break; // nothing to do
	default: UNREACHABLE_CASE_STATEMENT;
	}
    }

    mmap_ptr<bool> & get_b() {
	return *reinterpret_cast<mmap_ptr<bool> *>( &storage[0] );
    }
    mmap_ptr<unsigned char> & get_bit() {
	return *reinterpret_cast<mmap_ptr<unsigned char> *>( &storage[0] );
    }
    mmap_ptr<unsigned char> & get_bit2() {
	return *reinterpret_cast<mmap_ptr<unsigned char> *>( &storage[0] );
    }
    VID * & get_s() {
	return *reinterpret_cast<VID **>( &storage[0] );
    }
    template<unsigned short B>
	mmap_ptr<logical<B>> & get_l() {
	return *reinterpret_cast<mmap_ptr<logical<B>> *>( &storage[0] );
    }

    mmap_ptr<bool> get_b() const {
	return *reinterpret_cast<const mmap_ptr<bool> *>( &storage[0] );
    }
    mmap_ptr<unsigned char> get_bit() const {
	return *reinterpret_cast<const mmap_ptr<unsigned char> *>( &storage[0] );
    }
    mmap_ptr<unsigned char> get_bit2() const {
	return *reinterpret_cast<const mmap_ptr<unsigned char> *>( &storage[0] );
    }
    VID * get_s() const {
	return *reinterpret_cast<VID * const *>( &storage[0] );
    }
    template<unsigned short B>
	mmap_ptr<logical<B>> get_l() const {
	return *reinterpret_cast<const mmap_ptr<logical<B>> *>( &storage[0] );
    }

private:
    frontier_type ftype; //< storage format of the frontier
    VID nv;              //< number of vertices in graph
    VID nactv;           //< number of active vertices
    EID nacte;           //< number of active edges

    //< storage for different layouts. makes it cheap on copy (no embedded
    //  virtual memory allocation). same functionality as union
    char storage[sizeof(mmap_ptr<logical<8>>)];
};

template<>
inline frontier frontier::create<frontier_type::ft_bool>( const partitioner & part ) {
    return frontier::dense( part, part.get_num_elements() );
}

template<>
inline frontier frontier::create<frontier_type::ft_unbacked>( const partitioner & part ) {
    return frontier::unbacked( part );
}

template<>
inline frontier frontier::create<frontier_type::ft_bit>( const partitioner & part ) {
    return frontier::bit( part );
}

template<>
inline frontier frontier::create<frontier_type::ft_bit2>( const partitioner & part ) {
    return frontier::bit2( part );
}

template<>
inline frontier frontier::create<frontier_type::ft_logical1>( const partitioner & part ) {
    return frontier::dense<1>( part );
}

template<>
inline frontier frontier::create<frontier_type::ft_logical2>( const partitioner & part ) {
    return frontier::dense<2>( part );
}

template<>
inline frontier frontier::create<frontier_type::ft_logical4>( const partitioner & part ) {
    return frontier::dense<4>( part );
}

template<>
inline frontier frontier::create<frontier_type::ft_logical8>( const partitioner & part ) {
    return frontier::dense<8>( part );
}

template<>
inline void frontier::toDense<bool>( const partitioner & part ) {
    toBool( part );
}

template<>
inline void frontier::toDense<logical<4>>( const partitioner & part ) {
    toLogical4( part );
}

template<>
inline void frontier::toDense<logical<8>>( const partitioner & part ) {
    toLogical8( part );
}

template<>
inline void frontier::toDense<frontier_type::ft_bool>( const partitioner & part ) {
    toBool( part );
}

template<>
inline void frontier::toDense<frontier_type::ft_bit>( const partitioner & part ) {
    toBit( part );
}

template<>
inline void frontier::toDense<frontier_type::ft_bit2>( const partitioner & part ) {
    toBit2( part );
}

template<>
inline void frontier::toDense<frontier_type::ft_logical1>( const partitioner & part ) {
    toLogical1( part );
}

template<>
inline void frontier::toDense<frontier_type::ft_logical2>( const partitioner & part ) {
    toLogical2( part );
}

template<>
inline void frontier::toDense<frontier_type::ft_logical4>( const partitioner & part ) {
    toLogical4( part );
}

template<>
inline void frontier::toDense<frontier_type::ft_logical8>( const partitioner & part ) {
    toLogical8( part );
}

template<>
inline void frontier::toDense<frontier_type::ft_msb4>( const partitioner & part ) {
    // Cannot make this converion, leave as it is
}


template<>
inline bool * frontier::getDense<bool>() const {
    return getDenseB();
}

template<>
inline logical<1> * frontier::getDense<logical<1>>() const {
    return getDenseL<1>();
}

template<>
inline logical<2> * frontier::getDense<logical<2>>() const {
    return getDenseL<2>();
}

template<>
inline logical<4> * frontier::getDense<logical<4>>() const {
    return getDenseL<4>();
}

template<>
inline logical<8> * frontier::getDense<logical<8>>() const {
    return getDenseL<8>();
}

template<>
inline bool * frontier::getDense<frontier_type::ft_bool>() const {
    return getDenseB();
}

template<>
inline unsigned char * frontier::getDense<frontier_type::ft_bit>() const {
    return getDenseBit();
}

template<>
inline unsigned char * frontier::getDense<frontier_type::ft_bit2>() const {
    return getDenseBit2();
}

template<>
inline logical<1> * frontier::getDense<frontier_type::ft_logical1>() const {
    return getDenseL<1>();
}

template<>
inline logical<2> * frontier::getDense<frontier_type::ft_logical2>() const {
    return getDenseL<2>();
}

template<>
inline logical<4> * frontier::getDense<frontier_type::ft_logical4>() const {
    return getDenseL<4>();
}

template<>
inline logical<8> * frontier::getDense<frontier_type::ft_logical8>() const {
    return getDenseL<8>();
}

template<>
inline logical<4> * frontier::getDense<frontier_type::ft_msb4>() const {
    return getDenseL<4>();
}


using partitioned_vertices = frontier;

enum frontier_mode {
    fm_all_true = 0,
    fm_reduction = 1,
    fm_calculate = 2,
    fm_N = 3
};

extern const char * frontier_mode_names[fm_N+1];

inline std::ostream & operator << ( std::ostream & os, frontier_mode fr ) {
    int ifr = (int) fr;
    if( ifr >= 0 && ifr < (int)fm_N )
	return os << frontier_mode_names[ifr];
    else
	return os << frontier_mode_names[(int)fm_N];
}

#endif // GRAPTOR_FRONTIER_H
