// -*- C++ -*-
#include <unistd.h>
#include <stdlib.h>

#ifndef GRAPHGRIND_DRIVER_H
#define GRAPHGRIND_DRIVER_H

// This method can be overloaded if needed
template<class GraphType>
struct helper {
    static GraphType createGraph( const GraphCSx & G, commandLine & P,
				  int npart, int maxVL ) {
	return GraphType( G, npart, maxVL );
    }
};

template<>
struct helper<GraphGGVEBO> {
    static GraphGGVEBO createGraph( const GraphCSx & G, commandLine & P,
				    int npart, int maxVL ) {
	return GraphGGVEBO( G, npart );
    }
};

template<>
struct helper<GraphGG> {
    static GraphGG createGraph( const GraphCSx & G, commandLine & P,
				    int npart, int maxVL ) {
	bool balance_vertices = P.getOptionValue("-balancev");
	return GraphGG( G, npart, balance_vertices );
    }
};

template<>
struct helper<GraphVEBOSlimSell> {
    static GraphVEBOSlimSell createGraph( const GraphCSx & G, commandLine & P,
					  int npart, int maxVL ) {
	return GraphVEBOSlimSell( G, npart, maxVL );
    }
};

template<>
struct helper<GraphSlimSell> {
    static GraphSlimSell createGraph( const GraphCSx & G, commandLine & P,
				      int npart, int maxVL ) {
	return GraphSlimSell( G, npart, maxVL );
    }
};

using GraphVEBOGraptorT = GraphVEBOGraptor<GRAPTOR_MODE>;

using GraphVEBOGraptorPullDataParNotCached =
    GraphVEBOGraptor<GRAPTOR_MODE_MACRO(0,0,1)>;
using GraphVEBOGraptorPullDataParCached =
    GraphVEBOGraptor<GRAPTOR_MODE_MACRO(1,0,1)>;
using GraphVEBOGraptorPushSingleCached =
    GraphVEBOGraptor<GRAPTOR_MODE_MACRO(1,1,0)>;
using GraphVEBOGraptorPushSingleNotCached =
    GraphVEBOGraptor<GRAPTOR_MODE_MACRO(0,1,0)>;
using GraphVEBOGraptorPushDataParNotCached =
    GraphVEBOGraptor<GRAPTOR_MODE_MACRO(0,0,0)>;
using GraphVEBOGraptorPushDataParCached =
    GraphVEBOGraptor<GRAPTOR_MODE_MACRO(1,0,0)>;

template<graptor_mode_t Mode>
struct helper<GraphVEBOGraptor<Mode>> {
    static GraphVEBOGraptor<Mode>
	createGraph( const GraphCSx & G, commandLine & P,
		     int npart, int maxVL ) {
	return GraphVEBOGraptor<Mode>( G, npart, maxVL/2, maxVL, 4 );
    }
};

template<>
struct helper<GraphVEBOPartCSR> {
    static GraphVEBOPartCSR createGraph( const GraphCSx & G, commandLine & P,
				    int npart, int maxVL ) {
	return GraphVEBOPartCSR( G, npart );
    }
};

template<>
struct helper<GraphVEBOPartCCSR> {
    static GraphVEBOPartCCSR createGraph( const GraphCSx & G, commandLine & P,
					  int npart, int maxVL ) {
	return GraphVEBOPartCCSR( G, npart );
    }
};

template<>
struct helper<GraphCSR> {
    static GraphCSR createGraph( const GraphCSx & G, commandLine & P,
				 int npart, int maxVL ) {
	return GraphCSR( G, -1 );
    }
};

#ifndef GRAPHTYPE
#error "Require to pre-define GRAPHTYPE"
#endif // GRAPHTYPE

#define GT_CAT_NX(a,b) a ## b
#define GT_CAT(a,b) GT_CAT_NX(a,b)
#define gname(n) GT_CAT(Graph, n)
#define GNAME gname(GRAPHTYPE)
auto createGraph( const GraphCSx & G, commandLine & P ) {
    int npart = P.getOptionLongValue("-c", 256);   // # partitions
    int maxVL = P.getOptionLongValue("-l", 16);    // # SIMD lanes
    auto PG = helper<GNAME>::createGraph( G, P, npart, maxVL );
    PG.fragmentation();
    return PG;
}
#undef gname

template<typename GraphType>
void ShowPaddingStatistics( const GraphType & G, unsigned short VL ) {
    // Padding that would be incurred of working towards a vector length VL
    VID n = G.numVertices();
    unsigned nbuckets = ilog2( n ) + 5;
    VID * npad_dp = new VID[nbuckets];
    std::fill( npad_dp, &npad_dp[nbuckets], (VID)0 );
    VID * nv_dp = new VID[nbuckets];
    std::fill( nv_dp, &nv_dp[nbuckets], (VID)0 );
    VID * npad_sd = new VID[nbuckets];
    std::fill( npad_sd, &npad_sd[nbuckets], (VID)0 );
    VID * nv_sd = new VID[nbuckets];
    std::fill( nv_sd, &nv_sd[nbuckets], (VID)0 );

    auto B = []( VID deg ) {
	VID b;
	if( deg < 3 )
	    b = deg;
	else
	    b = ilog2( deg ) + 3;
	return b;
    };

    map_vertexL( G.get_partitioner(), [&]( VID v ) {
	    // Assuming VEBO
	    VID d = G.getOutDegree( v );
	    if( v % VL == 0 ) {
		VID b_dp = B( d );
		assert( b_dp < nbuckets );
		__sync_fetch_and_add( &nv_dp[b_dp], 1 );

		for( VID w=v+1; w < v+VL && w < n; ++w ) {
		    VID dw = G.getOutDegree( w );
		    assert( dw <= d );
		    VID pad_dp = d - dw;
		    VID b_dp = B( d );
		    assert( b_dp < nbuckets );
		    __sync_fetch_and_add( &npad_dp[b_dp], pad_dp );
		    __sync_fetch_and_add( &nv_dp[b_dp], 1 );
		}
	    }

	    VID pad_sd = d % VL;
	    if( pad_sd != 0 )
		pad_sd = VL - pad_sd;
	    VID b_sd = B( d );
	    assert( b_sd < nbuckets );
	    __sync_fetch_and_add( &npad_sd[b_sd], pad_sd );
	    __sync_fetch_and_add( &nv_sd[b_sd], 1 );
	} );

    std::cerr << "Statistics on padding for vector length VL=" << VL << "\n";
    for( VID b=0; b < nbuckets; ++b ) {
	std::cerr << "bucket=" << b
		  << " dp=" << npad_dp[b]
		  << " nv=" << nv_dp[b]
		  << " sd=" << npad_sd[b]
		  << " nv=" << nv_sd[b]
		  << '\n';
    }

    delete[] npad_sd;
    delete[] nv_sd;
    delete[] npad_dp;
    delete[] nv_dp;
}

//driver
int main(int argc, char* argv[])
{
    commandLine P(argc,argv," [-s] <inFile>");
    char* iFile = P.getArgument(0);
    bool symmetric = P.getOptionValue("-s");
    bool binary = P.getOptionValue("-b");             //Galois binary format
    // assert( !binary || ( sizeof(VID) == 4 && sizeof(EID) == 8 )
	    // && "encoding specifics of Galois binary format" );
    long rounds = P.getOptionLongValue("-rounds",3); // Usually 20 rounds
    int vlen = P.getOptionLongValue("-l", 8);    // NUMA node number
    const char * weights = P.getOptionValue("-weights"); // file with weights

    assert( binary && "driver supports only binary mode files" );

#if NUMA
    std::cerr << "driver: number of NUMA nodes: " << num_numa_node << "\n";
#endif
    
    GraphCSx G( iFile, -1, symmetric, weights );
    auto PG = createGraph( G, P );
    G.del(); // G no longer needed

    if( P.getOption( "-stats:padding" ) ) {
	ShowPaddingStatistics( PG, P.getOptionLongValue("-l", 16) );
    }
    
#if PAPI_CACHE 
    PAPI_initial();         /*PAPI Event inital*/
#endif
    std::vector<typename Benchmark<decltype(PG)>::stat> stat_buf( rounds );
    for(int r=0; r<rounds; r++)
    {
	Benchmark<decltype(PG)> problem( PG, P );

#if NUMA
	// NUMA report. Should have allocated all critical memory by now,
	// so allows to check for NUMA load balance.
	pid_t pid = getpid();
	char buf[128];
	snprintf( buf, sizeof(buf), "numastat -p %ld", (long)pid );
	int ret = system( buf );
	if( ret != 0 ) {
	    std::cerr << "execution of '" << buf << "' failed: " << (int)ret
		      << " error: " << strerror(errno) << "\n";
	}
#endif

#if PAPI_CACHE 
	PAPI_start_count();   /*start PAPI counters*/
#endif
	startTime();

	problem.run();

	nextTime("Running");
#if PAPI_CACHE 
	PAPI_stop_count();   /*stop PAPI counters*/
	PAPI_print();   /* PAPI results print*/
#endif

	// working around some code that does not restore 64-bit
	// callee-saved regs
	__asm__ __volatile__( "" : : : "r12", "r13", "r14", "r15" );

	problem.post_process( stat_buf[r] );
	problem.validate( stat_buf[r] );
    }
    Benchmark<decltype(PG)>::report( stat_buf );
    reportAvg(rounds);
    PG.del();

#if PAPI_CACHE 
    PAPI_total_print(rounds);   /* PAPI results print*/
    PAPI_end();
#endif
    //timeprint();    /* Time Details print*/
    return 0;
}

#endif // GRAPHGRIND_DRIVER_H
