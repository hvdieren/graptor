#ifndef GRAPTOR_LEGACY_PAPI_CODE_H
#define GRAPTOR_LEGACY_PAPI_CODE_H 1

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <papi.h>

#define NUM_EVENTS 4

// Number of Cilk threads in use
int n_threads;

// To control initialisation on each thread
bool init = false;
bool *init_start;

// Event codes
int event_codes[NUM_EVENTS];

// Per-thread event set
int *EventSet;

// Description of counters
struct counter_desc {
    const char * name;
    const char * friendly;
};

// Counters as recorded in last call to stop_workers
long long **values;
// Running totals of counters
long long totals[NUM_EVENTS];

#if PAPI_KNL
char local_DRAM[]= "PAPI_L2_TCM";
// char remote_DRAM[]="OFFCORE_RESPONSE_1:ANY_READ:DDR:u=1:k=1";
// char remote_DRAM[]="PAPI_BR_MSP";
// char remote_DRAM[]="perf::PERF_COUNT_HW_CACHE_DTLB:MISS:u=1";
char remote_DRAM[]= "PAPI_L1_LDM"; // "PAPI_TLB_DM";
// char remote_DRAM[]= "PAPI_L3_DCA";
char ins_count[]="PAPI_TOT_INS";
// char BR_MIS[]="PAPI_BR_MSP";
// char TLB[]="PAPI_VEC_SP"; // "PAPI_BR_MSP";
// char TLB[]="PAPI_TLB_DM";
// char TLB[]="PAPI_L1_DCM";
char TLB[]="PAPI_L3_TCM";

counter_desc event_desc[NUM_EVENTS] = {
    { ins_count, "Instruction Count" },
    { local_DRAM, "L2 misses" }, // "Local DRAM" },
    { remote_DRAM, "L3 misses" }, // "Remote DRAM" },
    { TLB, "TLB misses" }
};

#elif PAPI_CTL
char ins_count[]="PAPI_TOT_INS";
char local_DRAM[]= "PAPI_BR_MSP";
// char remote_DRAM[]= "CYCLE_ACTIVITY:CYCLES_NO_EXECUTE";
char remote_DRAM[]= "PAPI_BR_CN";
char TLB[]= "PAPI_BR_INS";

counter_desc event_desc[NUM_EVENTS] = {
    { ins_count, "Instruction Count" },
    { local_DRAM, "Branch mispredicts" },
    { remote_DRAM, "Conditional branch instructions" },
    { TLB, "Total branch instructions" }
};
#elif PAPI_PRF
char ins_count[]="PAPI_TOT_INS";
char local_DRAM[]= "L2_RQSTS.PF_MISS";
// char remote_DRAM[]= "MEM_LOAD_RETIRED.FB_HIT";
char remote_DRAM[]="L2_RQSTS.DEMAND_DATA_RD_MISS";
char TLB[]="L2_RQSTS.ALL_DEMAND_MISS";

counter_desc event_desc[NUM_EVENTS] = {
    { ins_count, "Instruction Count" },
    { local_DRAM, "L2 misses" }, // "Local DRAM" },
    { remote_DRAM, "L3 misses" }, // "Remote DRAM" },
    { TLB, "TLB misses" }
};
#elif PAPI_STALL
char ins_count[]="PAPI_TOT_CYC";
// Ivy Bridge
// char local_DRAM[]= "CYCLE_ACTIVITY.STALL_L1D_PENDING"; // "PAPI_MEM_SCY";
// char remote_DRAM[]="CYCLE_ACTIVITY.STALL_L2_PENDING"; // "PAPI_MEM_RCY";
// char TLB[]="CYCLE_ACTIVITY.STALL_LDM_PENDING"; // "PAPI_MEM_WCY";

// SkyLake
char local_DRAM[]= "PAPI_STL_ICY"; // "PAPI_MEM_SCY";
char remote_DRAM[]="PAPI_RES_STL"; // "PAPI_MEM_RCY";
char TLB[]="PAPI_STL_CCY"; // "PAPI_MEM_WCY";

counter_desc event_desc[NUM_EVENTS] = {
    { ins_count, "Cycle Count" },
    { local_DRAM, "Cycles with no instruction issue" },
    { remote_DRAM, "Cycles stalled on any resource" },
    { TLB, "Cycles with no instruction completion" }
};
#elif PAPI_PEND1
char ins_count[]="PAPI_TOT_CYC";
// Ivy Bridge
// char local_DRAM[]= "CYCLE_ACTIVITY.STALL_L1D_PENDING"; // "PAPI_MEM_SCY";
// char remote_DRAM[]="CYCLE_ACTIVITY.STALL_L2_PENDING"; // "PAPI_MEM_RCY";
// char TLB[]="CYCLE_ACTIVITY.STALL_LDM_PENDING"; // "PAPI_MEM_WCY";

// SkyLake
char local_DRAM[]= "L1D_PEND_MISS.PENDING";
char remote_DRAM[]="L1D_PEND_MISS.PENDING_CYCLES";
char TLB[]="PAPI_STL_CCY";

counter_desc event_desc[NUM_EVENTS] = {
    { ins_count, "Cycle Count" },
    { local_DRAM, "L1D pending misses" },
    { remote_DRAM, "L1 pending miss cycles" },
    { TLB, "Cycles with no instruction completion" }
};
#elif PAPI_SKYLAKE
char local_DRAM[]= "PAPI_L2_TCM";
char remote_DRAM[]="PAPI_L3_TCM";
char ins_count[]="PAPI_TOT_INS"; // INSTRUCTION_RETIRED:u=1:k=1";
// char BR_MIS[]="MISPREDICTED_BRANCH_RETIRED:u=1:k=1";
char TLB[]="perf::PERF_COUNT_HW_CACHE_DTLB:MISS";

counter_desc event_desc[NUM_EVENTS] = {
    { ins_count, "Instruction Count" },
    { local_DRAM, "L2 misses" }, // "Local DRAM" },
    { remote_DRAM, "L3 misses" }, // "Remote DRAM" },
    { TLB, "TLB misses" }
};
#else
char local_DRAM[]= "OFFCORE_RESPONSE_0:ANY_REQUEST:LLC_MISS_LOCAL:SNP_NONE:SNP_NOT_NEEDED:SNP_MISS:SNP_NO_FWD:u=1:k=1";
// char local_DRAM[]= "OFFCORE_RESPONSE_0:ANY_REQUEST:LLC_MISS_LOCAL:SNP_ANY:u=1:k=1";
// char local_DRAM[]= "MEM_LOAD_UOPS_LLC_MISS_RETIRED:LOCAL_DRAM:u=1:k=1";
char remote_DRAM[]="OFFCORE_RESPONSE_1:ANY_REQUEST:LLC_MISS_REMOTE:SNP_NONE:SNP_NOT_NEEDED:SNP_MISS:SNP_NO_FWD:u=1:k=1";
// char remote_DRAM[]="MEM_LOAD_UOPS_LLC_MISS_RETIRED:REMOTE_DRAM:u=1:k=1";
char ins_count[]="INSTRUCTION_RETIRED:u=1:k=1";
char BR_MIS[]="MISPREDICTED_BRANCH_RETIRED:u=1:k=1";
char TLB[]="perf::PERF_COUNT_HW_CACHE_DTLB:MISS";

counter_desc event_desc[NUM_EVENTS] = {
    { ins_count, "Instruction Count" },
    { local_DRAM, "L2 misses" }, // "Local DRAM" },
    { remote_DRAM, "L3 misses" }, // "Remote DRAM" },
    { TLB, "TLB misses" }
};
#endif

#define START 1
#define STOP 2

/*----------------------------------------------------------------------*
 * Helper function to distribute work over all Cilk worker threads.
 *----------------------------------------------------------------------*/
void on_all_workers_help( size_t i, size_t n, volatile bool * flags, int stage )
{
    int retval;

    if( n > 1 && i < n-1 )
	cilk_spawn on_all_workers_help( i+1, n, flags, stage);

    int id = __cilkrts_get_worker_number();
    if( id < 0 || id >= n )
	std::cerr << "PAPI/all workers: ID out of bounds: id=" << id
		  << " n=" << n << "\n";
    
    if( n > 1 ) {
        if( i == n-2 ) {
            for( int j=0; j<n; j++ )
                flags[j] = true;
        } else {
            while( !flags[i] ); // busy wait
        }
    }

    // Start PAPI counters
    if( stage == START ) {
        if( !init_start[id] ) {
            init_start[id] = true;

	    if( (retval = PAPI_create_eventset(&EventSet[id])) != PAPI_OK ) {
		std::cerr << "PAPI error: create eventset: ret=" << retval
			  << " id=" << id << "\n";
		exit( 1 );
	    }

	    if( (retval = PAPI_add_events(
		     EventSet[id], event_codes, NUM_EVENTS)) != PAPI_OK ) {
		std::cerr << "PAPI error: add events: ret=" << retval
			  << " id=" << id << "\n";
		// exit( 1 );
	    }
	}

	if( (retval = PAPI_start(EventSet[id])) != PAPI_OK ) {
	    std::cerr << "PAPI error: start events: ret=" << retval
		      << " id=" << id << "\n";
	    exit( 1 );
	}
    }

    // Stop PAPI counters
    if( stage == STOP ) {
        if( (retval = PAPI_stop( EventSet[id], values[id] )) != PAPI_OK ) {
	    std::cerr << "PAPI error: stop events: ret=" << retval
		      << " id=" << id << "\n";
	    exit( 1 );
	}
    }

    cilk_sync;
}

/*
 * start_on_all_workers
 * Start PAPI counters on all worker threads
 */
void start_on_all_workers()
{
    int n = __cilkrts_get_nworkers();
    n_threads = n;
    volatile bool flags[n];
    for( int i=0; i < n; ++i )
        flags[i] = false;

    // Once-off initialisation
    if( !init ){
       init = true;
       init_start = new bool[n];
       EventSet = new int[n];
       values = new long long*[n];
       for( int i=0; i < n; ++i ) {
           EventSet[i] = PAPI_NULL;
           values[i] = new long long[NUM_EVENTS];
	   std::fill( &values[i][0], &values[i][NUM_EVENTS], (long long)0 );
           init_start[i] = false;
        }
    }

    // Let all workers start PAPI counters.
    // Also initialise on first call.
    on_all_workers_help( 0, n, flags, START );
}

/*
 * stop_on_all_workers
 * Stop PAPI counters on all worker threads
 */
void stop_on_all_workers()
{
    int n = __cilkrts_get_nworkers();
    volatile bool flags_stop[n];
    for( int i=0; i < n; ++i )
        flags_stop[i] = false;
    on_all_workers_help( 0, n, flags_stop, STOP );
}

/*----------------------------------------------------------------------* 
 * Public interface 	 	 	 	 	 	 	*
 *----------------------------------------------------------------------*/
static __attribute__((noinline)) void PAPI_start_count()
{
    start_on_all_workers( );
}

static __attribute__((noinline)) void PAPI_stop_count()
{
    stop_on_all_workers( );
}

/*----------------------------------------------------------------------* 
 * PAPI initialisation and counter reading 	 	 	 	*
 *----------------------------------------------------------------------*/

/*
 * PAPI_initial:
 * Initial method to initialize the counters
 */
static __attribute__((noinline)) void PAPI_initial()
{
    int retval;

    // PAPI library initialisations
    if( (retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT ) {
	std::cerr << "PAPI error: library_init: ret=" << retval
		  << " version=" << PAPI_VER_CURRENT << "\n";
        exit(1);
    }
    if( (retval = PAPI_thread_init(pthread_self)) != PAPI_OK ) {
	std::cerr << "PAPI error: thread_init: ret=" << retval << "\n";
        exit(1);
    }

    // Map PAPI event names to hardware event codes
    for( int i=0; i < NUM_EVENTS; ++i ) {
	if( (retval = PAPI_event_name_to_code(
		 (char *)event_desc[i].name, &event_codes[i])) != PAPI_OK ) {
	    std::cerr << "PAPI error: event name->code: ret=" << retval
		      << " event[" << i << "]: '" << event_desc[i].name
		      << "' " << event_desc[i].friendly
		      << "\n";
	    exit(1);
	}
	std::cout << "PAPI: event codes: " << event_desc[i].friendly << ": "
		  << event_desc[i].name << ": " << event_codes[i] << "\n";
    }
}

/*
 * PAPI_print:
 * Called once ofter stopping counters.
 */
static __attribute__((noinline)) void PAPI_print()
{
    long long sum_values[NUM_EVENTS];

    std::fill( &sum_values[0], &sum_values[NUM_EVENTS], (long long)0 );

    // Sum up values of counters across threads and reset counters
    for( int k=0; k < n_threads; k++ )
	for( int i=0; i < NUM_EVENTS; ++i ) {
	    sum_values[i] += values[k][i];
	    values[k][i] = 0;
	}
	    
    // Add values up to running totals
    for( int i=0; i < NUM_EVENTS; ++i )
	totals[i] += sum_values[i];
}

static __attribute__((noinline)) void PAPI_total_print( int rounds ){
    double avg[NUM_EVENTS];

    for( int i=0; i < NUM_EVENTS; ++i )
	avg[i] = (double)totals[i] / (double)rounds;

    for( int i=0; i < NUM_EVENTS; ++i ) {
	double pki = avg[i] / avg[0] * 1000;
	std::cout << event_desc[i].friendly
		  << ' ' << avg[i] << ' ' << pki << "\n";
    }

    std::fill( &totals[0], &totals[NUM_EVENTS], (long long)0 );
}

static __attribute__((noinline)) void PAPI_total_reset() {
    std::fill( &totals[0], &totals[NUM_EVENTS], (long long)0 );
}

static __attribute__((noinline)) void PAPI_end(){
    delete[] init_start;
    delete[] EventSet;
    for( int k=0; k < n_threads; ++k )
	delete[] values[k];
    delete[] values;
}

#endif // GRAPTOR_LEGACY_PAPI_CODE_H
