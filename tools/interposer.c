#define _GNU_SOURCE

#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>
#include <pthread.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <stdarg.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <omp.h>

// Avoid reading in graptor/graptor.h as it brings in much unneeded code
#include "config.h"

//note _int_fini should be called again on exit, but this is not the case...why?
#define handle_error_en(en,msg) do {			       \
	fprintf( stderr, "ERROR %s:%d ", __FILE__, __LINE__ ); \
	fprintf( stderr, "%s", msg );			       \
	exit(errno);					       \
    } while(0)

#define SOCKET_WISE 2
#define SEQ 1
#define INTLV 3
#define LESVOS 4

int init=0;
#if GRAPTOR_PARALLEL == BACKEND_cilk || GRAPTOR_PARALLEL == BACKEND_cilk_numa || GRAPTOR_PARALLEL == BACKEND_parlay
int count=0; // Cilk
#else
int count=1; //initialise count to 0 for openmp, count=1 for Cilk & stand alone scheduler
#endif

int num_socket;
int num_threads=0;
int delta_socket;
int cores_per_socket=32; // 256; // 8; // 12;
int tp=0;
cpu_set_t cpuset;
int main_thread_remapped = 0;

/* Interposing pthread_create and pthread_exit */
static int (*pthread_create_p)(pthread_t *thread, const pthread_attr_t *attr, void *(*start_routine) (void *), void *arg);
//static void (*pthread_exit_p)(void *retval) __attribute__((noreturn));

/* Constructor/initializer. Automatically called when library is loaded */
static void init_interposer() __attribute__((constructor));

static void remap_main_thread();

void set_num_threads(int num, int topology){
   num_threads = num;
   tp = topology;

}

void init_interposer() {
    const char * s = NULL;
#if GRAPTOR_PARALLEL == BACKEND_cilk || GRAPTOR_PARALLEL == BACKEND_cilk_numa
    s = getenv( "CILK_NWORKERS" );
#elif GRAPTOR_PARALLEL == BACKEND_parlay
    s = getenv( "PARLAY_NUM_THREADS" );
#elif GRAPTOR_PARALLEL == BACKEND_openmp \
    || GRAPTOR_PARALLEL == BACKEND_openmp_numa
    s = getenv( "OMP_NUM_THREADS" );
#endif

    if( s ) {
    	int w = atoi( s );
	// set_num_threads( w, SOCKET_WISE );
	// set_num_threads( w, LESVOS );
	set_num_threads( w, SEQ );
    }
}

void remap_main_thread() {
    int aff;

    if( !main_thread_remapped ) {
	// Re-map current thread to first CPU
	int cpuid=count;
	CPU_ZERO(&cpuset);
	CPU_SET(cpuid, &cpuset);

	aff = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
	if (aff != 0)
	    handle_error_en(aff, "pthread_setaffinity_np");

	aff = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
	if (aff != 0)
	    handle_error_en(aff, "pthread_getaffinity_np");
	printf( "remapped main thread to %d\n", cpuid );

	count++;
	main_thread_remapped = 1;
    }
}

/* Interposer function for pthread_create */
int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start_routine) (void *), void *arg)
{
    
    int aff;
    char *error;
    pthread_attr_t attr_new;
    int s;
    int cpuid=0;
    // check if symbol is resolved and do it if not
    if(!pthread_create_p) {
        pthread_create_p = dlsym(RTLD_NEXT, "pthread_create");
        if((error = dlerror()) != NULL) {
            fputs(error, stderr);
            exit(1);
        }
    }

#if GRAPTOR_PARALLEL == BACKEND_cilk || GRAPTOR_PARALLEL == BACKEND_cilk_numa || GRAPTOR_PARALLEL == BACKEND_parlay
    // Move the main thread to its CPU now. Only do this once.
    // Only do this when creating the first thread, because prior to that
    // the runtime may decide on the number of threads to create based on
    // the affinity mask of the main thread.
    remap_main_thread();
#endif
    
    if(tp==SOCKET_WISE){
	if(init==0){
	    num_socket=  2; // ((num_threads-1)/cores_per_socket)+1;
	    delta_socket= (num_threads+num_socket-1)/num_socket;
	    init=1;
	}
	cpuid= ((count/delta_socket)*cores_per_socket) + (count % delta_socket);
    }
    else if(tp==SEQ){
    	  cpuid= count;
    } else if(tp == INTLV) {
	if( count >= num_threads )
	    cpuid = 2 * ( count - num_threads ) + 1;
	else
	    cpuid = 2*count;
    } else if(tp==LESVOS) {
	int off = 0;
	int c = count;
    	while(c >= 4) {
	    c -= 4;
	    off += 2;
	}
	if( c >= 2 )
	    off += 14;
	cpuid = c+off;
    }
    // printf( "binding thread to %d\n", (int)cpuid);
    CPU_ZERO(&cpuset);
    CPU_SET(cpuid, &cpuset);
    s = pthread_attr_init(&attr_new);

    aff = pthread_attr_setaffinity_np(&attr_new, sizeof(cpu_set_t), &cpuset);
    if (aff != 0)
	handle_error_en(aff, "pthread_attr_setaffinity_np");


    aff = pthread_attr_getaffinity_np(&attr_new, sizeof(cpu_set_t), &cpuset);
    if (aff != 0)
	handle_error_en(aff, "pthread_attr_getaffinity_np");

    count++;
    /* Actual call to pthread_create */
    int ret;
    ret=pthread_create_p(thread,  &attr_new, start_routine, arg);
    return ret;
}
