#include "graptor/graptor.h"
#include "graptor/dsl/vertexmap.h"
#include "graptor/api.h"
#include "check.h"
#include "unique.h"

enum variable_name {
    var_a = 0,
    var_b = 1
};

template <class GraphType>
class Test1 {
public:
    Test1( GraphType & _GA, commandLine & P ) : GA( _GA ), info_buf( 60 ) {
	itimes = P.getOption( "-itimes" );
	debug = P.getOption( "-debug" );
	calculate_active = P.getOption( "-cactive" );
    }
    ~Test1() {
	m_a.del();
	m_b.del();
    }

    struct info {
	double delay;
	float density;
	float active;
	EID nacte;
	VID nactv;

	void dump( int i ) {
	    std::cerr << "Iteration " << i << ": " << delay
		      << " density: " << density
		      << " nacte: " << nacte
		      << " nactv: " << nactv
		      << " active: " << active << "\n";
	}
    };

    struct stat { };

    void run() {
	const partitioner &part = GA.get_partitioner();
	VID n = GA.numVertices();
	EID m = GA.numEdges();

	m_a.allocate( numa_allocation_partitioned( part ) );
	m_b.allocate( numa_allocation_partitioned( part ) );

	expr::array_ro<VID,VID,var_a> a( m_a );
	expr::array_ro<VID,VID,var_b> b( m_b );

	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) { return a[v] = v; } )
	    .vertex_map( [&]( auto v ) { return b[v] = expr::iif( (v&expr::constant_val_one(v)) != expr::zero_val(v), expr::allones_val(v), v ); } )
	    .materialize();

	// Create initial frontier
	frontier F = frontier::all_true( n, m );

	iter = 0;

	timer tm_iter;
	tm_iter.start();

	std::cout << "a:";
	for( VID v=0; v < n; ++v )
	    std::cout << ' ' << m_a[v];
	std::cout << "\n";

	std::cout << "b:";
	for( VID v=0; v < n; ++v )
	    std::cout << ' ' << m_b[v];
	std::cout << "\n";

	std::cout << "a<b:";
	for( VID v=0; v < n; ++v )
	    std::cout << ( m_a[v] < m_b[v] ? 'T' : '.' );
	std::cout << "\n";

	while( !F.isEmpty() && iter < 3 ) {  // iterate until IDs converge
	    // Propagate labels
	    frontier output;

	    make_lazy_executor( part )
		.vertex_filter(
		    GA, 	 	 	// graph
		    F,  	  	// enable if vertex in frontier
		    output,  	// record new frontier
		    [&]( auto v ) {
			return a[v] < b[v];
		    } )
		.materialize();

	    // bool * f = output.getDenseB();
	    logical<4> * f = output.template getDense<frontier_type::ft_logical4>();
	    std::cout << "out:";
	    VID na = 0;
	    for( VID v=0; v < n; ++v ) {
		std::cout << ( f[v] ? 'T' : '.' );
		if( f[v] ) ++na;
	    }
	    std::cout << " nactv: " << output.nActiveVertices() << " na: " << na << "\n";
	    assert( output.nActiveVertices() == na );

	    if( itimes ) {
		VID active = 0;
		info_buf.resize( iter+1 );
		info_buf[iter].density = F.density( GA.numEdges() );
		info_buf[iter].nacte = F.nActiveEdges();
		info_buf[iter].nactv = F.nActiveVertices();
		info_buf[iter].active = float(active)/float(n);
		info_buf[iter].delay = tm_iter.next();
		if( debug )
		    info_buf[iter].dump( iter );
	    }

	    // Cleanup old frontier
	    F.del();
	    F = output;

	    iter++;
	}
    }

    void post_process( stat & stat_buf ) {
	if( itimes ) {
	    for( int i=0; i < iter; ++i )
		info_buf[i].dump( i );
	}
    }

    static void report( const std::vector<stat> & stat_buf ) { }

    void validate( stat & stat_buf ) { }

private:
    const GraphType & GA;
    bool itimes, debug, calculate_active;
    int iter;
    mmap_ptr<VID> m_a, m_b;
    std::vector<info> info_buf;
};

template <class GraphType>
using Benchmark = Test1<GraphType>;

#include "driver.C"
