// -*- c++ -*-

#ifndef GRAPHGRIND_VMAP_TIMING_H
#define GRAPHGRIND_VMAP_TIMING_H

#include <cstdlib>
#include <typeinfo>
#include <mutex>

template<typename Operator, typename Analysis>
__attribute__((noinline))
std::ostream & vmap_report( std::ostream & os, const char * str ) {
    static bool printed = false;
    if( printed )
	return os;

    printed = true;

    return os << str << " { VL: " << Analysis::VL
	      << ", ftype: " << Analysis::ftype << " }\n";
}

template<typename Operator, unsigned short VL>
std::ostream & vmap_report( std::ostream & os, const char * str ) {
    static bool printed = false;
    if( printed )
	return os;

    printed = true;

    return os << str << " { VL: " << VL << " }\n";
}

template<typename Operator, typename Analysis>
__attribute__((noinline))
std::ostream & vmap_report( std::ostream & os, const std::string & str ) {
    return vmap_report<Operator,Analysis>( os, str.c_str() );
}

template<typename Operator, unsigned short VL>
std::ostream & vmap_report( std::ostream & os, const std::string & str ) {
    return vmap_report<Operator,VL>( os, str.c_str() );
}

class vmap_time_queue {
public:
    vmap_time_queue( const char * name ) : m_name( name ) { }
	
    void record( double delay ) {
	m_list.push_back( delay );
    }

    std::ostream & report( std::ostream & os ) {
	size_t n = 0;
	double sum = 0;
	for( double d : m_list ) {
	    ++n;
	    sum += d;
	}

	os << "vmap timing " << m_name
	   << " runs=" << n
	   << " total=" << sum
	   << " avg=" << sum/double(n)
	   << " {";

	bool first = true;
	for( double d : m_list ) {
	    if( first ) {
		first = false;
		os << d;
	    } else
		os << ", " << d;
	}

	return os << "}\n";
    }

private:
    const char * m_name;
    std::vector<double> m_list;
};

static std::vector<vmap_time_queue *> vmap_time_queue_list;
static bool vmap_handler_registered = false;

static void vmap_time_handler() {
    std::cout << "** Vertexmap timings - start **\n";
    for( vmap_time_queue * q : vmap_time_queue_list ) {
	q->report( std::cout );
	delete q;
    }
    std::cout << "** Vertexmap timings - end **\n";
}

template<typename Operator>
void vmap_record_time( double delay ) {
    static vmap_time_queue * timings = nullptr;
    static std::mutex timings_lock;

    const std::lock_guard<std::mutex> lock( timings_lock );
    
    if( timings == nullptr ) {
	if( !vmap_handler_registered ) {
	    atexit( &vmap_time_handler );
	    vmap_handler_registered = true;
	}
	timings = new vmap_time_queue( typeid(Operator).name() ); //<Operator>;
	vmap_time_queue_list.push_back( timings );
    }

    timings->record( delay );
}

#endif // GRAPHGRIND_VMAP_TIMING_H
