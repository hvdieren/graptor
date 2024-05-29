// -*- c++ -*-
#ifndef GRAPTOR_TIMER_H
#define GRAPTOR_TIMER_H

#include <time.h>

namespace graptor {

template<bool cycle_counter = false>
class timer {
public:
    static constexpr bool is_cycle_counter = cycle_counter;
    
    timer() : m_last_time( 0 ), m_total_time( 0 ), m_time_zone( { 0, 0 } ) {
	start();
    }

    double get_time_point() {
	constexpr clockid_t clk =
	    is_cycle_counter ? CLOCK_MONOTONIC_RAW : CLOCK_TAI;
	struct timespec ts;
	if( clock_gettime( clk, &ts ) != 0 ) {
	    std::cerr << "Error clock_gettime(): " << strerror(errno) << "\n";
	    exit( 1 );
	}
        return ((double) ts.tv_sec) + ((double) ts.tv_nsec) / 1'000'000'000.0;
    }

    void start() {
	m_last_time = get_time_point();
    }

    double stop() {
	next();
	return total();
    }
    
    double total() const {
	return m_total_time;
    }

    double elapsed() const {
	return get_time_point() - m_last_time;
    }

    double next() {
	double t = get_time_point();
	double delay = t - m_last_time;
	m_total_time += delay;
	m_last_time = t;
	return delay;
    }

private:
    double m_last_time;
    double m_total_time;
    struct timezone m_time_zone;
};

} // namespace graptor

#endif // GRAPTOR_TIMER_H
