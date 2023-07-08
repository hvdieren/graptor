// -*- c++ -*-
#ifndef GRAPTOR_STAT_TIMING_H
#define GRAPTOR_STAT_TIMING_H

#include <algorithm>

#include "graptor/stat/bootstrap.h"

namespace graptor {

//! Summary timing information
struct scalar_timing {
    size_t cnt;
    double tm;

    scalar_timing() : cnt( 0 ), tm( 0 ) { }

    void add_sample( double t ) {
	cnt++;
	tm += t;
    }
};

std::ostream &
operator << ( std::ostream & os, const scalar_timing & tm ) {
    return os << tm.tm << ' ' << tm.cnt << ' ' << (tm.tm / double(tm.cnt));
}

//! Distribution of timing information
struct distribution_timing {
    std::vector<double> samples;

    void add_sample( double t ) {
	samples.push_back( t );
    }

    characterize_mean<double>
    characterize( double pctile, size_t nsamples, size_t sample_size ) const {
	return characterize_mean<double>(
	    samples, pctile, nsamples, sample_size );
    }
};

} // namespace graptor

#endif // GRAPTOR_STAT_TIMING_H
