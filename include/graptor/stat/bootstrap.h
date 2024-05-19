// -*- c++ -*-
#ifndef GRAPTOR_STAT_BOOTSTRAP_H
#define GRAPTOR_STAT_BOOTSTRAP_H

#include <cmath>
#include <random>
#include <algorithm>
#include <ostream>

namespace graptor {

template<typename T>
std::vector<T> get_sample(
    const std::vector<T> & x,
    size_t sample_size ) {

    static thread_local mt19937* generator = nullptr;
    if( !generator ) {
	pthread_t self = pthread_self();
	generator = new mt19937( clock() + self );
    }
    uniform_int_distribution<size_t> distribution( 0, x.size()-1 );
    std::vector<T> sample( sample_size );

    for( size_t i=0; i < sample_size; ++i ) {
	size_t item = distribution( *generator );
	assert( 0 <= item && item < x.size() );
	sample[i] = x[item];
    }

    assert( sample.size() == sample_size );
    return sample;
}

template<typename T, typename Fn>
std::vector<T> bootstrap(
    const std::vector<T> & x,
    Fn && fn,
    size_t samples,
    size_t sample_size ) {
    std::vector<T> dist;
    dist.reserve( samples );

    for( size_t i=0; i < samples; ++i ) {
	dist.push_back( fn( get_sample( x, sample_size ) ) );
    }

    return dist;
}

template<typename T, typename Fn>
std::vector<T> bootstrap_difference(
    const std::vector<T> & x,
    const std::vector<T> & y,
    Fn && fn,
    size_t samples,
    size_t sample_size ) {
    std::vector<T> dist;
    dist.reserve( samples );

    for( size_t i=0; i < samples; ++i ) {
	dist.push_back( fn( get_sample( x, sample_size ) )
			- fn( get_sample( y, sample_size ) ) );
    }

    return dist;
}

template<typename T>
double mean( const std::vector<T> & x ) {
    if( x.size() == 0 )
	return std::numeric_limits<T>::infinity();
    
    double sum = 0;
    for( auto v : x )
	sum += v;
    double mu = sum / double(x.size());
    assert( !std::isnan( mu ) && "NaN detected" );
    return mu;
}

template<typename T>
double sdev( const std::vector<T> & x ) {
    if( x.size() <= 1 )
	return std::numeric_limits<T>::infinity();
    
    double sum = 0, sumsq = 0;
    for( auto v : x ) {
	sum += v;
	sumsq += v * v;
    }
    double n = x.size();
    double sd = std::sqrt( n * ( sumsq/n - (sum/n)*(sum/n) ) / (n-1.0) );
    assert( !std::isnan( sd ) && "NaN detected" );
    return sd;
}

template<typename T>
double interpolate_percentile(
    const std::vector<T> & x,
    double pctile ) {
    double pos = pctile * double( x.size() );
    if( (size_t)pos >= x.size()-1 )  // takes care of some invalid arguments
	return x[x.size()-1];
    else if( pos == (double)(size_t)pos )
	return x[size_t(pos)];
    else {
	size_t lo = std::trunc(pos);
	double r = pos - lo;
	return r * x[lo] + (1.0 - r) * x[lo+1];
    }
}

struct percentile {
    percentile( double pctile ) : m_pctile( pctile ) { }
    template<typename T>
    double operator() ( std::vector<T> && x ) {
	std::sort( x.begin(), x.end() );
	return interpolate_percentile( x, m_pctile );
    }
private:
    double m_pctile;
};

struct confidence_interval {
    confidence_interval( double pctile ) : m_pctile( pctile ) { }
    template<typename T>
    std::pair<double,double> operator() ( std::vector<T> && x ) {
	std::sort( x.begin(), x.end() );
	double p = ( 1.0 - m_pctile ) / 2.0;
	return std::make_pair(
	    interpolate_percentile( x, p ),
	    interpolate_percentile( x, 1.0 - p ) );
    }
private:
    double m_pctile;
};

template<typename T>
double bootstrap_percentile(
    const std::vector<T> & x,
    double pctile,
    size_t samples,
    size_t sample_size ) {
    return mean( bootstrap( x, percentile( pctile ),
			    samples, sample_size ) );
}

template<typename T>
double bootstrap_mean(
    const std::vector<T> & x,
    size_t samples,
    size_t sample_size ) {
    return mean( bootstrap( x, mean<T>, samples, sample_size ) );
}

template<typename T>
double bootstrap_sdev(
    const std::vector<T> & x,
    size_t samples,
    size_t sample_size ) {
    return mean( bootstrap( x, sdev<T>, samples, sample_size ) );
}

// CI of the mean
template<typename T>
std::pair<double,double> bootstrap_confidence_interval_mean(
    const std::vector<T> & x,
    double pctile,
    size_t samples,
    size_t sample_size ) {
    return confidence_interval( pctile )
	( bootstrap( x, mean<T>, samples, sample_size ) );
}

template<typename T>
struct characterize_mean {
    characterize_mean( const std::vector<T> & x,
		       double pctile,
		       size_t samples, size_t sample_size )
	: m_x( x ), m_pctile( pctile ) {
	if( m_x.empty() ) {
	    m_mu = m_sdev = m_ci_lo = m_ci_hi
		= std::numeric_limits<double>::quiet_NaN();
	} else {
	    m_mu = bootstrap_mean( m_x, samples, sample_size );
	    m_sdev = bootstrap_sdev( m_x, samples, sample_size );
	    std::tie( m_ci_lo, m_ci_hi ) = bootstrap_confidence_interval_mean(
		m_x, pctile, samples, sample_size );
	}
    }

    double min() const {
	return m_x.empty() ? std::numeric_limits<double>::quiet_NaN()
	    : *std::min_element( m_x.begin(), m_x.end() );
    }
    double max() const {
	return m_x.empty() ? std::numeric_limits<double>::quiet_NaN()
	    : *std::max_element( m_x.begin(), m_x.end() );
    }
    
    const std::vector<T> & m_x;
    const double m_pctile;
    double m_mu, m_sdev, m_ci_lo, m_ci_hi;
};

template<typename T>
std::ostream &
operator << ( std::ostream & os, const characterize_mean<T> & m ) {
    return os << "n=" << m.m_x.size()
	      << " mu=" << m.m_mu
	      << " sdev=" << m.m_sdev
	      << " p=" << m.m_pctile << " CI mean="
	      << m.m_ci_lo << ',' << m.m_ci_hi
	      << " min=" << m.min()
	      << " max=" << m.max();
}

template<typename T>
struct characterize_mean_difference {
    characterize_mean_difference(
	const std::vector<T> & x,
	const std::vector<T> & y,
	double pctile,
	size_t samples, size_t sample_size )
	: m_x( x ), m_y( y ), m_pctile( pctile ) {
	if( m_x.empty() ) {
	    m_mu = m_sdev = m_ci_lo = m_ci_hi = m_min = m_max
		= std::numeric_limits<double>::quiet_NaN();
	} else {
	    m_dist = bootstrap_difference(
		m_x, m_y, mean<double>, samples, sample_size );
	    m_mu = mean<double>( m_dist );
	    m_sdev = sdev<double>( m_dist );
	    std::tie( m_ci_lo, m_ci_hi ) = confidence_interval( pctile )
		( bootstrap( m_dist, mean<double>, samples, sample_size ) );
	    m_min = *std::min_element( m_dist.begin(), m_dist.end() );
	    m_max = *std::max_element( m_dist.begin(), m_dist.end() );
	}
    }

    const std::vector<T> & m_x, & m_y;
    std::vector<T> m_dist;
    const double m_pctile;
    double m_mu, m_sdev, m_ci_lo, m_ci_hi, m_min, m_max;
};

template<typename T>
std::ostream &
operator << ( std::ostream & os, const characterize_mean_difference<T> & m ) {
    return os << "mean difference: x.n=" << m.m_x.size()
	      << " y.n=" << m.m_y.size()
	      << " mu=" << m.m_mu
	      << " sdev=" << m.m_sdev
	      << " p=" << m.m_pctile << " CI=" << m.m_ci_lo << ',' << m.m_ci_hi
	      << " min=" << m.m_min
	      << " max=" << m.m_max;
}


} // namespace graptor

#endif // GRAPTOR_STAT_BOOTSTRAP_H
