// -*- c++ -*-
#ifndef GRAPTOR_STAT_WELFORD_H
#define GRAPTOR_STAT_WELFORD_H

#include <cmath>
#include <numbers>

namespace graptor {

/***********************************************************************
 * Auxiliary for printing gnuplot formulas
 ***********************************************************************/
template<size_t k, size_t n>
void print_attributes( std::ostream & os ) {
    if constexpr ( k > 0 )
	os << ',';
	
    if constexpr ( k < n ) {
	os << 'x' << k;
	if constexpr ( k+1 < n )
	    print_attributes<k+1,n>( os );
    }
}

/***********************************************************************
 * Mean estimator (no variance)
 ***********************************************************************/
template<typename T, typename S>
class mean_estimator {
public:
    using type = T;
    using stat_type = S;

    mean_estimator()
	: m_mean( 0 ), m_samples( 0 ) { }

    void update( type x ) {
	++m_samples;
	if( m_samples <= 1 ) [[unlikely]] {
	    m_mean = x;
	} else {
	    m_mean += ( x - m_mean ) / type( m_samples );
	}
    }

    type get_mean() const {
	return m_mean; // / type( m_samples );
    }

    stat_type get_samples() const {
	return m_samples;
    }

private:
    type m_mean;
    stat_type m_samples;
};


/***********************************************************************
 * Robust variance estimator
 ***********************************************************************/
template<typename T, typename S>
class variance_estimator_no_samples {
public:
    using type = T;
    using stat_type = S;

    static constexpr type normal_constant =
	std::sqrt( type(2) * std::numbers::pi_v<type> );
    
    variance_estimator_no_samples()
	: m_mean( 0 ), m_M2( 0 ) { }

    void update( type x, stat_type samples ) {
	if( samples <= 1 ) [[unlikely]] {
	    m_mean = x;
	    m_M2 = 0;
	} else {
	    type mu = m_mean + ( x - m_mean ) / type( samples );
	    m_M2 += ( x - m_mean ) * ( x - mu );
	    m_mean = mu;
	}
    }

    variance_estimator_no_samples
    plus_equals( const variance_estimator_no_samples<T,S> & r,
		 stat_type lsamples, stat_type rsamples ) {
	type nl = lsamples;
	type nr = rsamples;
	stat_type n = lsamples + rsamples;
	// If no samples, then current contents are appropriate.
	if( n > 0 ) {
	    type m = ( nl * m_mean + nr * r.m_mean ) / type( n );
	    type delta = r.m_mean - m_mean;
	    type s = m_M2 + r.m_M2 + delta * delta * nl * nr / type( n );
	    m_mean = m;
	    m_M2 = s;
	}
	return *this;
    }

    variance_estimator_no_samples
    minus_equals( const variance_estimator_no_samples<T,S> & r,
		  stat_type lsamples, stat_type rsamples ) {
	type nl = lsamples;
	type nr = rsamples;
	stat_type n = lsamples - rsamples;
	type m = ( nl * m_mean - nr * r.m_mean ) / type( n );
	type delta = r.m_mean - m_mean;
	type s = m_M2 - r.m_M2 + delta * delta * nl * nr / type( n );
	m_mean = m;
	m_M2 = s;
	return *this;
    }

    type get_mean( stat_type samples ) const {
	return m_mean;
    }
    type get_stdev( stat_type samples ) const {
	return std::sqrt( get_variance( samples ) );
    }
    type get_variance( stat_type samples ) const {
	return samples > 1 ? m_M2 / type( samples - 1 ) : type(0);
    }

    stat_type estimate( const variance_estimator_no_samples & s,
			type rmin, type rmax, stat_type samples ) {
	// estimate statistics for the range rmin to rmax in distribution s
	// samples: #samples in s
	type pmin = s.probability_density( rmin, samples );
	type pmax = s.probability_density( rmax, samples );
	auto smin = s.split_ltge( rmin, samples );
	auto smax = s.split_ltge( rmax, samples );
	type rsamples = smax.first - smin.first;
	type smean = s.get_mean( samples );
	type sigma = s.get_stdev( samples );
	type alpha = ( smean - rmin ) / sigma;
	type beta = ( rmax - smean ) / sigma;
	m_mean = smean - sigma * ( pmax - pmin )
	    / ( smax.first - smin.first );
	type mean = m_mean / rsamples;
	m_M2 = rsamples * sigma * sigma
	    * ( type(1)
		- ( beta * pmax - alpha * pmin ) / ( smax.first - smin.first )
		- mean * mean );
	assert( !std::isnan( m_M2 ) );
	return rsamples;
    }
    stat_type estimate_lt( const variance_estimator_no_samples & s,
			   type r, stat_type samples ) {
	// estimate statistics for the range -inf to r in distribution s
	// samples: #samples in s
	// type pmin = 0;
	type pmax = s.probability_density( r, samples );
	// auto smin = std::make_pair( type(0), type(0) );
	auto smax = s.split_ltge( r, samples );
	type rsamples = smax.first;
	type smean = s.get_mean( samples );
	type sigma = s.get_stdev( samples );
	type beta = ( r - smean ) / sigma;
	m_mean = smean - sigma * pmax / smax.first;
	type mean = m_mean / rsamples;
	m_M2 = rsamples * sigma * sigma
	    * ( type(1) - beta * pmax / smax.first - mean * mean );
	assert( !std::isnan( m_M2 ) );
	return rsamples;
    }
    stat_type estimate_ge( const variance_estimator_no_samples & s,
			   type r, stat_type samples ) {
	// estimate statistics for the range rmax to +inf in distribution s
	// samples: #samples in s
	type pmin = s.probability_density( r, samples );
	// type pmax = type(1);
	auto smin = s.split_ltge( r, samples );
	// auto smax = std::make_pair( type(samples), type(samples) );
	type rsamples = type(samples) - smin.first;
	type smean = s.get_mean( samples );
	type sigma = s.get_stdev( samples );
	type alpha = ( smean - r ) / sigma;
	m_mean = smean - sigma * pmin / ( type(samples) - smin.first );
	type mean = m_mean / rsamples;
	m_M2 = rsamples * sigma * sigma
	    * ( type(1) + alpha * pmin / ( type(samples) - smin.first )
		- mean * mean );
	assert( !std::isnan( m_M2 ) );
	return rsamples;
    }

    type normalize( type x, stat_type samples ) const {
	return ( x - get_mean( samples ) ) / get_stdev( samples );
    }
    type denormalize( type x, stat_type samples ) const {
	return get_mean( samples ) + x * get_stdev( samples );
    }

    type probability_density( type x, stat_type samples ) const {
	if( samples > 0 ) {
	    type sd = get_stdev( samples );
	    type mean = get_mean( samples );
	    if( sd > type(0) ) {
		type diff = x - mean;
		return ( type(1) / ( normal_constant * sd ) )
		    * std::exp( - ( diff * diff / ( type(2) * sd * sd ) ) );
	    }
	    if( x == mean )
		return type(1);
	}
	return type(0);
    }

    std::pair<type,type> split_ltge( type split, stat_type samples ) const {
	type eqw = probability_density( split, samples ) * type( samples );
	type sd = get_stdev( samples );
	type mean = get_mean( samples );
	type ltw;
	if( sd > type(0) )
	    ltw = normal_probability( ( split - mean ) / sd )
		* type( samples ) - eqw;
	else
	    ltw = split < mean ? type( samples ) - eqw : type(0);
	type gtw = std::max( type(0), type( samples ) - eqw - ltw );
	return std::make_pair( ltw, gtw + eqw );
    }

    static type normal_probability( type a ) {
	static constexpr type SQRTH = 7.07106781186547524401E-1;
	type x = a * SQRTH;
	type z = std::abs( x );
	if( z < SQRTH )
	    return 0.5 + 0.5 * std::erf( x );
	else {
	    type y = 0.5 * std::erfc( z );
	    return x > type(0) ? type() - y : y;
	}
    }

private:
    type m_mean, m_M2;
};

template<typename T, typename S>
class variance_estimator
    : public variance_estimator_no_samples<T,S> {
public:
    using type = T;
    using stat_type = S;
    using parent_type = variance_estimator_no_samples<T,S>;
    
    variance_estimator()
	: m_samples( 0 ) { }

    void update( type x ) {
	// It is important that samples is incremented before updating
	// mean and standard deviation, such that m_samples is 1 for the
	// first data point, 2 for the second, and so on.
	parent_type::update( x, ++m_samples );
    }

    stat_type get_samples() const {
	return m_samples;
    }
    type get_mean() const {
	return parent_type::get_mean( m_samples );
    }
    type get_stdev() const {
	return parent_type::get_stdev( m_samples );
    }
    type get_variance() const {
	return parent_type::get_variance( m_samples );
    }

    void estimate( const variance_estimator & s,
		   type rmin, type rmax ) {
	m_samples = parent_type::estimate( s, rmin, rmax, s.get_samples() );
    }
    void estimate_lt( const variance_estimator & s, type r ) {
	m_samples = parent_type::estimate_lt( s, r, s.get_samples() );
    }
    void estimate_ge( const variance_estimator & s, type r ) {
	m_samples = parent_type::estimate_ge( s, r, s.get_samples() );
    }

    variance_estimator
    operator += ( const variance_estimator<T,S> & r ) {
	parent_type::plus_equals( r, m_samples, r.m_samples );
	m_samples += r.m_samples;
	return *this;
    }

/*
    variance_estimator
    operator -= ( const variance_estimator<T,S> & r ) {
	parent_type::minus_equals( r, m_samples, r.m_samples );
	m_samples -= r.m_samples;
	return *this;
    }
*/

    void show( std::ostream & os, size_t indent, size_t bin ) const {
	os << std::string( indent*2, ' ' )
	   << "bin " << bin
	   << " numerical_stats { ve: mean=" << get_mean()
	   << " var=" << get_variance()
	   << " #" << get_samples()
	   << " }\n";
    }

    std::pair<type,type> split_ltge( type split ) {
	return parent_type::split_ltge( split, m_samples );
    }

private:
    stat_type m_samples;
};

template<typename type, typename stat_type>
class descriptive_statistics : public variance_estimator<type,stat_type> {
public:
    using parent_type = variance_estimator<type,stat_type>;
    
    descriptive_statistics()
	: m_min( std::numeric_limits<type>::max() ),
	  m_max( std::numeric_limits<type>::lowest() ) { }

    void update( type x ) {
	parent_type::update( x );
	m_min = std::min( m_min, x );
	m_max = std::max( m_max, x );
    }

    stat_type get_samples() const {
	return parent_type::get_samples();
    }
    type get_mean() const {
	return parent_type::get_mean();
    }
    type get_stdev() const {
	return parent_type::get_stdev();
    }
    type get_variance() const {
	return parent_type::get_variance();
    }
    type get_min() const { return m_min; }
    type get_max() const { return m_max; }

    void show( std::ostream & os ) const {
	os << "mean=" << get_mean()
	   << " var=" << get_variance()
	   << " #" << get_samples()
	   << " min=" << get_min()
	   << " max=" << get_max();
    }

private:
    type m_min;		//!< The smallest sample observed
    type m_max;		//!< The largest sample observed
};


template<typename... T, typename S>
class variance_estimator<std::tuple<T...>,S> {
public:
    using type = std::tuple<T...>;
    using stat_type = S;
    using component_type =
	std::tuple<variance_estimator_no_samples<T,S> ...>;

    static constexpr size_t num_dimensions = sizeof...(T);
    
    variance_estimator() : m_samples( 0 ) { }

    void update( const type & x ) {
	for_each( m_components, x,
		  [&]( auto & c ) { c.update( m_samples ); } );
	m_samples++;
    }

    type normalize( const type & x ) const {
	return normalize_impl(
	    x, std::make_index_sequence<num_dimensions>() );
    }
    type denormalize( const type & x ) const {
	return denormalize_impl(
	    x, std::make_index_sequence<num_dimensions>() );
    }

/*
    std::tuple<stat_type> get_mean() const {
	return std::tuple<stat_type>( m_components.get_mean( m_samples ) ... );
    }
    std::tuple<stat_type> get_stdev() const {
	return parent_type::get_stdev( m_samples );
    }
    stat_type get_variance() const {
	return parent_type::get_variance( m_samples );
    }

    variance_estimator
    operator += ( const variance_estimator<T,S> & r ) {
	parent_type::plus_equals( r, m_samples, r.m_samples );
	m_samples += r.m_samples;
	return *this;
    }
*/

private:
    template<std::size_t... I>
    type normalize( const type & x, std::index_sequence<I...> ) const {
	return std::make_tuple(
	    std::get<I>( m_components ).normalize(
		std::get<I>( x ), m_samples ) ... );
    }
    template<std::size_t... I>
    type denormalize( const type & x, std::index_sequence<I...> ) const {
	return std::make_tuple(
	    std::get<I>( m_components ).denormalize(
		std::get<I>( x ), m_samples ) ... );
    }

private:
    component_type m_components;
    size_t m_samples;
};

/***********************************************************************
 * Robust co-variance estimator
 * Welford's
 * https://dl.acm.org/doi/10.1145/3221269.3223036
 ***********************************************************************/
template<typename T, typename S>
class covariance_estimator_no_samples {
public:
    using type = T;
    using stat_type = S;

    static constexpr type normal_constant =
	std::sqrt( type(2) * std::numbers::pi_v<type> );
    
    covariance_estimator_no_samples()
	: m_mean_x( 0 ), m_M2_x( 0 ),
	  m_mean_y( 0 ), m_M2_y( 0 ),
	  m_M1_xy( 0 ) { }

    void update( type y, type x, stat_type samples ) {
	if( samples <= 1 ) [[unlikely]] {
	    m_mean_x = x;
	    m_M2_x = 0;
	    m_mean_y = y;
	    m_M2_y = 0;
	    m_M1_xy = 0;
	} else {
	    type mu_x = m_mean_x + ( x - m_mean_x ) / type( samples );
	    type mu_y = m_mean_y + ( y - m_mean_y ) / type( samples );

	    m_M2_x += ( x - m_mean_x ) * ( x - mu_x );
	    m_M2_y += ( y - m_mean_y ) * ( y - mu_y );
	    // or: m_M1_xy += ( y - mu_y ) * ( x - m_mean_x );
	    m_M1_xy += ( y - m_mean_y ) * ( x - mu_x );

	    m_mean_x = mu_x;
	    m_mean_y = mu_y;
	}
    }

    // parallel algorithm
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    covariance_estimator_no_samples
    plus_equals( const covariance_estimator_no_samples<T,S> & r,
		 stat_type lsamples, stat_type rsamples ) {
	type nl = lsamples;
	type nr = rsamples;
	stat_type n = lsamples + rsamples;
	// If no samples, then current contents are appropriate.
	if( n > 0 ) {
	    type m_x = ( nl * m_mean_x + nr * r.m_mean_x ) / type( n );
	    type m_y = ( nl * m_mean_y + nr * r.m_mean_y ) / type( n );
	    type delta_x = r.m_mean_x - m_mean_x;
	    type delta_y = r.m_mean_y - m_mean_y;
	    type f = nl * nr / type( n );
	    type s_x = m_M2_x + r.m_M2_x + delta_x * delta_x * f;
	    type s_y = m_M2_y + r.m_M2_y + delta_y * delta_y * f;
	    m_M1_xy += r.m_M1_xy + delta_x * delta_y * f;
	    m_mean_x = m_x;
	    m_mean_y = m_y;
	    m_M2_x = s_x;
	    m_M2_y = s_y;
	}
	return *this;
    }

    covariance_estimator_no_samples
    minus_equals( const covariance_estimator_no_samples<T,S> & r,
		  stat_type lsamples, stat_type rsamples ) {
	type nl = lsamples;
	type nr = rsamples;
	stat_type n = lsamples - rsamples;
	type m_x = ( nl * m_mean_x - nr * r.m_mean_x ) / type( n );
	type m_y = ( nl * m_mean_y - nr * r.m_mean_y ) / type( n );
	type delta_x = r.m_mean_x - m_mean_x;
	type delta_y = r.m_mean_y - m_mean_y;
	type f = nl * nr / type( n );
	type s_x = m_M2_x - r.m_M2_x - delta_x * delta_x * f; // ??
	type s_y = m_M2_y - r.m_M2_y - delta_y * delta_y * f; // ??
	m_M1_xy = m_M1_xy - r.m_M1_xy + delta_x * delta_y * f; // ??
	m_mean_x = m_x;
	m_mean_y = m_y;
	m_M2_x = s_x;
	m_M2_y = s_y;
	return *this;
    }

    type get_mean( stat_type samples ) const {
	return get_mean_y( samples );
    }
    type get_stdev( stat_type samples ) const {
	return std::sqrt( get_variance( samples ) );
    }
    type get_variance( stat_type samples ) const {
	return get_variance_y( samples );
    }
    type get_sq_correlation( stat_type samples ) const {
	if( samples < 1 )
	    return 0;
	else if( m_M2_x == 0 || m_M2_y == 0 ) {
	    return std::numeric_limits<type>::infinity();
	} else {
	    return ( m_M1_xy / m_M2_x ) * ( m_M1_xy / m_M2_y );
	}
    }

    type get_mean_x( stat_type samples ) const {
	return m_mean_x;
    }
    type get_variance_x( stat_type samples ) const {
	return samples > 1 ? m_M2_x / type( samples - 1 ) : type(0);
    }
    type get_mean_y( stat_type samples ) const {
	return m_mean_y;
    }
    type get_variance_y( stat_type samples ) const {
	return samples > 1 ? m_M2_y / type( samples - 1 ) : type(0);
    }

    stat_type estimate( const covariance_estimator_no_samples & s,
			type rmin, type rmax, stat_type samples ) {
	assert( 0 );
    }
    stat_type estimate_lt( const covariance_estimator_no_samples & s,
			   type r, stat_type samples ) {
	assert( 0 );
    }
    stat_type estimate_ge( const covariance_estimator_no_samples & s,
			   type r, stat_type samples ) {
	assert( 0 );
    }

    type normalize( type x, stat_type samples ) const {
	return ( x - get_mean( samples ) ) / get_stdev( samples );
    }
    type denormalize( type x, stat_type samples ) const {
	return get_mean( samples ) + x * get_stdev( samples );
    }

    type probability_density( type x, stat_type samples ) const {
	assert( 0 );
    }

    std::pair<type,type> split_ltge( type split, stat_type samples ) const {
	assert( 0 );
    }

private:
    type m_mean_x, m_M2_x;
    type m_mean_y, m_M2_y;
    type m_M1_xy;
};

template<typename T, typename S>
class covariance_estimator
    : public covariance_estimator_no_samples<T,S> {
public:
    using type = T;
    using stat_type = S;
    using parent_type = covariance_estimator_no_samples<T,S>;
    
    covariance_estimator()
	: m_samples( 0 ) { }

    void update( type y, type x ) {
	// It is important that samples is incremented before updating
	// mean and standard deviation, such that m_samples is 1 for the
	// first data point, 2 for the second, and so on.
	parent_type::update( y, x, ++m_samples );
    }

    stat_type get_samples() const {
	return m_samples;
    }
    type get_mean() const {
	return parent_type::get_mean( m_samples );
    }
    type get_stdev() const {
	return parent_type::get_stdev( m_samples );
    }
    type get_variance() const {
	return parent_type::get_variance( m_samples );
    }
    type get_sq_correlation() const {
	return parent_type::get_sq_correlation( m_samples );
    }
    type get_mean_x() const {
	return parent_type::get_mean_x( m_samples );
    }
    type get_variance_x() const {
	return parent_type::get_variance_x( m_samples );
    }
    type get_mean_y() const {
	return parent_type::get_mean_y( m_samples );
    }
    type get_variance_y() const {
	return parent_type::get_variance_y( m_samples );
    }

    void estimate( const covariance_estimator & s,
		   type rmin, type rmax ) {
	m_samples = parent_type::estimate( s, rmin, rmax, s.get_samples() );
    }
    void estimate_lt( const covariance_estimator & s, type r ) {
	m_samples = parent_type::estimate_lt( s, r, s.get_samples() );
    }
    void estimate_ge( const covariance_estimator & s, type r ) {
	m_samples = parent_type::estimate_ge( s, r, s.get_samples() );
    }

    covariance_estimator
    operator += ( const covariance_estimator<T,S> & r ) {
	parent_type::plus_equals( r, m_samples, r.m_samples );
	m_samples += r.m_samples;
	return *this;
    }

/*
    covariance_estimator
    operator -= ( const covariance_estimator<T,S> & r ) {
	parent_type::minus_equals( r, m_samples, r.m_samples );
	m_samples -= r.m_samples;
	return *this;
    }
*/

    std::pair<type,type> split_ltge( type split ) {
	return parent_type::split_ltge( split, m_samples );
    }

    void show( std::ostream & os, size_t indent, size_t bin ) const {
	os << std::string( indent*2, ' ' )
	   << "bin " << bin
	   << " numerical_stats { cove: mean_x=" << get_mean_x()
	   << " var_x=" << get_variance_x()
	   << " mean_y=" << get_mean_y()
	   << " var_y=" << get_variance_y()
	   << " rho-sq=" << get_sq_correlation()
	   << " #" << get_samples()
	   << " }\n";
    }

private:
    size_t m_samples;
};


struct adam_optimizer {
    adam_optimizer()
	: m_alpha( 0.01 ), m_beta1( 0.9 ), m_beta2( 0.999 ), m_eps( 1e-8 ),
	  m_m( 0 ), m_v( 0 ), m_t( 0 ) { };

    void reset() {
	m_t = 2;
	m_m = 0;
	m_v = 0;
    }
    float probe_learning_rate( float gradient ) const {
	size_t _m_t = m_t+1;
	
	float gt = gradient;
	float mt = m_beta1 * m_m + ( 1.0f - m_beta1 ) * gt;
	float vt = m_beta2 * m_v + ( 1.0f - m_beta2 ) * gt * gt;
	float mh = mt * ( 1.0f - std::pow( m_beta1, _m_t ) );
	float vh = vt * ( 1.0f - std::pow( m_beta2, _m_t ) );

	float corr = m_alpha * ( mh / ( std::sqrt( vh ) + m_eps ) );

	return corr;
    }

    float get_learning_rate( float gradient ) {
	++m_t;
	
	float gt = gradient;
	float mt = m_beta1 * m_m + ( 1.0f - m_beta1 ) * gt;
	float vt = m_beta2 * m_v + ( 1.0f - m_beta2 ) * gt * gt;
	float mh = mt * ( 1.0f - std::pow( m_beta1, m_t ) );
	float vh = vt * ( 1.0f - std::pow( m_beta2, m_t ) );

	float corr = m_alpha * ( mh / ( std::sqrt( vh ) + m_eps ) );

	m_m = mt;
	m_v = vt;

	return corr;
    }

    size_t get_samples() const { return m_t; }
    
private:
    float m_alpha, m_beta1, m_beta2, m_eps, m_m, m_v;
    size_t m_t;
};

template<typename T, size_t dim>
struct linear_regression_model {
    static constexpr size_t num_dimensions = dim;

    using type = T;
    using x_type = std::array<T,dim>;

    // static constexpr float learning_rate = 0.01;

    linear_regression_model() {
	std::mt19937_64 generator;
	std::uniform_real_distribution<double> wdist{ -1.0, 1.0 };
	for( size_t i=0; i < dim; ++i )
	    m_weights[i] = wdist( generator );
	m_intercept = wdist( generator );
    }

    type predict( const x_type & x ) const {
	type y = m_intercept;
	for( size_t i=0; i < num_dimensions; ++i )
	    y += x[i] * m_weights[i];
	return y;
    }

    // Loss function: L(y,x) = sum_i ( y - t )**2 with t = (1,x) dot* w
    // Gradient loss: d/dx_i = - 2 * ( y - t ) * w_i
    // Gradient loss: d/dw_i = - 2 * ( y - t ) * x_i
    // Gradient loss: d/dw_icept = - 2 * ( y - t )
    type fit( type y, const x_type & x ) {
	// predict
	type t = predict( x );
	
	// Special-case handling as we can fit exactly one straight line
	// through two data points. This should be a decent starting point.
	if constexpr ( false && num_dimensions ==  1 ) {
	    if( m_opt.get_samples() == 1 ) [[unlikely]] {
		// y = ax+b
		// a = ( y1 - y0 ) / ( x1 - x0 )
		// b = y0-ax0 = y1-ax1
		type y0 = m_first_y;
		type y1 = y;
		type x0 = m_first_x;
		type x1 = x[0];
		m_weights[0] = ( y1 - y0 ) / ( x1 - x0 );
		m_intercept = y1 - m_weights[0] * x1;

		m_opt.reset(); // counts samples
		return y - t;
	    } else if( m_opt.get_samples() == 0 ) [[unlikely]] {
		m_first_x = x[0];
		m_first_y = y;
	    }
	}
	
	// update gradients
	type loss_gradient = type(-2) * ( y - t );
	type lrate = m_opt.get_learning_rate( loss_gradient );
	for( size_t i=0; i < num_dimensions; ++i )
	    m_weights[i] -= lrate * x[i];

	// update intercept
	m_intercept -= lrate;

	return y - t;
    }

    void show( std::ostream & os, size_t indent ) const {
	os << std::string( indent*2, ' ' )
	   << "linear_regressor { weights=" << m_intercept
	   << " : ";
	for_each( m_weights, [&]( type t ) { os << t << ' '; } );
	os << "}\n";
    }

    void show_gnuplot( std::ostream & os, const std::string & fname,
		       size_t & fid ) const {
	os << fname << fid << '(';
	print_attributes<0,dim>( os );
	os << ")=" << m_intercept;
	show_gnuplot_impl( os, m_weights,
			   std::make_index_sequence<num_dimensions>() );
	os << '\n';
	++fid;
    }

    type get_learning_rate( type y, const x_type & x ) {
	// predict
	type t = predict( x );
	
	// calculate gradient
	type loss_gradient = type(-2) * ( y - t );
	return m_opt.probe_learning_rate( loss_gradient );
    }
    type get_intercept() const { return m_intercept; }
    const x_type & get_weights() const { return m_weights; }

private:
    template<size_t... I>
    void show_gnuplot_impl( std::ostream & os, const x_type & w,
			    std::integer_sequence<size_t, I...> ) const {
	( ( os << '+' << std::get<I>( w ) << "*x" << I ), ... );
    }

private:
    type m_intercept;
    x_type m_weights;
    adam_optimizer m_opt;
    type m_first_x, m_first_y;
};
    

} // namespace graptor

#endif // GRAPTOR_STAT_WELFORD_H
