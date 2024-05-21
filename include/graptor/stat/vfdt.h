// -*- c++ -*-
#ifndef GRAPTOR_STAT_VFDT_H
#define GRAPTOR_STAT_VFDT_H

#include <random>
#include <string>
#include <algorithm>
#include <ratio>
#include <cmath>
#include <random>

#include <cstddef>
#include <cassert>

#include "graptor/container/index_range.h"
#include "graptor/stat/welford.h"

namespace graptor {

/***********************************************************************
 * Attributes
 ***********************************************************************/

template<typename Y, typename S, size_t YCategories>
struct categorical_stats {
    using y_type = Y;
    using stat_type = std::make_signed_t<S>;
    static_assert( std::is_integral_v<stat_type>, "integral type required" );

    static constexpr size_t num_categories = YCategories;

    using count_type = std::array<stat_type,num_categories>;

    categorical_stats() : m_stats{ 0 } { }

    y_type predict() const {
	return std::max_element( m_stats.begin(), m_stats.end() )
	    - m_stats.begin();
    }
    void update( y_type && y ) { ++m_stats[y]; }

    stat_type add_counts( count_type & cnt ) const {
	stat_type t = 0;
	for( size_t c=0; c < num_categories; ++c ) {
	    cnt[c] += m_stats[c];
	    t += m_stats[c];
	}
	return t;
    }

    stat_type get_total_count() const {
	stat_type t = 0;
	for( size_t c=0; c < num_categories; ++c )
	    t += m_stats[c];
	return t;
    }
    
    stat_type get_count( y_type c ) const {
	return m_stats[c];
    }

    void show( std::ostream & os, size_t indent, size_t bin ) const {
	os << std::string( indent*2, ' ' )
	   << "bin " << bin
	   << " categorical_stats { counts="
	   << m_stats[0];
	for( size_t c=1; c < num_categories; ++c )
	    os << ',' << m_stats[c];
	os << " }\n";
    }

private:
    count_type m_stats;
};

template<typename T, typename S, typename RVE>
struct numerical_stats {
    using type = T;
    using stat_type = std::make_signed_t<S>;
    static_assert( std::is_floating_point_v<type>,
		   "floating-point type required" );

    using rve_type = RVE;

    type predict() const { return m_stats.get_mean(); }
    void update( type t ) { m_stats.update( t ); }
    void update( type y, type x ) { m_stats.update( y, x ); }

    const rve_type & get_estimator() const { return m_stats; }
    stat_type get_samples() const { return m_stats.get_samples(); }
    type get_mean() const { return m_stats.get_mean(); }
    type get_stdev() const { return m_stats.get_stdev(); }
    type get_variance() const { return m_stats.get_variance(); }

    type get_sq_correlation() const { return m_stats.get_sq_correlation(); }

    void estimate( const numerical_stats & s,
		   type rmin, type rmax ) {
	m_stats.estimate( s.m_stats, rmin, rmax );
    }
    void estimate_lt( const numerical_stats & s, type r ) {
	m_stats.estimate_lt( s.m_stats, r );
    }
    void estimate_ge( const numerical_stats & s, type r ) {
	m_stats.estimate_ge( s.m_stats, r );
    }

    const numerical_stats &
    operator += ( const numerical_stats & s ) {
	m_stats += s.m_stats;
	return *this;
    }

/*
    const numerical_stats &
    operator -= ( const numerical_stats & s ) {
	m_stats -= s.m_stats;
	return *this;
    }
*/

    void show( std::ostream & os, size_t indent, size_t bin ) const {
	m_stats.show( os, indent, bin );
    }

private:
    rve_type m_stats;
};

template<typename T>
struct is_ratio : public std::false_type { };

template<std::intmax_t Num, std::intmax_t Denom>
struct is_ratio<std::ratio<Num,Denom>> : public std::true_type { };

template<typename X, typename Y, typename Enable = void>
struct is_argument_type : public std::is_same<X,Y> { };

template<typename X, typename Y>
struct is_argument_type<X,Y,
			std::enable_if_t<
			    std::is_floating_point_v<X>
			    && is_ratio<Y>::value>>
    : public std::true_type { };

template<typename X, typename T, typename Enable = void>
struct get_argument_val {
    static constexpr X get( T t ) { return t; }
};

template<typename X, typename T>
struct get_argument_val<X,T,
			std::enable_if_t<std::is_floating_point_v<X>
					 && is_ratio<T>::value>> {
    static constexpr X get( T t ) { return float(t.num()) / float(t.den()); }
};

template<typename YRange, typename XRange, typename S>
struct numeric_categorical_attribute_observer {
    using x_range = XRange;
    using x_type = typename XRange::type;
    using y_range = YRange;
    using y_type = typename YRange::type;
    using stat_type = std::make_signed_t<S>;
    static_assert( std::is_integral_v<stat_type>, "integral type required" );

    static constexpr x_type x_min = x_range::min_value;
    static constexpr x_type x_max = x_range::max_value;
    static constexpr size_t num_bins = x_range::num_bins;
    static constexpr size_t num_categories = y_range::num_bins;

    using cat_stat_type = categorical_stats<y_type,stat_type,num_categories>;

    bool predict() const {
	std::array<stat_type, num_categories> counts{ 0 };
	for( size_t b=0; b < num_bins; ++b )
	    m_bins[b].add_counts( counts );
	return std::max_element( counts.begin(), counts.end() )
	    - counts.begin();
    }
    void update( y_type y, const x_type & x ) {
	size_t b = bin_of( x );
	m_bins[b].update( std::forward<y_type>( y ) );
    }

    void show( std::ostream & os, size_t indent ) const {
	os << std::string( indent*2, ' ' )
	   << "numeric_categorical_ao {\n";
	for( size_t b=0; b < num_bins; ++b )
	    m_bins[b].show( os, indent+1, b );
	os << std::string( indent*2, ' ' ) << "}\n";
    }

    // Calculate number of samples
    stat_type get_num_samples() {
	stat_type nelem = 0;
	for_each( m_bins, [&]( auto bin ) {
	    nelem += bin.get_total_count();
	} );
	return nelem;
    }

    // Evaluate improvement on G metric if split
    std::pair<float,float> evaluate_fitness() const {
	// Calculate number of elements in unsplit node and in each
	// proposed child
	stat_type nelem = 0;
	std::array<stat_type,num_bins> bcnt;
	std::array<stat_type,num_categories> ccnt{ 0 };
	for_each( m_bins, bcnt, [&]( const cat_stat_type & b, stat_type & bc ) {
	    nelem += bc = b.add_counts( ccnt );
	} );

	// If there are no elements, then the gain is zero.
	if( nelem == 0 )
	    return std::make_pair( 0.f, 0.f );

	// Calculate gain of unsplit node.
	float gain = 0;
	for( size_t c=0; c < num_categories; ++c ) {
	    const float f = float(ccnt[c]) / float(nelem);
	    if( f > 0.f )
		gain -= f * std::log2(f);
	}

	// Calculate impurity of split nodes and subtract from overall gain
	for( size_t b=0; b < num_bins; ++b ) {
	    if( bcnt[b] > 0 ) {
		float bgain = 0;
		for( size_t c=0; c < num_categories; ++c ) {
		    const float f = float(m_bins[b].get_count( c ))
			/ float(bcnt[b]);
		    if( f > 0.f )
			bgain += f * std::log2(f);
		}
		gain += (float(bcnt[b]) / float(nelem)) * bgain;
	    }
	}

	return std::make_pair( gain, 0.f );
    }

    // Determine best split value, constrained to bin boundaries
    x_type get_split_value() const {
	// Calculate number of elements in unsplit node and in each
	// proposed child
	stat_type nelem = 0;
	std::array<stat_type,num_bins> bcnt;
	std::array<stat_type,num_categories> ccnt{ 0 };
	for_each( m_bins, bcnt, [&]( const cat_stat_type & b, stat_type & bc ) {
	    nelem += bc = b.add_counts( ccnt );
	} );

	// If there are no elements, then we can't split...
	if( nelem == 0 )
	    return 0;

	// Calculate gain of unsplit node.
	// We know that column 0 (ltcnt) is all zeros, so we ignore in
	// calculation.
	float ugain = 0;
	for( size_t c=0; c < num_categories; ++c ) {
	    const float f = float(ccnt[c]) / float(nelem);
	    if( f > 0.f )
		ugain -= f * std::log2(f);
	}

	// Simulate gain of splitting at each bin
	float best_gain = std::numeric_limits<float>::lowest();
	size_t best_bin = 0;
	std::array<stat_type,num_categories> ltcnt{ 0 };
	std::array<stat_type,num_categories> gecnt( ccnt );
	stat_type lt_total = 0;
	stat_type ge_total = nelem;
	for( size_t b=0; b < num_bins; ++b ) {
	    // Assume two bins: less than b, or greater than or equal than b
	    float gain = ugain;
	    if( lt_total > 0 ) {
		float ltgain = 0;
		for( size_t c=0; c < num_categories; ++c ) {
		    const float f = float(ltcnt[c]) / float(lt_total);
		    if( f > 0.f )
			ltgain += f * std::log2( f );
		}
		gain += float(lt_total) / float(nelem) * ltgain;
	    }
	    if( ge_total > 0 ) {
		float gegain = 0;
		for( size_t c=0; c < num_categories; ++c ) {
		    const float f = float(gecnt[c]) / float(ge_total);
		    if( f > 0.f )
			gegain += f * std::log2( f );
		}
		gain += float(ge_total) / float(nelem) * gegain;
	    }

	    if( gain > best_gain ) {
		best_gain = gain;
		best_bin = b;
	    }

	    // Move data points in bin b
	    for( size_t c=0; c < num_categories; ++c ) {
		stat_type m = m_bins[b].get_count( c );
		ltcnt[c] += m;
		lt_total += m;
		gecnt[c] -= m;
		ge_total -= m;
	    }
	}

	return x_min + x_type(best_bin)
	    * ( x_max - x_min ) / x_type( num_bins );
    }

private:
    static size_t bin_of( x_type x ) {
	size_t b = ( x_type( num_bins ) * ( x - x_min ) ) / ( x_max - x_min );
	return std::min( b, num_bins-1 );
    }

private:
    std::array<cat_stat_type,num_bins> m_bins;
};

// Stores extra bin to sum over all bins (avoid repeated accumulation)
template<typename YRange, typename XRange, typename S>
struct numeric_regressor_attribute_observer {
    using x_range = XRange;
    using x_type = typename XRange::type;
    using y_range = YRange;
    using y_type = typename YRange::type;
    using stat_type = std::make_signed_t<S>;
    static_assert( std::is_integral_v<stat_type>, "integral type required" );

    // static constexpr x_type x_min = x_range::min_value;
    // static constexpr x_type x_max = x_range::max_value;
    static constexpr size_t num_bins = x_range::num_bins;

    using cat_stat_type =
	numerical_stats<y_type,stat_type,
			variance_estimator<y_type,stat_type>>;

    numeric_regressor_attribute_observer(
	x_type x_min, x_type x_max, size_t required_observations )
    // : m_min_obs( x_min ), m_max_obs( x_max ),
	: m_min_obs( x_min ),
	  m_max_obs( x_max ),
	  m_bin_width( ( x_max - x_min + 1 + num_bins - 1 ) / num_bins ),
	  m_required_observations( (stat_type)required_observations ) { }
    numeric_regressor_attribute_observer(
	size_t required_observations )
	    // : m_min_obs( std::numeric_limits<x_type>::max() ),
	      // m_max_obs( std::numeric_limits<x_type>::min() ),
	: m_min_obs( x_range::min_value ),
	  m_max_obs( x_range::max_value ),
	  m_bin_width( ( x_range::max_value - x_range::min_value + 1 + num_bins - 1 ) / num_bins ),
	  m_required_observations( (stat_type)required_observations ) { }

    y_type predict() const { return m_bins[num_bins].get_mean(); }
    void update( y_type y, const x_type & x ) {
	// Have bin boundaries been determined?
	if( true || get_num_samples() >= m_required_observations ) { // TEMP
	    if( false && get_num_samples() == m_required_observations ) { // TEMP
		// Adapt formula if x_type is integral
		m_bin_width = x_type( num_bins ) / ( m_max_obs - m_min_obs );
	    }
	    size_t b = bin_of( x );
	    m_bins[b].update( y );
	} else {
	    if( m_min_obs > x )
		m_min_obs = x;
	    if( m_max_obs < x )
		m_max_obs = x;
	}
	m_bins[num_bins].update( y );
    }

    void show( std::ostream & os, size_t indent ) const {
	os << std::string( indent*2, ' ' )
	   << "numeric_regressor_ao [" << m_min_obs
	   << ',' << m_max_obs
	   << '#' << get_num_samples() << "] {\n";
	for( size_t b=0; b < num_bins+1; ++b )
	    m_bins[b].show( os, indent+1, b );
	os << std::string( indent*2, ' ' ) << "}\n";
    }

    // Retrieve number of samples
    stat_type get_num_samples() const { return m_bins[num_bins].get_samples(); }

    // Evaluate improvement on G metric if split
    std::pair<float,float> evaluate_fitness() const {
	// No information yet
	if( get_num_samples() < m_required_observations )
	    return std::make_pair( 0.f, 0.f );
	
	// Calculate variance reduction of unsplit node.
	float vr = m_bins[num_bins].get_variance();
	float d = m_bins[num_bins].get_samples();

	// Calculate variance of split nodes
	for( size_t b=0; b < num_bins; ++b )
	    vr -= ( float(m_bins[b].get_samples()) / d )
		* m_bins[b].get_variance();

	return std::make_pair( vr, 0.f );
    }

    // Determine best split value, constrained to bin boundaries
    x_type get_split_value() const {
	// If bins have not been determined yet, then we can't split...
	if( get_num_samples() < m_required_observations )
	    return 0;
	
	// If there are no elements, then we can't split...
	if( m_bins[num_bins].get_samples() == 0 )
	    return 0;

	// Calculate gain of unsplit node.
	float ugain = m_bins[num_bins].get_variance();
	float d = m_bins[num_bins].get_samples();

	// Pre-calculate greater-than halves
	std::array<typename cat_stat_type::rve_type,num_bins> ge;
	ge[num_bins-1] = m_bins[num_bins-1].get_estimator();
	for( size_t bb=1; bb < num_bins; ++bb ) {
	    size_t b = num_bins - 1 - bb;
	    ge[b] = ge[b+1];
	    ge[b] += m_bins[b].get_estimator();
	}

	// Simulate gain of splitting at each bin
	float best_gain = std::numeric_limits<float>::lowest();
	size_t best_bin = 0;
	typename cat_stat_type::rve_type lt = m_bins[0].get_estimator();
	for( size_t b=1; b < num_bins; ++b ) {
	    // Assume two bins: less than b, or greater than or equal than b
	    float gain = ugain
		- ( float(lt.get_samples()) / d ) * lt.get_variance()
		- ( float(ge[b].get_samples()) / d ) * ge[b].get_variance();

	    // std::cout << "  bin=" << b << " gain=" << gain << "\n";
	    if( gain > best_gain ) {
		best_gain = gain;
		best_bin = b;
	    }

	    // Move split point
	    lt += m_bins[b].get_estimator();
	}

	assert( best_bin != 0 );

	// std::cout << "  split bin=" << best_bin
	// << " value=" << start_of( best_bin ) << '\n';

	return start_of( best_bin );
    }

    void copy_observations(
	numeric_regressor_attribute_observer & lt,
	numeric_regressor_attribute_observer & ge,
	x_type split_val ) {
	size_t sb = bin_of( split_val );

	// Have bins been determined?
	assert( get_num_samples() >= m_required_observations
		&& "Cannot split without beens determined" );

	// To do this properly, would need to keep an rve for the x value

	// Leave scope to discover which elements in the bins on the
	// boundary of the split are observed
#if 0
	lt.m_min_obs = m_min_obs;
	lt.m_max_obs = sb == 0 ? m_min_obs : start_of( sb-1 );
	ge.m_min_obs = sb == num_bins-1 ? m_max_obs : end_of( sb );
	ge.m_max_obs = m_max_obs;

	// Initialize {lt,ge}.m_bins[num_bins] as without it bins are undefined
	lt.m_bins[num_bins].estimate_lt( m_bins[num_bins], split_val );
	ge.m_bins[num_bins].estimate_ge( m_bins[num_bins], split_val );

	// Prime the bins using estimated values
	if( lt.get_num_samples() >= m_required_observations ) {
	    for( size_t b=0; b < sb; ++b )
		lt.m_bins[b].estimate(
		    m_bins[num_bins], lt.start_of( b ), lt.end_of( b ) );
	}
	if( ge.get_num_samples() >= m_required_observations ) {
	    for( size_t b=sb; b < num_bins; ++b )
		ge.m_bins[b].estimate(
		    m_bins[num_bins], ge.start_of( b ), ge.end_of( b ) );
	}
#endif
    }

private:
    size_t bin_of( x_type x ) const {
	size_t b = ( x - m_min_obs ) * m_bin_width;
	return std::max( size_t(0), std::min( b, num_bins-1 ) );
    }
    x_type start_of( size_t b ) const {
	return x_type(m_min_obs) + x_type(b) / m_bin_width;
    }
    x_type end_of( size_t b ) const {
	return start_of( b+1 );
    }

private:
    std::array<cat_stat_type,num_bins+1> m_bins;
    x_type m_min_obs, m_max_obs, m_bin_width;
    stat_type m_required_observations;
};

// Stores extra bin to sum over all bins (avoid repeated accumulation)
template<typename YRange, typename XRange, typename S>
struct linear_model_attribute_observer {
    using x_range = XRange;
    using x_type = typename XRange::type;
    using y_range = YRange;
    using y_type = typename YRange::type;
    using stat_type = std::make_signed_t<S>;
    static_assert( std::is_integral_v<stat_type>, "integral type required" );

    static constexpr size_t num_bins = x_range::num_bins;

    static_assert( std::is_same_v<y_type,x_type>,
		   "correlation requires X and Y same type" );
    using cat_stat_type =
	numerical_stats<y_type,stat_type,
			covariance_estimator<y_type,stat_type>>;

    linear_model_attribute_observer(
	x_type x_min, x_type x_max, size_t required_observations )
	: m_min_obs( x_min ), m_max_obs( x_max ),
	  m_required_observations( (stat_type)required_observations ) { }
    linear_model_attribute_observer(
	size_t required_observations )
	: m_min_obs( std::numeric_limits<x_type>::max() ),
	  m_max_obs( std::numeric_limits<x_type>::min() ),
	  m_required_observations( (stat_type)required_observations ) { }

    y_type predict() const { return m_bins[num_bins].get_mean(); }
    void update( y_type y, const x_type & x ) {
	// Have bin boundaries been determined?
	if( get_num_samples() >= m_required_observations ) {
	    if( get_num_samples() == m_required_observations ) {
		// Adapt formula if x_type is integral
		m_bin_width = x_type( num_bins ) / ( m_max_obs - m_min_obs );
	    }
	    size_t b = bin_of( x );
	    m_bins[b].update( y, x );
	} else {
	    if( m_min_obs > x )
		m_min_obs = x;
	    if( m_max_obs < x )
		m_max_obs = x;
	}
	m_bins[num_bins].update( y, x );
    }

    void show( std::ostream & os, size_t indent ) const {
	os << std::string( indent*2, ' ' )
	   << "linear_model_ao [" << m_min_obs
	   << ',' << m_max_obs
	   << '#' << get_num_samples() << "] {\n";
	for( size_t b=0; b < num_bins+1; ++b )
	    m_bins[b].show( os, indent+1, b );
	os << std::string( indent*2, ' ' ) << "}\n";
    }

    // Retrieve number of samples
    stat_type get_num_samples() const { return m_bins[num_bins].get_samples(); }

    // Evaluate improvement on G metric if split
    std::pair<float,float> evaluate_fitness() const {
	// No information yet
	if( get_num_samples() < m_required_observations )
	    return std::make_pair( 0.f, 0.f );
	
	// Calculate spearson correlation of unsplit node.
	float vr = m_bins[num_bins].get_sq_correlation();
	float d = m_bins[num_bins].get_samples();

	// Calculate variance of split nodes
	for( size_t b=0; b < num_bins; ++b )
	    vr -= ( float(m_bins[b].get_samples()) / d )
		* m_bins[b].get_sq_correlation();

	for( size_t b=0; b < num_bins; ++b )
	    std::cout << "fitness b=" << b << " rho-sq=" << 
		m_bins[b].get_sq_correlation() << "\n";

	return std::make_pair( -vr, 0.f );
    }

    // Determine best split value, constrained to bin boundaries
    x_type get_split_value() const {
	// If bins have not been determined yet, then we can't split...
	if( get_num_samples() < m_required_observations )
	    return 0;
	
	// If there are no elements, then we can't split...
	if( m_bins[num_bins].get_samples() == 0 )
	    return 0;

	// Calculate gain of unsplit node.
	float ugain = m_bins[num_bins].get_sq_correlation();
	float d = m_bins[num_bins].get_samples();

	// Pre-calculate greater-than halves
	std::array<typename cat_stat_type::rve_type,num_bins> ge;
	ge[num_bins-1] = m_bins[num_bins-1].get_estimator();
	for( size_t bb=1; bb < num_bins; ++bb ) {
	    size_t b = num_bins - 1 - bb;
	    ge[b] = ge[b+1];
	    ge[b] += m_bins[b].get_estimator();
	}

	// Simulate gain of splitting at each bin
	float best_gain = std::numeric_limits<float>::lowest();
	size_t best_bin = 0;
	typename cat_stat_type::rve_type lt = m_bins[0].get_estimator();
	for( size_t b=1; b < num_bins; ++b ) {
	    // Assume two bins: less than b, or greater than or equal than b
	    float gain = ugain
		- ( float(lt.get_samples()) / d ) * lt.get_sq_correlation()
		- ( float(ge[b].get_samples()) / d ) * ge[b].get_sq_correlation();
	    gain = -gain; // maximise

	    // std::cout << "  bin=" << b << " gain=" << gain << "\n";
	    // >= splits inf inf real real between inf and real
	    if( gain >= best_gain ) {
		best_gain = gain;
		best_bin = b;
	    }

	    // Move split point
	    lt += m_bins[b].get_estimator();
	}

	assert( best_bin != 0 );

	// std::cout << "  split bin=" << best_bin
		  // << " value=" << start_of( best_bin ) << '\n';

	return start_of( best_bin );
    }

    void copy_observations(
	linear_model_attribute_observer & lt,
	linear_model_attribute_observer & ge,
	x_type split_val ) {
	size_t sb = bin_of( split_val );

	// Have bins been determined?
	assert( get_num_samples() >= m_required_observations
		&& "Cannot split without beens determined" );

	// To do this properly, would need to keep an rve for the x value

	// Leave scope to discover which elements in the bins on the
	// boundary of the split are observed
#if 0
	lt.m_min_obs = m_min_obs;
	lt.m_max_obs = sb == 0 ? m_min_obs : start_of( sb-1 );
	ge.m_min_obs = sb == num_bins-1 ? m_max_obs : end_of( sb );
	ge.m_max_obs = m_max_obs;

	// Initialize {lt,ge}.m_bins[num_bins] as without it bins are undefined
	lt.m_bins[num_bins].estimate_lt( m_bins[num_bins], split_val );
	ge.m_bins[num_bins].estimate_ge( m_bins[num_bins], split_val );

	// Prime the bins using estimated values
	if( lt.get_num_samples() >= m_required_observations ) {
	    for( size_t b=0; b < sb; ++b )
		lt.m_bins[b].estimate(
		    m_bins[num_bins], lt.start_of( b ), lt.end_of( b ) );
	}
	if( ge.get_num_samples() >= m_required_observations ) {
	    for( size_t b=sb; b < num_bins; ++b )
		ge.m_bins[b].estimate(
		    m_bins[num_bins], ge.start_of( b ), ge.end_of( b ) );
	}
#endif
    }

private:
    size_t bin_of( x_type x ) const {
	size_t b = ( x - m_min_obs ) * m_bin_width;
	return std::max( size_t(0), std::min( b, num_bins-1 ) );
    }
    x_type start_of( size_t b ) const {
	return x_type(m_min_obs) + x_type(b) / m_bin_width;
    }
    x_type end_of( size_t b ) const {
	return start_of( b+1 );
    }

private:
    std::array<cat_stat_type,num_bins+1> m_bins;
    x_type m_min_obs, m_max_obs, m_bin_width;
    stat_type m_required_observations;
};



/**
 * hoeffding tree config
 */
struct hoeffding_tree_config {
    hoeffding_tree_config( size_t min_samples,
			   size_t max_samples,
			   size_t check_interval,
			   float success_probability,
			   size_t bin_observations )
	: m_min_samples( min_samples ),
	  m_max_samples( max_samples ),
	  m_check_interval( check_interval ),
	  m_success_probability( success_probability ),
	  m_bin_observations( bin_observations ) { }

    size_t get_min_samples() const { return m_min_samples; }
    size_t get_max_samples() const { return m_max_samples; }
    size_t get_check_interval() const { return m_check_interval; }
    float get_success_probability() const { return m_success_probability; }
    size_t get_bin_observations() const { return m_bin_observations; }
    
private:
    size_t m_min_samples, m_max_samples, m_check_interval;
    float m_success_probability;
    size_t m_bin_observations;
};

/**
 * attribute ranges
 */
template<typename T, T Min, T Max, size_t Bins, typename Enable = void>
struct numeric_integral_dense_range;

template<typename T, T Min, T Max, size_t Bins>
struct numeric_integral_dense_range<T,Min,Max,Bins,
				    std::enable_if_t<std::is_integral_v<T>>> {
    using type = T;

    static constexpr T min_value = Min;
    static constexpr T max_value = Max;
    static constexpr size_t num_bins = Bins;
    static constexpr bool is_nominal = true;
};

template<typename T, typename Min, typename Max, size_t Bins,
	 typename Enable = void>
struct numeric_float_dense_range;

template<typename T, typename Min, typename Max, size_t Bins>
struct numeric_float_dense_range<T,Min,Max,Bins,
				 std::enable_if_t<std::is_floating_point_v<T>>> {
    using type = T;

    static constexpr T min_value = T(Min::num) / T(Min::den);
    static constexpr T max_value = T(Max::num) / T(Max::den);
    static constexpr size_t num_bins = Bins;
    static constexpr bool is_nominal = false;
};

/**
 * attribute descriptor
 */
template<typename XRange>
struct numeric_attribute {
    using x_range = XRange;
    using x_type = typename x_range::type;

/*
    template<typename YRange, typename S>
    using attribute_observer =
	numeric_categorical_attribute_observer<YRange,x_range,S>;
*/
};

/**
 * attribute list. Each T must be an attribute<.,.> type
 */
template<typename Tuple, typename Idx>
struct attribute_types_impl;

template<typename... T, std::size_t... I>
struct attribute_types_impl<std::tuple<T...>,
			    std::integer_sequence<std::size_t, I...>> {
    using type = std::tuple<typename std::tuple_element_t<I,std::tuple<T...>>::x_type ...>;
};

/**
 * attribute observer list. Each T must be an attribute<.,.> type
 */
template<typename Y, typename S, typename Tuple, typename Idx,
	 typename Enable = void>
struct attribute_observers_impl;

template<typename Y, typename S, typename... T, std::size_t... I>
struct attribute_observers_impl<Y, S, std::tuple<T...>,
				std::integer_sequence<std::size_t, I...>,
				std::enable_if_t<Y::is_nominal>> {
    using type = std::tuple<
	numeric_categorical_attribute_observer<
	    Y, typename std::tuple_element_t<I,std::tuple<T...>>::range,
	    S> ...>;

    template<typename... A>
    static type make_attribute_observers( A ... args ) {
	return std::make_tuple(
	    numeric_categorical_attribute_observer<
	    Y, typename std::tuple_element_t<I,std::tuple<T...>>::x_range,
	    S>{ args... } ... );
    }
};

template<typename Y, typename S, typename... T, std::size_t... I>
struct attribute_observers_impl<Y, S, std::tuple<T...>,
				std::integer_sequence<std::size_t, I...>,
				std::enable_if_t<!Y::is_nominal>> {
    using type = std::tuple<
	numeric_regressor_attribute_observer<
	// linear_model_attribute_observer<
	    Y, typename std::tuple_element_t<I,std::tuple<T...>>::x_range,
	    S> ...>;

    template<typename... A>
    static type make_attribute_observers( A ... args ) {
	return std::make_tuple(
	    numeric_regressor_attribute_observer<
	    // linear_model_attribute_observer<
	    Y, typename std::tuple_element_t<I,std::tuple<T...>>::x_range,
	    S>{ args... } ... );
    }
};

template<typename YRange, typename... T>
struct attribute_list {
    static constexpr std::size_t num_attributes = sizeof...(T);
    
    using type = std::tuple<T...>;

    using x_types = typename attribute_types_impl<std::tuple<T...>,
	std::make_index_sequence<num_attributes>>::type;

    using y_range = YRange;
    using y_type = typename y_range::type;

    static constexpr bool y_is_nominal = YRange::is_nominal;

    template<typename S>
    using attribute_observers_type = attribute_observers_impl<
	y_range,S,std::tuple<T...>,
	std::make_index_sequence<num_attributes>>;

    template<typename S>
    using attribute_observers = typename attribute_observers_type<S>::type;

    template<typename S, typename... A>
    static attribute_observers<S> make_attribute_observers( A ... args ) {
	return attribute_observers_type<S>::make_attribute_observers( args... );
    }
};

template<typename YRange, typename Tuple>
struct attribute_list_from_tuple;

template<typename YRange, typename... T>
struct attribute_list_from_tuple<YRange,std::tuple<T...>>
    : public attribute_list<YRange,T...> { };


// Y type is categorical/nominal
template<typename S, typename AL>
struct categorical_learner_type {
    using attributes = AL;
    using y_type = typename AL::y_type;
    using y_range = typename AL::y_range;
    using observer_list = typename AL::attribute_observers<S>;

    static_assert( y_range::is_nominal, "Y must be of nominal type" );

    static constexpr size_t num_categories = AL::num_categories;
    static constexpr std::size_t num_attributes =
	std::tuple_size_v<observer_list>;

    template<typename... T>
    y_type predict( const std::tuple<T...> & ) const {
	return std::get<0>( m_observers ).predict();
    }
    template<typename... T>
    void update( y_type y, const std::tuple<T...> & x ) {
	update_impl( y, x, std::make_index_sequence<num_attributes>() );
    }

    void show( std::ostream & os, size_t indent ) const {
	show_impl( os, indent+1, m_observers,
		   std::make_index_sequence<num_attributes>() );
    }

    template<size_t D>
    auto get_split_value() const {
	return std::get<D>( m_observers ).get_split_value();
    }

    // Check if a split would satisfy the Hoeffding bound, and on
    // which attribute we should split.
    std::pair<bool,size_t> split_check( const hoeffding_tree_config & cfg ) {
	if constexpr ( num_attributes == 0 )
	    return std::make_pair( false, size_t(-1) );

	size_t nsamples = std::get<0>( m_observers ).get_num_samples();

	// TODO: store previous samples at time of split and check
	//       difference > grace period/interval
	if( nsamples <= cfg.get_min_samples()
	    || nsamples % cfg.get_check_interval() != 0 )
	    return std::make_pair( false, size_t(-1) );

	// show( std::cout, 0 );

	// Check fitness of each dimension
	std::array<std::pair<float,float>,num_attributes> gain;
	for_each( m_observers, gain,
		  [&]( auto & ob, std::pair<float,float> & g ) {
		      g = ob.evaluate_fitness();
		  } );

	float lg_gain = std::numeric_limits<float>::lowest(),
	    lg2_gain = std::numeric_limits<float>::lowest();
	size_t lg_index = 0;
	for( size_t d=0; d < num_attributes; ++d ) {
	    // std::cout << "split d=" << d << " gain=" << gain[d].first << "\n";
	    if( gain[d].first > lg_gain ) {
		lg2_gain = lg_gain;
		lg_gain = gain[d].first;
		lg_index = d;
	    } else if( gain[d].first > lg2_gain )
		lg2_gain = gain[d].first;

	    if( gain[d].second > lg2_gain )
		lg2_gain = gain[d].second;
	}

	const float range = (float)y_range::num_bins;
	const float rsq = range * range;
	const float epsilon = std::sqrt(
	    rsq * std::log( 1.0f / (1.0f - cfg.get_success_probability() ))
	    / float( 2 * nsamples ) );

	if( lg_gain > 0.f
	    && ( ( lg_gain - lg2_gain > epsilon )
		 || ( nsamples > cfg.get_max_samples() )
		 || epsilon <= 0.05f ) ) {
	    return std::make_pair( true, lg_index );
	} else
	    return std::make_pair( false, size_t(-1) );
    }

private:
    template<typename... T, size_t... I>
    void update_impl( y_type y, const std::tuple<T...> & x,
		      std::integer_sequence<size_t, I...> ) {
	( std::get<I>( m_observers ).update( y, std::get<I>( x ) ), ... );
    }

    template<typename... T, size_t... I>
    void show_impl( std::ostream & os, size_t indent,
		    const std::tuple<T...> & x,
		    std::integer_sequence<size_t, I...> ) const {
	( std::get<I>( x ).show( os, indent ), ... );
    }

private:
    observer_list m_observers;
};

// Y type is numeric
// Assumes all attributes are numeric including the predicted attribute
template<typename S, typename AL>
struct regression_learner_type {
    using attributes = AL;
    static constexpr size_t num_dimensions = attributes::num_attributes;
    using y_type = typename AL::y_type;
    using y_range = typename AL::y_range;
    using observer_list = typename AL::attribute_observers<S>;

    // Also requires that all X's have the same type as Y
    static_assert( !y_range::is_nominal, "Y must not be of nominal type" );
    static_assert( std::is_floating_point_v<y_type>,
		   "floating-point type required" );

    static constexpr size_t num_categories = AL::num_categories;
    static constexpr std::size_t num_attributes =
	std::tuple_size_v<observer_list>;

    regression_learner_type( const hoeffding_tree_config & cfg )
	: m_observers(
	    AL::template make_attribute_observers<S>(
		cfg.get_bin_observations() ) ) { }

    template<typename... T>
    y_type predict( const std::tuple<T...> & x ) const {
	return std::get<0>( m_observers ).predict();
    }
    template<typename... T>
    void update( y_type y, const std::tuple<T...> & x ) {
	update_impl( y, x, std::make_index_sequence<num_attributes>() );
    }

    void show( std::ostream & os, size_t indent ) const {
	show_impl( os, indent+1, m_observers,
		   std::make_index_sequence<num_attributes>() );
    }
    void show_gnuplot( std::ostream & os, const std::string & fname,
		       size_t & fid ) const {
    }

    template<size_t D>
    auto get_split_value() const {
	return std::get<D>( m_observers ).get_split_value();
    }

    // Check if a split would satisfy the Hoeffding bound, and on
    // which attribute we should split.
    std::pair<bool,size_t> split_check( const hoeffding_tree_config & cfg ) {
	if constexpr ( num_attributes == 0 )
	    return std::make_pair( false, size_t(-1) );

	size_t nsamples = std::get<0>( m_observers ).get_num_samples();

	// TODO: store previous samples at time of split and check
	//       difference > grace period/interval
	if( nsamples <= cfg.get_min_samples()
	    || nsamples % cfg.get_check_interval() != 0 )
	    return std::make_pair( false, size_t(-1) );

	// show( std::cout, 0 );

	// TODO: determine fitness to split based on error,
	//       not variance of error?

	// Check fitness of each dimension
	std::array<std::pair<float,float>,num_attributes> gain;
	for_each( m_observers, gain,
		  [&]( auto & ob, std::pair<float,float> & g ) {
		      g = ob.evaluate_fitness();
		  } );

	float lg_gain = std::numeric_limits<float>::lowest(),
	    lg2_gain = std::numeric_limits<float>::lowest();
	size_t lg_index = 0;
	for( size_t d=0; d < num_attributes; ++d ) {
	    // std::cout << "split d=" << d << " gain=" << gain[d].first << "\n";
	    if( gain[d].first > lg_gain ) {
		lg2_gain = lg_gain;
		lg_gain = gain[d].first;
		lg_index = d;
	    } else if( gain[d].first > lg2_gain )
		lg2_gain = gain[d].first;

	    if( gain[d].second > lg2_gain )
		lg2_gain = gain[d].second;
	}

	const float range = (float)( y_range::max_value - y_range::min_value );
	const float rsq = range * range;
	const float epsilon = std::sqrt(
	    rsq * std::log( 1.0f / (1.0f - cfg.get_success_probability() ))
	    / float( 2 * nsamples ) );

	if( lg_gain > 0.f
	    && ( ( lg_gain - lg2_gain > epsilon )
		 || ( nsamples > cfg.get_max_samples() )
		 || epsilon <= 0.05f ) ) {
	    return std::make_pair( true, lg_index );
	} else
	    return std::make_pair( false, size_t(-1) );
    }

    template<size_t dim>
    void copy_observations(
	regression_learner_type & lt,
	regression_learner_type & ge,
	std::tuple_element_t<dim,typename attributes::x_types> split_val ) {
/*
	std::get<dim>( m_observers ).copy_observations(
	    std::get<dim>( lt.m_observers ),
	    std::get<dim>( ge.m_observers ),
	    split_val );
*/
    }

private:
    template<typename... T>
    static std::array<y_type,num_dimensions>
    ttoa( const std::tuple<T...> & x ) {
	return std::apply( 
	    []( auto & ... t ) {
		return std::array<y_type,num_dimensions>{ t... };
	    }, x );
    }

    template<typename... T, size_t... I>
    void update_impl( y_type y, const std::tuple<T...> & x,
		      std::integer_sequence<size_t, I...> ) {
	( std::get<I>( m_observers ).update( y, std::get<I>( x ) ), ... );
    }

    template<typename... T, size_t... I>
    void show_impl( std::ostream & os, size_t indent,
		    const std::tuple<T...> & x,
		    std::integer_sequence<size_t, I...> ) const {
	( std::get<I>( x ).show( os, indent ), ... );
    } 
private:
    observer_list m_observers;
};

// Y type is numeric
// Assumes all attributes are numeric including the predicted attribute
template<typename S, typename AL>
struct model_learner_type {
    using attributes = AL;
    static constexpr size_t num_dimensions = attributes::num_attributes;
    using y_type = typename AL::y_type;
    using y_range = typename AL::y_range;
    using observer_list = typename AL::attribute_observers<S>;

    using regressor_type = linear_regression_model<y_type,num_dimensions>;

    // Also requires that all X's have the same type as Y
    static_assert( !y_range::is_nominal, "Y must not be of nominal type" );
    static_assert( std::is_floating_point_v<y_type>,
		   "floating-point type required" );

    static constexpr bool differentiate_errors = false; // true;

    static constexpr size_t num_categories = AL::num_categories;
    static constexpr std::size_t num_attributes =
	std::tuple_size_v<observer_list>;

    model_learner_type( const hoeffding_tree_config & cfg )
	: m_observers(
	    AL::template make_attribute_observers<S>(
		cfg.get_bin_observations() ) ) { }

    template<typename... T>
    y_type predict( const std::tuple<T...> & x ) const {
	return m_regressor.predict( ttoa( x ) );
    }
    template<typename... T>
    void update( y_type y, const std::tuple<T...> & x ) {
	y_type err = m_regressor.fit( y, ttoa( x ) );
	update_impl( differentiate_errors ? err : y,
		     x, std::make_index_sequence<num_attributes>() );
    }

    void show( std::ostream & os, size_t indent ) const {
	m_regressor.show( os, indent+1 );
	show_impl( os, indent+1, m_observers,
		   std::make_index_sequence<num_attributes>() );
    }
    void show_gnuplot( std::ostream & os, const std::string & fname,
		       size_t & fid ) const {
	m_regressor.show_gnuplot( os, fname, fid );
    }

    template<size_t D>
    auto get_split_value() const {
	return std::get<D>( m_observers ).get_split_value();
    }

    // Check if a split would satisfy the Hoeffding bound, and on
    // which attribute we should split.
    std::pair<bool,size_t> split_check( const hoeffding_tree_config & cfg ) {
	if constexpr ( num_attributes == 0 )
	    return std::make_pair( false, size_t(-1) );

	size_t nsamples = std::get<0>( m_observers ).get_num_samples();

	// TODO: store previous samples at time of split and check
	//       difference > grace period/interval
	if( nsamples <= cfg.get_min_samples()
	    || nsamples % cfg.get_check_interval() != 0 )
	    return std::make_pair( false, size_t(-1) );

	// show( std::cout, 0 );

	// TODO: determine fitness to split based on error,
	//       not variance of error?

	// Check fitness of each dimension
	std::array<std::pair<float,float>,num_attributes> gain;
	for_each( m_observers, gain,
		  [&]( auto & ob, std::pair<float,float> & g ) {
		      g = ob.evaluate_fitness();
		  } );

	float lg_gain = std::numeric_limits<float>::lowest(),
	    lg2_gain = std::numeric_limits<float>::lowest();
	size_t lg_index = 0;
	for( size_t d=0; d < num_attributes; ++d ) {
	    // std::cout << "split d=" << d << " gain=" << gain[d].first << "\n";
	    if( gain[d].first > lg_gain ) {
		lg2_gain = lg_gain;
		lg_gain = gain[d].first;
		lg_index = d;
	    } else if( gain[d].first > lg2_gain )
		lg2_gain = gain[d].first;

	    if( gain[d].second > lg2_gain )
		lg2_gain = gain[d].second;
	}

	const float range = (float)( y_range::max_value - y_range::min_value );
	const float rsq = range * range;
	const float epsilon = std::sqrt(
	    rsq * std::log( 1.0f / (1.0f - cfg.get_success_probability() ))
	    / float( 2 * nsamples ) );

	if( lg_gain > 0.f
	    && ( ( lg_gain - lg2_gain > epsilon )
		 || ( nsamples > cfg.get_max_samples() )
		 || epsilon <= 0.05f ) ) {
	    return std::make_pair( true, lg_index );
	} else
	    return std::make_pair( false, size_t(-1) );
    }

    template<size_t dim>
    void copy_observations(
	model_learner_type & lt,
	model_learner_type & ge,
	std::tuple_element_t<dim,typename attributes::x_types> split_val ) {
	std::get<dim>( m_observers ).copy_observations(
	    std::get<dim>( lt.m_observers ),
	    std::get<dim>( ge.m_observers ),
	    split_val );

	// Keep the learned regression models
	// TODO: reset Adam optimizer to high learning rate?
	lt.m_regressor = m_regressor;
	ge.m_regressor = m_regressor;
    }

private:
    template<typename... T>
    static std::array<y_type,num_dimensions>
    ttoa( const std::tuple<T...> & x ) {
	return std::apply( 
	    []( auto & ... t ) {
		return std::array<y_type,num_dimensions>{ t... };
	    }, x );
    }

    template<typename... T, size_t... I>
    void update_impl( y_type y, const std::tuple<T...> & x,
		      std::integer_sequence<size_t, I...> ) {
	( std::get<I>( m_observers ).update( y, std::get<I>( x ) ), ... );
    }

    template<typename... T, size_t... I>
    void show_impl( std::ostream & os, size_t indent,
		    const std::tuple<T...> & x,
		    std::integer_sequence<size_t, I...> ) const {
	( std::get<I>( x ).show( os, indent ), ... );
    } 
private:
    observer_list m_observers;
    regressor_type m_regressor;
};


template<typename S, typename AL, typename Enable = void>
using learner_type =
    std::conditional_t<AL::y_is_nominal,
		       categorical_learner_type<S,AL>,
		       // model_learner_type<S,AL>>;
		       regression_learner_type<S,AL>>;

/**
 * VFDT node: leaf or internal
 * Abstract base class, because we cannot know the list of untested attributes.
 */
template<typename S, typename AL, typename UL>
struct node {
    using attributes = AL;
    using attribute_types = typename attributes::x_types;
    using undecided_attributes = UL;
    using y_type = typename AL::y_type;

    virtual y_type predict( attribute_types && t ) const = 0;
    virtual bool update( y_type &&, attribute_types &&,
			 node *&,
			 const hoeffding_tree_config & ) = 0;
    virtual void show( std::ostream &, size_t ) const = 0;
    virtual void show_gnuplot( std::ostream &, const std::string &,
			       size_t & ) const = 0;
};

template<typename S, typename AL, typename UL>
struct leaf_node;

template<typename S, typename AL, typename UL, size_t AI>
struct binary_split_node : public node<S,AL,UL> {
    static constexpr size_t split_index = AI; //!< index in AL, element of UL
    using attributes = AL;
    using y_type = typename AL::y_type;
    using undecided_attributes = UL;
    using attribute_types = typename attributes::x_types;
    using split_attribute =
	std::tuple_element_t<split_index, typename attributes::type>;
    using split_type = typename split_attribute::x_type;
    using remaining_attributes = undecided_attributes;
    // typename integer_sequence_remove<split_index,undecided_attributes>::type;
    using child_node = node<S, attributes, remaining_attributes>;
    using child_leaf_node = leaf_node<S, attributes, remaining_attributes>;

    static_assert( split_index < attributes::num_attributes,
		   "split_index out of range" );

    binary_split_node( child_node * l, child_node * r, split_type split )
	: m_left( l ), m_right( r ), m_split( split ) { }
    ~binary_split_node() {
	m_left->deallocate();
	m_right->deallocate();
    }

    virtual y_type predict( attribute_types && x ) const {
	auto xi = std::get<split_index>( x );
	const child_node * p = xi < m_split ? m_left : m_right;
	return p->predict( std::forward<attribute_types>( x ) );
    }

    bool update( y_type && y, attribute_types && x, node<S,AL,UL> *& me,
		 const hoeffding_tree_config & cfg ) {
	auto xi = std::get<split_index>( /*std::make_tuple(*/ x/*... )*/ );
	child_node *& p = xi < m_split ? m_left : m_right;
	return p->update( std::forward<y_type>( y ),
			  std::forward<attribute_types>( x ),
			  p, cfg );
    }

    child_node * get_left() const { return m_left; }
    child_node * get_right() const { return m_right; }

    void show( std::ostream & os, size_t indent ) const {
	os << std::string( indent*2, ' ' )
	   << "binary_split_node [split_index=" << split_index
	   << " @" << m_split
	   << "] {\n";
	m_left->show( os, indent+1 );
	m_right->show( os, indent+1 );
	os << std::string( indent*2, ' ' ) << "}\n";
    }
    void show_gnuplot( std::ostream & os, const std::string & fname,
		       size_t & fid ) const {
	m_left->show_gnuplot( os, fname, fid );
	size_t fidl = fid-1;
	m_right->show_gnuplot( os, fname, fid );
	size_t fidr = fid-1;
	os << fname << fid << "(";
	print_attributes<0,attributes::num_attributes>( os );
	os << ")=x" << split_index << '<' << m_split << '?'
	   << fname << fidl << '(';
	print_attributes<0,attributes::num_attributes>( os );
	os << "):" << fname << fidr << '(';
	print_attributes<0,attributes::num_attributes>( os );
	os << ")\n";
	++fid;
    }

private:
    child_node * m_left, * m_right;
    split_type m_split;
};

template<typename S, typename AL, typename UL>
struct leaf_node : public node<S,AL,UL> {
    using attributes = AL;
    using y_type = typename AL::y_type;
    using stat_type = S;
    using undecided_attributes = UL;
    using attribute_types = typename attributes::x_types;
/* To exclude repeated splitting on the same attribute
    using observed_attributes =
	attribute_list_from_tuple<
	typename attributes::y_range,
	typename tuple_select_type<typename attributes::type,
				   undecided_attributes>::type>;
*/
    using observed_attributes = attributes;
    using node_learner_type = learner_type<S,observed_attributes>;

    leaf_node( const hoeffding_tree_config & cfg ) : m_learner( cfg ) { };

    void deallocate() { }

    y_type predict( attribute_types && x ) const {
	return m_learner.predict( std::forward<attribute_types>( x ) );
    }
    bool update( y_type && y, attribute_types && x, node<S,AL,UL> *& me,
		 const hoeffding_tree_config & cfg ) {
	auto xr = tuple_select( std::forward<attribute_types>( x ),
				undecided_attributes() );
	m_learner.update( std::forward<y_type>( y ), std::move( xr ) );

	std::pair<bool,size_t> split = m_learner.split_check( cfg );
	if( split.first ) {
	    // std::cout << "split at dimension " << split.second << "\n";

	    split_node<0>( split.second, me, cfg );
	    return true;
	}
	return false;
    }

    void show( std::ostream & os, size_t indent ) const {
	os << std::string( indent*2, ' ' )
	   << "leaf_node [observed=";
	show_helper( os, undecided_attributes() );
	os << "] {\n";
	m_learner.show( os, indent+1 );
	os << std::string( indent*2, ' ' ) << "}\n";
    }
    void show_gnuplot( std::ostream & os, const std::string & fname,
		       size_t & fid ) const {
	m_learner.show_gnuplot( os, fname, fid );
    }

private:
    template<size_t... Is>
    void show_helper( std::ostream & os, std::integer_sequence<size_t,Is...> )
	const {
	std::array<std::size_t,sizeof...(Is)> ua{ Is... };
	for( size_t i=0; i < sizeof...(Is); ++i )
	    os << ua[i] << ',';
    }

    template<size_t dim>
    void split_node( size_t split_dim, node<S,AL,UL> *& me,
		     const hoeffding_tree_config & cfg ) {
	if( dim == split_dim ) {
	    using bsn = binary_split_node<
		stat_type,attributes,undecided_attributes,dim>;
	    using ln = typename bsn::child_leaf_node;
	    typename bsn::split_type split_val
		= m_learner.template get_split_value<dim>();
	    ln * nl = new ln( cfg );
	    ln * nr = new ln( cfg );
	    m_learner.template copy_observations<dim>(
		nl->m_learner, nr->m_learner, split_val );
	    auto * sn = new bsn( nl, nr, split_val );
	    // Should consider transfering gained knowledge on likely
	    // categories without retaining individual data points, e.g.,
	    // copy over observers for undecided attributes to children.
	    // If observers remain the same, could also recycle me as one of
	    // the new children.
	    delete me; // this == me
	    me = sn;
	} else if constexpr ( dim+1 < attributes::num_attributes )
	    split_node<dim+1>( split_dim, me, cfg );
    }

private:
    node_learner_type m_learner;
};

template<typename YRange, typename... AL>
class binary_vfdt {
public:
    using y_range = YRange;
    using attributes = attribute_list<YRange,numeric_attribute<AL>...>;
    using root_node_type =
	node<long, attributes,
	     std::make_index_sequence<attributes::num_attributes>>;
    using leaf_node_type =
	leaf_node<long, attributes,
		  std::make_index_sequence<attributes::num_attributes>>;

    using attribute_types = typename attributes::x_types;
    using predicted_type = typename attributes::y_type;

    binary_vfdt( hoeffding_tree_config && cfg )
	: m_root( new leaf_node_type() ),
	  m_cfg( std::forward<hoeffding_tree_config >( cfg ) ) { }
    binary_vfdt( const hoeffding_tree_config & cfg )
	: m_root( new leaf_node_type( cfg ) ),
	  m_cfg( cfg ) { }
    binary_vfdt( const binary_vfdt & ) = delete; // NYI
    binary_vfdt( binary_vfdt && v )
	: m_root( std::forward<root_node_type *>( v.m_root ) ),
	  m_cfg( std::forward<hoeffding_tree_config>( v.m_cfg ) ) {
	v.m_root = nullptr;
    }
    ~binary_vfdt() {
	if( m_root != nullptr )
	    delete m_root;
    }

    binary_vfdt & operator = ( const binary_vfdt & ) = delete; // NYI
    binary_vfdt & operator = ( binary_vfdt && ) = delete; // NYI

    template<typename... T>
    predicted_type predict( T &&... x ) const {
	return m_root->predict( std::make_tuple( std::forward<T>( x )... ) );
    }

    template<typename... T>
    bool update( predicted_type y, T &&... x ) {
	return m_root->update( std::forward<predicted_type>( y ),
			       std::make_tuple( std::forward<T>( x )... ),
			       m_root,
			       m_cfg );
    }

    void show( std::ostream & os ) {
	os << "binary_vfdt:\n";
	m_root->show( os, 0 );
    }

    size_t show_gnuplot( std::ostream & os, const std::string & fname ) const {
	size_t fid = 0;
	m_root->show_gnuplot( os, fname, fid );
	return fid-1;
    }


private:
    root_node_type * m_root;
    hoeffding_tree_config m_cfg;
};

} // namespace graptor

#endif // GRAPTOR_STAT_VFDT_H

