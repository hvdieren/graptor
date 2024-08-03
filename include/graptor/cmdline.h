#ifndef GRAPTOR_CMDLINE_H
#define GRAPTOR_CMDLINE_H 1

#include <string>
#include <iostream>
#include <cstdlib>
#include <cstring>

struct CommandLine {
    CommandLine( int argc, char** argv, std::string help ) 
	: m_argc( argc ), m_argv( argv ), m_help( help ) {
	if( m_argc <= 1
	    || find_option( "-h" ) > 0 || find_option( "--help" ) > 0 )
	    usage();
    }

    void usage() const {
	std::cout << "usage: " << m_argv[0]
		  << " {options}...\n" << m_help << "\n";
	exit( 1 );
    }

    int __attribute__((noinline))
    find_option( const char * option ) const {
	for( int i=1; i < m_argc; i++ )
	    if( !strcmp( option, m_argv[i] ) )
		return i;
	return -1;
    }

    bool get_bool_option( const char * option, bool dflt = false ) const {
	int pos = find_option( option );
	return pos < 0 ? dflt : true;
    }

    const char *
    get_string_option( const char * option,
		       const char * dflt = nullptr ) const {
	int pos = find_option( option );
	return pos < 0 && pos+1 < m_argc ? dflt : m_argv[pos+1];
    }

    int get_int_option( const char * option, int dflt ) const {
	const char * str = get_string_option( option );
	return str == nullptr ? dflt : atoi( str );
    }

    long get_long_option( const char * option, long dflt ) const {
	const char * str = get_string_option( option );
	return str == nullptr ? dflt : atol( str );
    }

    double get_double_option( const char * option, double dflt ) const {
	const char * str = get_string_option( option );
	if( str != nullptr ) {
	    double val;
	    if( std::sscanf( str, "%lf", &val ) == EOF ) {
		std::cerr
		    << "Bad command line argument: '%s'; expecting double\n";
		exit( -1 );
	    }
	    return val;
	} else
	    return dflt;
    }

    /*
     * Legacy routines for in-place replacement of legacy/parseCommandLine.h
     */
    bool getOption( const char * option ) {
	return get_bool_option( option );
    }

    const char * getOptionValue( const char * option ) {
	return get_string_option( option );
    }

    const char * getOptionValue( const char * option, const char * dflt ) {
	return get_string_option( option, dflt );
    }

    int getOptionIntValue( const char * option, int dflt ) {
	return get_int_option( option, dflt );
    }

    long getOptionLongValue( const char * option, long dflt ) {
	return get_long_option( option, dflt );
    }

    double getOptionDoubleValue( const char * option, double dflt ) { 
	return get_double_option( option, dflt );
    }

private:
    int m_argc;
    char** m_argv;
    std::string m_help;
};
 
#endif // GRAPTOR_CMDLINE_H
