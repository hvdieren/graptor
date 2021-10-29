#include <cstdlib>
#include <unistd.h>
#include <cstdio>
#include <string>

#include "ligra-numa.h"
#include "cgraph.h"

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );

    size_t n = P.getOptionLongValue( "-n", 1 );
    size_t m = P.getOptionLongValue( "-m", 1 );
    const char * dir = P.getOptionValue( "-d" );
    const char * ofile = P.getOptionValue( "-o" );

    GraphCSx G( n, m, -1 );

    EID * idx = G.getIndex();
    VID * edge = G.getEdges();
    EID enxt = 0;
    VID vnxt = 0;

    for( int i=0; i < 125; ++i ) {
	char buf[30];
	snprintf( buf, sizeof(buf)-1, "/friends-%03d______.txt.bz2", i );
	std::string infile = std::string(dir) + std::string(buf);

	printf( "Reading file '%s'\n", infile.c_str() );

	int fd[2];
	pipe( fd );

	int childpid;
	if( (childpid = fork()) == 0 ) {
	    // Close stdout of child
	    close( 1 );

	    // Duplicate output side of pipe to stdin
	    dup( fd[1] );

	    // Decompress
	    execlp( "/bin/bunzip2", "bunzip2", "-c", infile.c_str(), NULL );

	    exit( 0 );
	} else {
	    // Parent
	    const size_t SIZE = 1024*1024;
	    char * buf = new char[SIZE];

	    FILE *fp = fdopen( fd[0], "r" );

	    char * p = buf, * q;
	    while( (q = fgets( buf, SIZE, fp )) != nullptr ) {
		size_t src = strtoull( p, &q, 10 );
		assert( p != q );
		assert( *q == ':' );
		++q;

		assert( src == vnxt );
		idx[vnxt++] = enxt;
		
		if( !strncmp( q, "private", 7 ) ) {
		    q += 8;
		} else if( !strncmp( q, "notfound", 8 ) ) {
		    q += 9;
		} else if( *q != '\n' ) {
		    bool done = false;
		    do {
			size_t dst = strtoull( q, &q, 10 );
			assert( *q == ',' || *q == '\n' );
			done = *q == '\n';
			++q;

			edge[enxt++] = dst;
		    } while( !done );
		}

		if( vnxt % 1000000 == 0 )
		    break;
	    }
	    
	    delete[] buf;
	    fclose( fp );
	    close( fd[0] );
	}
    }

    std::cerr << "vnxt=" << vnxt << "\n";
    std::cerr << "n=" << n << "\n";
    std::cerr << "enxt=" << enxt << "\n";
    std::cerr << "m=" << m << "\n";

    assert( enxt == m );
    assert( vnxt == n );

    G.writeToBinaryFile( ofile );

    // Write graph to file.
}
