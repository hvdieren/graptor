#include <cstdlib>
#include <unistd.h>
#include <cstdio>
#include <string>

#include "graptor/graptor.h"
#include "graptor/graph/CGraphCSx.h"

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    char* iFile = P.getArgument(0);
    bool binary = P.getOptionValue("-b");             //Galois binary format

    const char * ofile = P.getOptionValue( "-o" );

    // using SrcVID = unsigned int;
    using SrcVID = unsigned long;
    using SrcEID = unsigned long;
    using TgtVID = unsigned int;
    // using TgtEID = unsigned int;
    using TgtEID = unsigned long;

    CGraphCSx<SrcVID,SrcEID> SrcG( iFile, -1 );

    std::cerr << "Read graph.\n";
    std::cerr << "Graph: n=" << SrcG.numVertices()
	      << " m=" << SrcG.numEdges() << std::endl;

    CGraphCSx<TgtVID,TgtEID> TgtG( SrcG, -1 );
    std::cerr << "Converted graph.\n";

    TgtG.writeToBinaryFile( ofile );
    std::cerr << "Wrote graph.\n";

    TgtG.del();
    SrcG.del();

    return 0;
}
