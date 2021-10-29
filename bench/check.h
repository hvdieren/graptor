#ifndef GRAPHGRIND_CHECK_H
#define GRAPHGRIND_CHECK_H

template<typename GraphType, typename floatty>
void readfile( const GraphType & GA, const char *fname, floatty *pr ) {
    VID n = GA.numVertices();
    std::ifstream ifs( fname, std::ifstream::in );
    using flim = std::numeric_limits<floatty>;
    ifs.precision( flim::max_digits10 ); // full precision
    for( VID v=0; v < n; ++v ) {
	ifs >> pr[GA.remapID(v)];
    }
}

template<typename GraphType, typename floatty>
void readfile( const GraphType & GA, const char *fname,
	       const mmap_ptr<floatty> &data ) {
    readfile( GA, fname, data.get() );
}

template<typename GraphType, typename floatty>
void writefile( const GraphType & GA, const char *fname, const floatty *pr,
		bool remap = true ) {
    VID n = GA.numVertices();
    std::ofstream ofs( fname, std::ofstream::out );
    using flim = std::numeric_limits<floatty>;
    ofs.precision( flim::max_digits10 ); // full precision
    if( remap ) {
	for( VID v=0; v < n; ++v )
	    ofs << pr[GA.remapID(v)] << '\n';
    } else {
	for( VID v=0; v < n; ++v )
	    ofs << pr[v] << '\n';
    }
}

template<typename GraphType, typename floatty>
void writefile( const GraphType & GA, const char *fname,
		const mmap_ptr<floatty> &data,
		bool remap = true ) {
    writefile( GA, fname, data.get(), remap );
}


#endif // GRAPHGRIND_CHECK_H
