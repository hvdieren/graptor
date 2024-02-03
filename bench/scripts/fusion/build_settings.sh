#!/bin/bash

function get_commit() {
    local x=shift
    if [ x$1 != x ] ; then
	commit=$1
    else
	commit=`( cd ${GRAPTOR} ; git log -n 1 --oneline ) | cut -d' ' -f1 2> /dev/null`
    fi
    echo $commit
}

# build ${GRAPTOR}/bench/$bench.C ${bindir}/${bench}_Graptor_${version} ${outdir}/${bench}_Graptor_${version}.txt "$(get_flags $version)"
function build() {
    local srcfile=$1
    local dstfile=$2
    local outfile=$3
    local flags=$4
    local arch=$5
    #local CXX=g++ # 10.3.0
    #local LIB="${GRAPTOR}/build/libgraptorlib.a ${CILKLDPATH}/libcilkrts.so"
    local LIB=${LD_FLAGS}
    #local CXXFLAGS="-g -I${GRAPTOR}/build -I${GRAPTOR}/include -I${CILK_INC}/include -Wno-ignored-attributes -ftemplate-backtrace-limit=0 -O4 -DLONG -ldl -std=c++17 -march=$arch ${LIB}"
    local cxxflags="${CXXFLAGS} -O4 -std=c++17"
    local PAPIFLAGS=
    if echo $flags | grep PAPI_REGION=1 > /dev/null 2>&1 ; then
	PAPI_FLAGS=-lpapi
    fi
    if [[ "$dstfile" == *_noavx512* ]] ; then
	cxxflags="${cxxflags} -mno-avx512f"
    fi
    if [ ! -f $dstfile ] ; then
	echo $dstfile ...
	#touch $dstfile
	echo "${CXX} ${cxxflags} ${flags} ${srcfile} -o ${dstfile} ${LIB} $PAPI_FLAGS ${LD_FLAGS} " > ${outfile} 
	${CXX} ${cxxflags} ${flags} ${srcfile} -o ${dstfile} ${LIB} $PAPI_FLAGS ${LD_FLAGS} >> ${outfile} 2>&1
    fi

    if [ ! -f $dstfile ] ; then
	echo FAIL $dstfile
	#cat $outfile
	#exit 1
    fi
}

function get_flags() {
    local v=$1
    local flags
    declare -A flags

    flags["HUGE_PAGES"]=1;
    flags["VMAP_TIMING"]=0;

    flags["WIDEN_APPROACH"]=1;

    flags["GRAPTOR"]=2;
    flags["GRAPTOR_CACHED"]=1;
    flags["GRAPTOR_CSC"]=1;
    flags["GRAPTOR_EXTRACT_OPT"]=1;
    flags["VEBO_FORCE"]=1;
    flags["GRAPTOR_DEGREE_BITS"]=1;
    flags["GRAPTOR_USE_MMX"]=0;
    flags["MEMORY_CLOBBER_WORKAROUND"]=0;
    flags["PAPI_REGION"]=0;

    if [[ "$v" == *_bitf* ]] ; then
	flags["FRONTIER_BITMASK"]=1;
    fi
    if [[ "$v" == *_uvvid* ]] ; then
	flags["UNVISITED_BIT"]=0;
    fi
    if [[ "$v" == *_uvbit* ]] ; then
	flags["UNVISITED_BIT"]=1;
    fi

    if [[ "$v" == *_papi* ]] ; then
 	flags["PAPI_REGION"]=1;
    fi
    if [[ "$v" == *_llf* ]] ; then
	flags["LLF"]=1;
    fi
    if [[ "$v" == *_fusion* ]] ; then
	flags["FUSION"]=1;
    fi
    if [[ "$v" == *_nofusion* ]] ; then
	flags["FUSION"]=0;
    fi
    if [[ "$v" == *_clobber* ]] ; then
	flags["MEMORY_CLOBBER_WORKAROUND"]=1;
    fi
    if [[ "$v" == *_vl* ]] ; then
	flags["MAX_VL"]=`echo "${v}_" | sed -e 's/^.*_vl\([0-9][0-9]*\)_.*$/\1/'`;
	flags["GRAPTOR_DEGREE_BITS"]=${flags["MAX_VL"]};
    fi

    for key in ${!flags[@]}; do echo -D$key=${flags[$key]} ; done

    echo -DGRAPHTYPE=VEBOGraptorT
}

