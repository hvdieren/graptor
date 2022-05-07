#!/bin/bash

bindir=bin_ics22
outdir=log_ics22

mkdir -p $bindir
mkdir -p $outdir

function build() {
    local srcfile=$1
    local dstfile=$2
    local outfile=$3
    local flags=$4
    local CXX=g++-7.2.0
    # building Cilk
    local LIB="../build/libgraptorlib.a ./lib/libcilkrts.so"
    local CXXFLAGS="-g -I../build -I../include -Wno-ignored-attributes -ftemplate-backtrace-limit=0 -fcilkplus -O4 -DLONG -ldl -std=c++17 -march=skylake-avx512 ${LIB} -DPAPI_CACHE=1 -DPAPI_SKYLAKE=1"
    # building OpenMP
    #local LIB="../build/libgraptorlib.a"
    #local CXXFLAGS="-g -I../build -I../include -Wno-ignored-attributes -ftemplate-backtrace-limit=0 -fopenmp -O4 -DLONG -ldl -std=c++17 -march=skylake-avx512 ${LIB} -DPAPI_CACHE=1 -DPAPI_SKYLAKE=1"
    if [ ! -f $dstfile ] ; then
	echo $dstfile ...
	echo "${CXX} ${CXXFLAGS} ${flags} ${srcfile} -o ${dstfile} ${LIB} -lnuma -ldl -lpapi " > ${outfile} 
	${CXX} ${CXXFLAGS} ${flags} ${srcfile} -o ${dstfile} ${LIB} -lnuma -ldl -lpapi >> ${outfile} 2>&1
    fi

    if [ ! -f $dstfile ] ; then
	cat $outfile
	exit 1
    fi
}

function get_flags() {
    local v=$1
    local g=$2
    local flags
    declare -A flags

    #flags["GRAPTOR_MM_DEBUG"]=1;

    flags["HUGE_PAGES"]=1;
    flags["VMAP_TIMING"]=1;
    flags["PAPI_CACHE"]=0;

    flags["WIDEN_APPROACH"]=0;

    flags["GRAPTOR"]=2;
    flags["GRAPTOR_CACHED"]=1;
    flags["GRAPTOR_CSC"]=1;
    flags["GRAPTOR_EXTRACT_OPT"]=1;
    flags["ANALYSE_VALUES"]=0;
    flags["VEBO_FORCE"]=1;
    flags["GRAPTOR_DEGREE_BITS"]=1;
    flags["GRAPTOR_USE_MMX"]=0;
    flags["MEMORY_CLOBBER_WORKAROUND"]=0;
    flags["SWITCH_DYNAMIC"]=0;
    flags["SCALE_CONTRIB"]=1;
    flags["NUM_BUCKETS"]=127;
    flags["DENSE_COPY"]=1;
    flags["DEFERRED_UDPATE"]=1;
    flags["SP_THRESHOLD"]=-1;

    if [[ "$v" == *_REDUCE* ]] ; then
	flags["GRAPTOR"]=1;
    fi

    if [[ "$v" == *_var* ]] ; then
	flags["VARIANT"]=`echo "${v}_" | sed -e 's/^.*_var\([0-9][0-9]*\)_.*$/\1/'`;
    fi
    if [[ "$v" == *_bkt* ]] ; then
	flags["NUM_BUCKETS"]=`echo "${v}_" | sed -e 's/^.*_bkt\([0-9][0-9]*\)_.*$/\1/'`;
    fi
    if [[ "$v" == *_dyn* ]] ; then
	flags["SWITCH_DYNAMIC"]=1;
    fi
    if [[ "$v" == *_noscale* ]] ; then
	flags["SCALE_CONTRIB"]=0;
    fi
    if [[ "$v" == *_wide* ]] ; then
	flags["WIDEN_APPROACH"]=1;
    fi
    if [[ "$v" == *_nodense* ]] ; then
	flags["DENSE_COPY"]=0;
    fi
    if [[ "$v" == *_nodefer* ]] ; then
	flags["DEFERRED_UPDATE"]=0;
    fi
    if [[ "$v" == *_th50* ]] ; then
	flags["SP_THRESHOLD"]=50;
    fi
    if [[ "$v" == *_th30* ]] ; then
	flags["SP_THRESHOLD"]=30;
    fi
    if [[ "$v" == *_clobber* ]] ; then
	flags["MEMORY_CLOBBER_WORKAROUND"]=1;
    fi
    if [[ "$v" == *_vl* ]] ; then
	flags["MAX_VL"]=`echo "${v}_" | sed -e 's/^.*_vl\([0-9][0-9]*\)_.*$/\1/'`;
	flags["GRAPTOR_DEGREE_BITS"]=${flags["MAX_VL"]};
    fi
    if [[ "$v" == *_deg* ]] ; then
	flags["GRAPTOR_DEGREE_BITS"]=`echo "${v}_" | sed -e 's/^.*_deg\([0-9][0-9]*\)_.*$/\1/'`;
    fi

    if [[ "$v" == *_anal* ]] ; then
	flags["ANALYSE_VALUES"]=1;
    fi
    if [[ "$v" == *_anal2* ]] ; then
	flags["ANALYSE_VALUES"]=1;
    fi

    if [[ "$v" == *_push* ]] ; then
	flags["GRAPTOR"]=2;
	flags["GRAPTOR_CACHED"]=0;
	flags["GRAPTOR_CSC"]=0;
	flags["GRAPTOR_CSR_INDIR"]=1;
    fi

    if [[ "$v" == *_2push* ]] ; then
	flags["GRAPTOR"]=2;
	flags["GRAPTOR_CACHED"]=1;
	flags["GRAPTOR_CSC"]=0;
	flags["GRAPTOR_CSR_INDIR"]=1;
    fi

    if [[ "$v" == *_papic* ]] ; then
	echo -DPAPI_CACHE=1;
    fi
    if [[ "$v" == *_papib* ]] ; then
	echo -DPAPI_CACHE=1 -DPAPI_CTL=1;
    fi
    if [[ "$v" == *_papis* ]] ; then
	echo -DPAPI_CACHE=1 -DPAPI_STALL=1;
    fi
    if [[ "$v" == *_papip* ]] ; then
	echo -DPAPI_CACHE=1 -DPAPI_PEND1=1;
    fi

    if [[ "$v" == *_nouncond* ]] ; then
	echo -DUNCOND_EXEC=0;
    fi
    if [[ "$v" == *_fbit* ]] ; then
	echo -DFRONTIER_BITMASK=1;
    fi

    if [[ "$v" == *_initid* ]] ; then
	echo -DINITID=1;
    fi

    for key in ${!flags[@]}; do echo -D$key=${flags[$key]} ; done

    if [ $g = "GGVEBO_CSC" ] ; then
	echo -DGRAPHTYPE=GGVEBO -DGG_ALWAYS_MEDIUM -UGG_ACTIVE_EXIST;
    elif [ $g = "GGVEBO_COO" ] ; then
	echo -DGRAPHTYPE=GGVEBO -DGG_ALWAYS_DENSE -UGG_ACTIVE_EXIST;
    elif [ $g = "GGVEBO_COO_CSC" ] ; then
	echo -DGRAPHTYPE=GGVEBO -DGG_ALWAYS_DENSE -UGG_ACTIVE_EXIST -DGGVEBO_COO_CSC_ORDER=1;
    elif [ $g = "VEBOPartCCSR" ] ; then
	echo -DGRAPHTYPE=VEBOPartCCSR -UGG_ACTIVE_EXIST;
    elif [ $g = "GGVEBO" ] ; then
	echo -DGRAPHTYPE=GGVEBO
    elif [ $g = "GGVEBOSIMD" ] ; then
	echo -DGRAPHTYPE=GGVEBOSIMD
    elif [ $g = "Graptor" ] ; then
	echo -DGRAPHTYPE=VEBOGraptorT
    elif [ $g = "CSR" ] ; then
	echo -DGRAPHTYPE=CSR;
    fi
}

function one() {
    local bench=$1
    local version=$2
    local G=$3
    
    echo $bench $version $G ...
    build ../bench/$bench.C ${bindir}/${bench}_${G}_${version} ${outdir}/${bench}_${G}_${version}.txt "$(get_flags $version $G)"
}

# Accelerated PageRank. Note: all versions are mixed-precision version with FP64 accumulators
# FP32 version
one APR_vary base_var10_vl8 Graptor &
# FP64 version
one APR_vary base_var11_vl8 Graptor &
# <1,6,9,1,0>
one APR_vary base_var2046_vl8_wide Graptor &
# Scalar FP32 version
one APR_vary base_var10_vl1 GGVEBO_CSC &
# Scalar FP64 version
one APR_vary base_var11_vl1 GGVEBO_CSC &
# Scalar <1,6,9,1,0>
one APR_vary base_var2046_vl1 GGVEBO_CSC

# PageRank. Note: all versions are mixed-precision version with FP64 accumulators
# FP32 version
one PR_vary base_var10_vl8 Graptor &
# FP64 version
one PR_vary base_var11_vl8 Graptor &
# <0,6,10,0,0>
one PR_vary base_var4620_vl8_wide Graptor &
# FP32 head
one PR_vary base_var9920_vl8_wide Graptor &
# FP32 head + switch to <0,6,10,0,0>
one PR_vary base_var994620_vl8_wide Graptor &
# Scalar FP32
one PR_vary base_var10_vl1 GGVEBO_CSC &
# Scalar FP64
one PR_vary base_var11_vl1 GGVEBO_CSC &
# Scalar <0,6,10,0,0>
one PR_vary base_var4620_vl1 GGVEBO_CSC

# 4 bytes per vertex
one MISv base_var10_vl16 Graptor &
# 2 bits per vertex
one MISv base_var13_vl16_wide Graptor &
# Scalar 4 bytes per vertex
one MISv base_var10_vl1 GGVEBO_CSC &
# Scalar 2 bits per vertex
one MISv base_var13_vl1 GGVEBO_CSC

# FP32
one DSSSP base_var0_vl16_wide_nodense_th30 Graptor &
# custom format
one DSSSP base_var3_vl16_wide_nodense_th30 Graptor

wait
