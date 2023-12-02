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
    flags["ONLY_CUTOUT"]=0;
    flags["USE_PRESTUDY"]=0;
    flags["UNDERLYING"]="";
    flags["PAPI_REGION"]=0;

    if [[ "$v" == *_papi* ]] ; then
 	flags["PAPI_REGION"]=1;
    fi
    if [[ "$v" == *_fusion* ]] ; then
	flags["FUSION"]=1;
    fi
    if [[ "$v" == *_op0* ]] ; then
	flags["OPERATION"]=0;
    fi
    if [[ "$v" == *_op1* ]] ; then
	flags["OPERATION"]=1;
    fi
    if [[ "$v" == *_op2* ]] ; then
	flags["OPERATION"]=2;
    fi
    if [[ "$v" == *_nofusion* ]] ; then
	flags["FUSION"]=0;
    fi
    if [[ "$v" == *_cutout* ]] ; then
	flags["ONLY_CUTOUT"]=1;
    fi
    if [[ "$v" == *_pre* ]] ; then
	flags["USE_PRESTUDY"]=1;
    fi
    if [[ "$v" == *_merge_scalar* ]] ; then
	flags["UNDERLYING"]="merge_scalar";
    fi
    if [[ "$v" == *_merge_vector* ]] ; then
	flags["UNDERLYING"]="merge_vector";
    fi
    if [[ "$v" == *_merge_jump* ]] ; then
	flags["UNDERLYING"]="merge_jump";
    fi
    if [[ "$v" == *_hash_scalar* ]] ; then
	flags["UNDERLYING"]="hash_scalar";
    fi
    if [[ "$v" == *_hash_vector* ]] ; then
	flags["UNDERLYING"]="hash_vector";
    fi
    if [[ "$v" == *_no512* ]] ; then
	flags["USE_512_VECTOR"]=0;
    fi
    if [[ "$v" == *_yes512* ]] ; then
	flags["USE_512_VECTOR"]=1;
    fi
    if [[ "$v" == *_abDxph* ]] ; then
	flags["ABLATION_DENSE_DISABLE_XP_HASH"]="1";
    fi
    if [[ "$v" == *_abBxph* ]] ; then
	flags["ABLATION_BLOCKED_DISABLE_XP_HASH"]="1";
    fi
    if [[ "$v" == *_abAxph* ]] ; then
	flags["ABLATION_HADJPA_DISABLE_XP_HASH"]="1";
    fi
    if [[ "$v" == *_abA2xph* ]] ; then
	flags["ABLATION_HADJPA_DISABLE_XP_HASH"]="2";
    fi
    if [[ "$v" == *_abPxph* ]] ; then
	flags["ABLATION_PIVOT_DISABLE_XP_HASH"]="1";
    fi
    if [[ "$v" == *_abDnpt* ]] ; then
	flags["ABLATION_DENSE_NO_PIVOT_TOP"]="1";
    fi
    if [[ "$v" == *_abBnpt* ]] ; then
	flags["ABLATION_BLOCKED_NO_PIVOT_TOP"]="1";
    fi
    if [[ "$v" == *_abBhm* ]] ; then
	flags["ABLATION_BLOCKED_HASH_MASK"]="1";
    fi
    if [[ "$v" == *_abDf* ]] ; then
	flags["ABLATION_DENSE_PIVOT_FILTER"]="1";
    fi
    if [[ "$v" == *_abBf* ]] ; then
	flags["ABLATION_BLOCKED_PIVOT_FILTER"]="1";
    fi
    if [[ "$v" == *_abDhm* ]] ; then
	flags["ABLATION_DENSE_HASH_MASK"]="1";
    fi
    if [[ "$v" == *_abBc* ]] ; then
	flags["ABLATION_BLOCKED_EXCEED"]="1";
    fi
    if [[ "$v" == *_abDc* ]] ; then
	flags["ABLATION_DENSE_EXCEED"]="1";
    fi
    if [[ "$v" == *_abGc* ]] ; then
	flags["ABLATION_GENERIC_EXCEED"]="1";
    fi
    if [[ "$v" == *_abDi* ]] ; then
	flags["ABLATION_DENSE_ITERATE"]="1";
    fi
    if [[ "$v" == *_abBi* ]] ; then
	flags["ABLATION_BLOCKED_ITERATE"]="1";
    fi
    if [[ "$v" == *_abTd* ]] ; then
	flags["ABLATION_DISABLE_TOP_DENSE"]="1";
    fi
    if [[ "$v" == *_abpdeg* ]] ; then
	flags["ABLATION_PDEG"]="1";
    fi
    if [[ "$v" == *_abTy* ]] ; then
	flags["ABLATION_DISABLE_TOP_TINY"]="1";
    fi
    if [[ "$v" == *_abL* ]] ; then
	flags["ABLATION_DISABLE_LEAF"]="1";
    fi
    if [[ "$v" == *_abRPs* ]] ; then
	flags["ABLATION_RECPAR_CUTOUT"]="0";
    fi
    if [[ "$v" == *_abRPs* ]] ; then
	flags["ABLATION_RECPAR_CUTOUT"]="0";
    fi
    if [[ "$v" == *_abRPn* ]] ; then
	flags["ABLATION_RECPAR_CUTOUT"]="1";
    fi
    if [[ "$v" == *_abRPa* ]] ; then
	flags["ABLATION_RECPAR_CUTOUT"]="2";
    fi
    if [[ "$v" == *_abxpv* ]] ; then
	flags["ABLATION_BITCONSTRUCT_XP_VEC"]="1";
    fi
    if [[ "$v" == *_fopt1* ]] ; then
	flags["FURTHER_OPTIMIZATION"]="1";
    fi
    if [[ "$v" == *_clobber* ]] ; then
	flags["MEMORY_CLOBBER_WORKAROUND"]=1;
    fi
    if [[ "$v" == *_par0* ]] ; then
	flags["PAR_LOOP"]=0;
    fi
    if [[ "$v" == *_par1* ]] ; then
	flags["PAR_LOOP"]=1;
    fi
    if [[ "$v" == *_par2* ]] ; then
	flags["PAR_LOOP"]=2;
    fi
    if [[ "$v" == *_par4* ]] ; then
	flags["PAR_LOOP"]=4;
    fi
    if [[ "$v" == *_par5* ]] ; then
	flags["PAR_LOOP"]=5;
    fi
    if [[ "$v" == *_par6* ]] ; then
	flags["PAR_LOOP"]=6;
    fi
    if [[ "$v" == *_par7* ]] ; then
	flags["PAR_LOOP"]=7;
    fi
    if [[ "$v" == *_par8* ]] ; then
	flags["PAR_LOOP"]=8;
    fi
    if [[ "$v" == *_top0* ]] ; then
	flags["ABLATION_GENERIC_TOP"]=0;
    fi
    if [[ "$v" == *_top1* ]] ; then
	flags["ABLATION_GENERIC_TOP"]=1;
    fi
    if [[ "$v" == *_top2* ]] ; then
	flags["ABLATION_GENERIC_TOP"]=2;
    fi
    if [[ "$v" == *_BQ* ]] ; then
	flags["BLOCKED_THRESHOLD_SEQUENTIAL_PBITS"]=`echo "${v}_" | sed -e 's/^.*_BQ\([0-9][0-9]*\)_.*$/\1.0/'`;
    fi
    if [[ "$v" == *_Bq* ]] ; then
	flags["BLOCKED_THRESHOLD_SEQUENTIAL"]=`echo "${v}_" | sed -e 's/^.*_Bq\([0-9][0-9]*\)_.*$/\1.0/'`;
    fi
    if [[ "$v" == *_Bd* ]] ; then
	flags["BLOCKED_THRESHOLD_DENSITY"]=`echo "${v}_" | sed -e 's/^.*_Bd\([0-9][0-9]*\)_.*$/\1.0/'`;
    fi
    if [[ "$v" == *_DQ* ]] ; then
	flags["DENSE_THRESHOLD_SEQUENTIAL_BITS"]=`echo "${v}_" | sed -e 's/^.*_DQ\([0-9][0-9]*\)_.*$/\1.0/'`;
    fi
    if [[ "$v" == *_Dq* ]] ; then
	flags["DENSE_THRESHOLD_SEQUENTIAL"]=`echo "${v}_" | sed -e 's/^.*_Dq\([0-9][0-9]*\)_.*$/\1.0/'`;
    fi
    if [[ "$v" == *_Dd* ]] ; then
	flags["DENSE_THRESHOLD_DENSITY"]=`echo "${v}_" | sed -e 's/^.*_Dd\([0-9][0-9]*\)_.*$/\1.0/'`;
    fi
    if [[ "$v" == *_scut* ]] ; then
	flags["TUNABLE_SMALL_AVOID_CUTOUT_LEAF"]=`echo "${v}_" | sed -e 's/^.*_scut\([0-9][0-9]*\)_.*$/\1/'`;
    fi
    if [[ "$v" == *_vl* ]] ; then
	flags["MAX_VL"]=`echo "${v}_" | sed -e 's/^.*_vl\([0-9][0-9]*\)_.*$/\1/'`;
	flags["GRAPTOR_DEGREE_BITS"]=${flags["MAX_VL"]};
    fi

    for key in ${!flags[@]}; do echo -D$key=${flags[$key]} ; done

    echo -DGRAPHTYPE=VEBOGraptorT
}

