#!/bin/bash

LDPATH=$HOME/graptor/lib

WEIGHTS=none

LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$LDPATH"
export LD_LIBRARY_PATH

module load compilers/gcc/10.3.0
. $HOME/oneTBB-2020.2/build/linux_intel64_gcc_cc7.2.0_libc2.17_kernel3.10.0_release/tbbvars.sh intel64 linux auto_tbbroot

hostname

. $HOME/graptor/exec-vl/graptor_tree.sh

# args 1:bench 2:graph-path 3:graph-name 4:threads 5:commit 6:round
function run()
{
    local prog=$1
    local graph_path=$2
    local graph_name=$3
    local threads=$4
    local commit=$5
    local round=$6

    local commit_full=$commit

    if [[ "$commit" == *_numa* ]] ; then
	commit=`echo $commit | cut -d_ -f1`
    fi

    bindir=bin_epyc_$commit
    arg=""

    local outdir=./graptor_epyc/$commit_full
    [ -d $outdir ] || mkdir -p $outdir
    local file=$outdir/output.${commit}.t${threads}.r${round}.${prog}.${graph_name} 

    if [ ! -f $file ] ; then
	echo "./${bindir}/${prog} -n $threads -p -f ${graph_path} > $file"
	./${bindir}/${prog} -n $threads -p -f ${graph_path} > $file 2>&1
    fi
}

# one 1:bench 2:graph-path 3:graph-name 4:threads 5:commit 6:round
function one() {
    run ${1} $2 $3 $4 $5 $6
}

# args: graph start-vertex
function for_gg()
{
    mtxmkt_graph $1 $2 $WEIGHTS
    local graph_path=`mtxmkt_graph_path $1 $2 $WEIGHTS`

    local start=$3
    local commit=$4

    for threads in 1 ; do
	for r in `seq 0 9` ; do
	    one mce_profiling $graph_path ${1}_${2} $threads $commit $r
	done
    done

    
    for threads in 1 8 16 32 64 128 ; do
	for r in `seq 0 9` ; do
	    one mce $graph_path ${1}_${2} $threads $commit $r
	done
	#one mce_nohopscotch $graph_path ${1}_${2} $threads $commit 0
    done
}

true
