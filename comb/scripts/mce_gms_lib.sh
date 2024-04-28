#!/bin/bash

# For custom gcc build
. $HOME/tools/gcc-13-install/bin/load.sh

TREE_ROOT=/var/shared/projects/asap/graphs/adj/realworld/recent

TOOLSPATH=/home/hvandierendonck/research/ligra-partition/build-sapphirerapids/tools
CILK_LIB=/home/hvandierendonck/research/ligra-partition/build-sapphirerapids/external/Build/cilkrts_project/lib/libcilkrts.so.5

WEIGHTS=none

hostname

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

    bindir=bin_avx512_$commit
    arg=""

    local outdir=./graptor_avx512/$commit_full
    [ -d $outdir ] || mkdir -p $outdir
    local file=$outdir/output.${commit}.t${threads}.r${round}.${prog}.${graph_name} 

    if [ ! -f $file ] ; then
	echo "./${bindir}/${prog} -t $threads -n ${round} -f ${graph_path} > $file"
	./${bindir}/${prog} -t $threads -n ${round} -f ${graph_path} > $file 2>&1
    fi
}

# one 1:bench 2:graph-path 3:graph-name 4:threads 5:commit 6:round
function one() {
    run ${1} $2 $3 $4 $5 $6
}

# args: graph start-vertex
function for_gg()
{
    local start=$3
    local commit=$4

    local graph_path=`/home/hvandierendonck/research/ligra-partition/tools/graptor_tree.sh -l $TREE_ROOT materialise gapbs $1 $2 $WEIGHTS`

    for threads in 1 8 16 32 64 128 ; do
	one maximal_clique_enum_bron_kerbosch $graph_path ${1}_${2} $threads $commit 10
    done
}

true
