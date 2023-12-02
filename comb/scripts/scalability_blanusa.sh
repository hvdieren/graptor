#!/bin/bash

function get_avg() {
    local commit=$1
    local threads=$2
    local bench=$3
    local graph=$4

    local sum=0
    local count=0
    local fail=0
    
    for file in `ls -1 graptor_epyc/$commit/output.${commit}.t${threads}.r*.${bench}.${graph} 2> /dev/null` ; do
	if [ -e $file ] ; then
	    if grep FAIL $file > /dev/null 2>&1 ; then
		fail=1
	    else
		local val=`( grep "Maximal clique enumeration time:" $file 2> /dev/null || echo Maximal clique enumeration time: ABORT ) | cut -d' ' -f5 | sed -e 's/s$//'`
		if [ x$val != xABORT ] ; then
		    sum=`dc <<< "4 k $sum $val + p"`
		    count=`dc <<< "$count 1 + p"`
		fi
	    fi
	else
	    fail=1
	fi
    done

    if [ $fail -eq 0 -a $count -ne 0 ] ; then
	dc <<< "4 k $sum $count / p"
    else
	echo ABSENT # graptor_epyc/$commit/output.${commit}.t${threads}.r*.${bench}.${graph} 
    fi
}

function one() {
    local commit=$1
    local threads=$2
    local bench=$3
    local graph=$4

    get_avg "$commit" "$threads" "$bench" "$graph"
}

function mce() {
    local vl=$1
    local graphs="USAroad_undir CAroad_undir wiki-talk_undir cit-patents_undir sx-stackoverflow_undir Yahoo_mem_undir warwiki_undir pokec_undir wiki-topcats_undir friendster_full_undir bio-HS-CX_undir higgs-twitter_undir orkut_undir bio-WormNet-v3_undir"

    local threads="1 8 16 32 64 128"
    echo "VL=$vl"
    (
	echo -ne "\"\" "
	echo " $threads" | tr ' ' "\n"
	for graph in $graphs ; do
	    echo $graph
	    for thr in $threads ; do
		one blanusa $thr mce ${graph}
	    done
	done
    ) | paste - $(echo $threads | sed -e 's/\b[a-zA-Z0-9_]*\b/-/g')

    echo
}

mce 8
