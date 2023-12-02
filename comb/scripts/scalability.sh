#!/bin/bash

function get_avg() {
    local commit=$1
    local threads=$2
    local bench=$3
    local version=$4
    local opts=$5
    local graph=$6
    local part=$7

    local sum=0
    local count=0
    local fail=0
    
    for file in `ls -1 graptor_avx512/$commit/output.${commit}.t${threads}.c${part}.r*.${bench}_${version}_${opts}.${graph} 2> /dev/null` ; do
	if [ -e $file ] ; then
	    if grep FAIL $file > /dev/null 2>&1 ; then
		fail=1
	    else
		local val=`( grep "Enumeration:" $file 2> /dev/null || echo Enumeration: ABORT ) | cut -d' ' -f2`
		if [ x$val != xABORT ] ; then
		    sum=`bc <<< "scale=4; $sum + $val"`
		    count=`bc <<< "$count + 1"`
		else
		    fail=1
		fi
	    fi
	else
	    fail=2
	fi
    done

    if [ $fail -eq 0 -a $count -ne 0 ] ; then
	bc <<< "scale=4; $sum / $count"
    elif [ $fail -eq 1 ] ; then
	echo FAIL
    else
	echo ABSENT
	#echo ABSENT-graptor_avx512/$commit/output.${commit}.t${threads}.c${part}.r*.${bench}_${version}_${opts}.${graph}
    fi
}

function one() {
    local commit=$1
    local threads=$2
    local bench=$3
    local version=$4
    local opts=$5
    local graph=$6
    local part=$7

    get_avg "$commit" "$threads" "$bench" "$version" "$opts" "$graph" "$part"
}

function mce() {
    local vl=$1
    local part=$2
    local commit=$3
    local graphs="USAroad_undir CAroad_undir wiki-talk_undir cit-patents_undir sx-stackoverflow_undir Yahoo_mem_undir warwiki_undir pokec_undir wiki-topcats_undir friendster_full_undir bio-HS-CX_undir higgs-twitter_undir orkut_undir bio-WormNet-v3_undir"

    local threads="1 8 16 32 64 128"
    echo "VL=$vl commit=$commit part=$part"
    (
	echo -ne "\"\" "
	echo " $threads" | tr ' ' "\n"
	for graph in $graphs ; do
	    echo $graph
	    for thr in $threads ; do
		one $commit $thr MCE "" vl$vl ${graph} ${part}
	    done
	done
    ) | paste - $(echo $threads | sed -e 's/\b[a-zA-Z0-9_]*\b/-/g')

    echo
}

#mce 8 512 3ae3ceec
#mce 8 512 c5e03980
#mce 8 1024 0f66b914
mce 16 1024 0f20dd74

