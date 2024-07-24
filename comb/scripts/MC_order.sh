#!/bin/bash

#what="filter:"
#what="Enumeration:"
#what="Calculting coreness:"
what="Completed MC in"
whatn=$((`echo "$what" | sed -e 's/[^ ]//g' | wc -c` + 1))

function get_avg() {
    local commit=$1
    local threads=$2
    local bench=$3
    local version=$4
    local opts=$5
    local graph=$6
    local part=$7
    local dir=$8
    local arch=$9
    local heur=${10}

    local sum=0
    local count=0
    local fail=0
    
    #echo graptor_$arch/$dir/output.${commit}.t${threads}.c${part}.r0.h${heur}.${bench}_${version}_${opts}.${graph}_undir 
    for file in `ls -1 graptor_$arch/$dir/output.${commit}.t${threads}.c${part}.r*.h${heur}.${bench}_${version}_${opts}.${graph}_undir 2> /dev/null` ; do
	if [ -e $file ] ; then
	    if grep FAIL $file > /dev/null 2>&1 ; then
		fail=1
	    else
		#local val=`( grep "Enumeration:" $file 2> /dev/null || echo Enumeration: ABORT ) | cut -d' ' -f2`
		local val=`( grep "$what" $file 2> /dev/null || echo $what: ABORT ) | cut -d' ' -f$whatn`
		if [ x$val != xABORT ] ; then
		    sum=`echo "scale=4; $sum + $val" | bc`
		    count=$(( $count + 1 ))
		else
		    fail=1
		fi
	    fi
	else
	    fail=2
	fi
    done

    if [ $fail -eq 0 -a $count -ne 0 ] ; then
	echo "scale=4; $sum / $count" | bc
    elif [ $fail -eq 1 ] ; then
	echo FAIL
    else
	echo ABSENT
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
    local dir=$8
    local arch=$9
    local heur=${10}

    get_avg "$commit" "$threads" "$bench" "$version" "$opts" "$graph" "$part" "$dir" "$arch" "$heur"
}

function mce() {
    local vl=vl$1
    local threads=$2
    local commit=$3
    local part=$4
    local arch=$5
    local dir=$6
    local heur=$7
    local graphs="CAroad GG-NE-bio-heart GG-NE-bio-neuron LiveJournal M87127560 RMAT27 USAroad Yahoo_mem Yahoo_web bio-HS-CX bio-WormNet-v3 bio-human-gene1 bio-human-gene2 bio-mouse-gene btc-2009 cit-patents clueweb09 dblp2012 dimacs flickr friendster_full higgs-twitter hollywood2009 hudong it-2004 keller4 keller5 keller6 orkut pokec sinaweibo sx-stackoverflow twitter_xl uk-2005 uk_union_06 warwiki webcc wiki-talk wiki-topcats"

    if [ x$dir = x ] ; then
	dir=$commit
    fi

    local variants=""
    if [ $heur == 1 ] ; then
	for s in `seq 0 5` ; do
	    for t in `seq 0 3` ; do
		variants="$variants itrim_sort${s}_trav${t}_$vl"
	    done
	done
    else
	for s in `seq 4 5` ; do
	    for t in `seq 0 3` ; do
		variants="$variants itrim_sort${s}_trav${t}_$vl"
	    done
	done
    fi

    echo "VL=$vl threads=$threads commit=$commit part=$part"
    (
	echo -ne "\"\" "
	echo "$variants" | tr ' ' "\n" # variants starts with a space
	for graph in $graphs ; do
	    echo $graph
	    for variant in $variants ; do
		one $commit $threads MC "" ${variant} ${graph} ${part} ${dir} ${arch} ${heur}
	    done
	done
    ) | paste - $(echo $variants | sed -e 's/\b[a-zA-Z0-9_]*\b/-/g')

    echo
}

mce 16 128 dc3fc239 1024 avx512 dc3fc239 1
mce 16 128 dc3fc239 1024 avx512 dc3fc239 2
