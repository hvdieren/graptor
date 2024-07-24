#!/bin/bash

#what="filter"
what="Enumeration"
#what="Calculting coreness"
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

    local sum=0
    local count=0
    local fail=0
    
    #echo graptor_$arch/$dir/output.${commit}.t${threads}.c${part}.r0.${bench}_${version}_${opts}.${graph}_undir 
    for file in `ls -1 graptor_$arch/$dir/output.${commit}.t${threads}.c${part}.r*.${bench}_${version}_${opts}.${graph}_undir 2> /dev/null` ; do
	if [ -e $file ] ; then
	    if grep FAIL $file > /dev/null 2>&1 ; then
		fail=1
	    else
		#local val=`( grep "Enumeration:" $file 2> /dev/null || echo Enumeration: ABORT ) | cut -d' ' -f2`
		local val=`( grep "$what:" $file 2> /dev/null || echo $what: ABORT ) | cut -d' ' -f$whatn`
		if [ x$val != xABORT ] ; then
		    sum=`echo | perl -ne "END { printf \"%f\n\", ($sum+$val); }"`
		    #sum=`echo "scale=4; $sum + $val" | bc`
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
    	echo | perl -ne "END { printf \"%f\n\", ($sum/$count); }"
	#echo ${sum}:${count}
	#echo "scale=4; $sum / $count" | bc
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

    get_avg "$commit" "$threads" "$bench" "$version" "$opts" "$graph" "$part" "$dir" "$arch"
}

function mce() {
    local vl=vl$1
    local threads=$2
    local commit=$3
    local part=$4
    local arch=$5
    local dir=$6
    local graphs="CAroad GG-NE-bio-heart GG-NE-bio-neuron LiveJournal M87127560 RMAT27 USAroad Yahoo_mem Yahoo_web bio-HS-CX bio-WormNet-v3 bio-human-gene1 bio-human-gene2 bio-mouse-gene btc-2009 cit-patents clueweb09 dblp2012 dimacs flickr friendster_full higgs-twitter hollywood2009 hudong it-2004 keller4 keller5 keller6 orkut pokec sinaweibo sx-stackoverflow twitter_xl uk-2005 uk_union_06 warwiki webcc wiki-talk wiki-topcats"

    if [ x$dir = x ] ; then
	dir=$commit
    fi

    #local variants="ins0_$vl ins1_$vl ins2_$vl ins3_$vl ins4_$vl ins5_$vl ins6_$vl ins7_$vl ins8_$vl itrim_ins0_$vl itrim_ins1_$vl itrim_ins2_$vl itrim_ins3_$vl itrim_ins4_$vl itrim_ins5_$vl itrim_ins6_$vl itrim_ins7_$vl itrim_ins8_$vl"
    local variants="itrim_sort5_trav1_$vl itrim_sort5_trav1_vcmono_$vl itrim_sort5_trav1_pivc_$vl itrim_sort5_trav1_pivd_$vl itrim_sort5_trav1_pivc_pivd_$vl"

    echo "VL=$vl threads=$threads commit=$commit part=$part"
    (
	echo -ne "\"\" "
	echo " $variants" | tr ' ' "\n"
	for graph in $graphs ; do
	    echo $graph
	    for variant in $variants ; do
		one $commit $threads MC "" ${variant} ${graph} ${part} ${dir} ${arch}
	    done
	done
    ) | paste - $(echo $variants | sed -e 's/\b[a-zA-Z0-9_]*\b/-/g')

    echo
}

#mce 16 1 700306b4 1024 avx512 700306b4
#mce 8_noavx512f 1 700306b4 1024 avx512 700306b4

mce 16 128 4d1bb30b 1024 avx512 4d1bb30b
