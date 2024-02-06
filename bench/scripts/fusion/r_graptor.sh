#!/bin/bash

what="Average "
#what="Removed self-edges"
#what="Calculating coreness"
#what="Determining sort order"
#what="Remapping graph"
#what="Building hashed graph"
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
    
    #echo graptor_$arch/$dir/output.${commit}.t${threads}.c${part}.r0.${bench}_${version}_${opts}.${graph} 
    for file in `ls -1 graptor_$arch/$dir/output.${commit}.t${threads}.c${part}.${bench}_${version}_${opts}.${graph} 2> /dev/null` ; do
	if [ -e $file ] ; then
	    if grep FAIL $file > /dev/null 2>&1 ; then
		fail=1
	    else
		#local val=`( grep "Enumeration:" $file 2> /dev/null || echo Enumeration: ABORT ) | cut -d' ' -f2`
		local val=`( grep "$what:" $file 2> /dev/null || echo $what: ABORT ) | cut -d' ' -f$whatn`
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

    get_avg "$commit" "$threads" "$bench" "$version" "$opts" "$graph" "$part" "$dir" "$arch"
}

function presults() {
    local vl=vl$1
    local threads=$2
    local commit=$3
    local part=$4
    local arch=$5
    local dir=$6
    local graphs="USAroad_undir CAroad_undir Yahoo_mem_undir warwiki_undir pokec_undir friendster_full_undir orkut_undir LiveJournal_undir clueweb09_undir dimacs_undir twitter_xl_undir uk_union_06_undir webcc_undir"

    if [ x$dir = x ] ; then
	dir=$commit
    fi

    local variants="BFSv-none BFSLVL_narrow-nofusion BFSLVL_narrow-fusion BFSLVLv-nofusion BFSLVLv-fusion BFSBool-uvvid_bitf BFSBool-uvbit_bitf CCv-nofusion CCv-fusion GC_JP_fusion-nofusion GC_JP_fusion-fusion GC_JP_fusion-llf_fusion GC_gm3p-vl16 GC_gm3p_v2-vl16 KC_bucket-nofusion KC_bucket-fusion"

    echo "VL=$vl threads=$threads commit=$commit part=$part"
    (
	echo -ne "\"\" "
	echo " $variants" | tr ' ' "\n"
	for graph in $graphs ; do
	    echo $graph
	    for variant in $variants ; do
		local vprog="$(echo $variant | cut -d- -f1)"
		local vvariant="$(echo $variant | cut -d- -f2-)_"
		if [ $vvariant = none_ ] ; then vvariant="" ; fi
		one $commit $threads $vprog "" ${vvariant}${vl} ${graph} ${part} ${dir} ${arch}
	    done
	done
    ) | paste - $(echo "$variants" | sed -e 's/\b[a-zA-Z0-9_-]*\b/ - /g')

    echo
}

presults 16 128 d5cbb0b4 1024 avx512 d5cbb0b4 
