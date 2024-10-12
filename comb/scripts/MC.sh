#!/bin/bash

#what="heuristic 2:"
#what="phase 1 (one vertex per degeneracy):"
#what="filter"
#what="early pruning"
what="Enumeration"
#what="Calculting coreness"
#what="Calculating degeneracy time:"
#what="Constructing remap info:"
#what="Building hashed graph:"
#what="Completed MC in"
#what="Completed search in"
#what="filter0"
#what="filter1"
#what="filter2"
whatn=$((`echo "$what" | sed -e 's/[^ ]//g' | wc -c` + 1))

#graphs="CAroad LiveJournal USAroad Yahoo_mem bio-HS-CX bio-WormNet-v3 cit-patents dblp2012 flickr friendster_full higgs-twitter hollywood2009 hudong it-2004 keller4 orkut pokec sinaweibo sx-stackoverflow warwiki webcc wiki-talk wiki-topcats uk-2005 dimacs bio-human-gene2 mawi ppminer"

graphs="USAroad sinaweibo friendster_full webcc uk_union_06 dimacs CAroad sx-stackoverflow wiki-talk cit-patents LiveJournal hudong flickr Yahoo_mem warwiki wiki-topcats pokec dblp2012 orkut it-2004 hollywood2009 higgs-twitter uk-2005 bio-WormNet-v3 bio-HS-CX bio-mouse-gene bio-human-gene1 bio-human-gene2"

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
    local density=${11}
    local early=${12}
    local what=${13}

    local whatn=$((`echo "$what" | sed -e 's/[^ ]//g' | wc -c` + 1))

    local sum=0
    local sumsq=0
    local count=0
    local fail=0
    
    #echo graptor_$arch/$dir/output.${commit}.t${threads}.c${part}.r0.${bench}_${version}_${opts}.${graph}_undir 
    for file in `ls -1 graptor_$arch/$dir/output.${commit}.t${threads}.c${part}.r[0-9].h${heur}.d${density}.ep${early}.${bench}_${version}_${opts}.${graph}_undir 2> /dev/null` ; do
	if [ -e $file ] ; then
	    if grep FAIL $file > /dev/null 2>&1 ; then
		fail=1
	    else
		#local val=`( grep "Enumeration:" $file 2> /dev/null || echo Enumeration: ABORT ) | cut -d' ' -f2`
		local val=`( grep "$what" $file 2> /dev/null || echo $what ABORT ) | cut -d' ' -f$whatn`
		if [ x$val != xABORT ] ; then
		    sum=`echo | perl -ne "END { printf \"%g\n\", ($sum+$val); }"`
		    sumsq=`echo | perl -ne "END { printf \"%g\n\", ($sumsq+$val*$val); }"`
		    #sum=`echo "scale=4; $sum + $val" | bc`
		    count=$(( $count + 1 ))
		else
		    fail=3
		fi
	    fi
	else
	    fail=2
	fi
    done

    if [ $fail -eq 0 -a $count -ne 0 ] ; then
	#echo $count
    	echo | perl -ne "END { printf \"%g\n\", ($sum/$count); }"
    	#echo | perl -ne "END { printf \"%g\n\", ($sumsq-($sum*$sum)/$count)/($count-1); }"
	#echo ${sum}:${count}
	#echo "scale=4; $sum / $count" | bc
    elif [ $fail -eq 1 ] ; then
	echo FAIL
    elif [ $fail -eq 3 ] ; then
	echo NOMETRIC
    else
	echo ABSENT
    fi
}

function get_stats() {
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
    local fail=2
    
    for file in graptor_$arch/$dir/output.${commit}.t${threads}.c${part}.r0.h2.d0_1.ep${early}.${bench}_${version}_${opts}.${graph}_undir ; do
	if [ -e $file ] ; then
	    if grep FAIL $file > /dev/null 2>&1 ; then
		fail=1
	    else
		# Expect, e.g.: Undirected graph: n=456627 m=25016826 density=0.00011998 dmax=51386 davg=54.7861
		local val=`( grep "^Undirected graph:" $file 2> /dev/null || echo nada: ABORT ) | cut -d: -f2-`
		if [ "x$val" != xABORT ] ; then
		    echo $val | tr -d [a-zA-Z=] | tr -s ' ' "\n" | sed -e 's/\([0-9]\)-\([0-9]\)/\1e-\2/g'
		else
		    echo V E d dmax davg
		fi
		( grep "^degeneracy=" $file 2> /dev/null || echo nada=ABORT ) | cut -d= -f2
		( grep "^Maximum clique size:" $file 2> /dev/null || echo m c s: ABORT ) | cut -d' ' -f4
		perl -ne 'BEGIN { $cl="FAIL"; } END { print "$cl\n"; } chomp; $cl=$1 if m/^max_clique: ([0-9]*) /; do { exit 0; } if m/^early pruning:/;' < $file 2> /dev/null
		perl -ne 'BEGIN { $cl="FAIL"; } END { print "$cl\n"; } chomp; $cl=$1 if m/^max_clique: ([0-9]*) /; do { exit 0; } if m/^heuristic 2:/;' < $file 2> /dev/null
		fail=0
	    fi
	else
	    fail=2
	fi
    done

    if [ $fail -eq 0 ] ; then
	true # already printed
    elif [ $fail -eq 1 ] ; then
	echo FAIL
    else
	seq 9 | xargs -I{} echo ABSENT
    fi
}

function get_maymust() {
    local commit=$1
    local threads=$2
    local bench=$3
    local version=$4
    local opts=$5
    local graph=$6
    local part=$7
    local dir=$8
    local arch=$9

    local fail=2
    
    for file in graptor_$arch/$dir/output.${commit}.t${threads}.c${part}.r0.h2.d0_1.ep128.${bench}_${version}_${opts}.${graph}_undir ; do
	if [ -e $file ] ; then
	    if grep FAIL $file > /dev/null 2>&1 ; then
		fail=1
	    else
		local val=`( grep "^May analyse:" $file 2> /dev/null || echo nada: ABORT ) | cut -d: -f2-`
		if [ "x$val" != xABORT ] ; then
		    echo "$val" | cut -d' ' -f3,5,7 | tr ' ' "\n"
		else
		    echo mayv maye mayd
		fi
		val=`( grep "^Must analyse:" $file 2> /dev/null || echo nada: ABORT ) | cut -d: -f2-`
		if [ "x$val" != xABORT ] ; then
		    echo "$val" | cut -d' ' -f3,5,7 | tr ' ' "\n"
		else
		    echo mustv muste mustd
		fi
		val=`( grep "^May attached:" $file 2> /dev/null || echo nada: ABORT ) | cut -d: -f2-`
		if [ "x$val" != xABORT ] ; then
		    echo "$val" | cut -d' ' -f3,5,7 | tr ' ' "\n"
		else
		    echo mayav mayae mayaf
		fi
		fail=0
	    fi
	else
	    fail=2
	fi
    done

    if [ $fail -eq 0 ] ; then
	true # already printed
    elif [ $fail -eq 1 ] ; then
	echo FAIL
    else
	seq 9 | xargs -I{} echo ABSENT
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
    local density=${11}
    local early=${12}

    get_avg "$commit" "$threads" "$bench" "$version" "$opts" "$graph" "$part" "$dir" "$arch" "$heur" "$density" "$early" "$what"
}

function get_steps() {
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
    local density=${11}
    local early=${12}

    get_avg "$commit" "$threads" "$bench" "$version" "$opts" "$graph" "$part" "$dir" "$arch" "$heur" "$density" "$early" "early pruning:"

    get_avg "$commit" "$threads" "$bench" "$version" "$opts" "$graph" "$part" "$dir" "$arch" "$heur" "$density" "$early" "Constructing remap info:"

    get_avg "$commit" "$threads" "$bench" "$version" "$opts" "$graph" "$part" "$dir" "$arch" "$heur" "$density" "$early" "Building hashed graph:"

    get_avg "$commit" "$threads" "$bench" "$version" "$opts" "$graph" "$part" "$dir" "$arch" "$heur" "$density" "$early" "heuristic 2:"

    get_avg "$commit" "$threads" "$bench" "$version" "$opts" "$graph" "$part" "$dir" "$arch" "$heur" "$density" "$early" "Enumeration:"

    get_avg "$commit" "$threads" "$bench" "$version" "$opts" "$graph" "$part" "$dir" "$arch" "$heur" "$density" "$early" "Completed MC in"
}

function mce() {
    local vl=vl$1
    local threads=$2
    local commit=$3
    local part=$4
    local arch=$5
    local dir=$6

    if [ x$dir = x ] ; then
	dir=$commit
    fi

    #local variants="ins0_$vl ins1_$vl ins2_$vl ins3_$vl ins4_$vl ins5_$vl ins6_$vl ins7_$vl ins8_$vl itrim_ins0_$vl itrim_ins1_$vl itrim_ins2_$vl itrim_ins3_$vl itrim_ins4_$vl itrim_ins5_$vl itrim_ins6_$vl itrim_ins7_$vl itrim_ins8_$vl"
    #local variants="itrim_sort5_trav1_$vl itrim_sort5_trav1_vccc_$vl itrim_sort5_trav1_nopivc_$vl itrim_sort5_trav1_nopivd_$vl itrim_sort5_trav1_nopivc_nopivd_$vl"
    #local variants="itrim_sort4_trav0_$vl itrim_sort4_trav1_$vl itrim_sort4_trav3_$vl itrim_sort5_trav1_$vl itrim_sort4_trav3_onesided_$vl itrim_sort4_trav3_vccc_$vl itrim_sort4_trav3_nopivc_nopivd_$vl itrim_sort4_trav3_${vl}_noavx512"
    #local variants="itrim_sort4_trav3_$vl itrim_sort4_trav3_mcins1_$vl itrim_sort4_trav3_hopscotch1_$vl itrim_sort4_trav3_hopscotch1_mcins1_$vl itrim_sort4_trav3_ld1_$vl itrim_sort4_trav3_ld2_$vl itrim_sort4_trav3_ld3_$vl itrim_sort4_trav3_ld1_hopscotch1_$vl itrim_sort4_trav3_ld2_hopscotch1_$vl itrim_sort4_trav3_ld3_hopscotch1_$vl"
    local variants="itrim_sort4_trav3_geabove_$vl itrim_sort4_trav3_geabove_noadvins_$vl itrim_sort4_trav3_geabove_nolazy_$vl itrim_sort4_trav3_geabove_nolazy_nohopscotch_$vl itrim_sort2_trav3_geabove_$vl itrim_sort4_trav3_geabove_nolhf_$vl itrim_sort4_trav3_$vl"
    #local variants="itrim_sort4_trav3_$vl"

    echo "VL=$vl threads=$threads commit=$commit part=$part"
    (
	echo -ne "\"\" "
	echo " $variants" | tr ' ' "\n"
	for graph in $graphs ; do
	    echo $graph
	    for variant in $variants ; do
		one $commit $threads MC "" ${variant} ${graph} ${part} ${dir} ${arch} 2 0_1 128
	    done
	done
    ) | paste - $(echo $variants | sed -e 's/\b[a-zA-Z0-9_]*\b/-/g')

    echo
}

function scalability() {
    local vl=$1
    local part=$2
    local commit=$3
    local arch=$4
    local dir=$commit

    local threads="1 8 16 32 64 128"
    local variant="itrim_sort4_trav3_geabove_vl${vl}"
    echo "VL=$vl commit=$commit part=$part"
    (
        echo -ne "\"\" "
        echo " $threads" | tr ' ' "\n"
        for graph in $graphs ; do
            echo $graph
            for thr in $threads ; do
                one $commit $thr MC "" ${variant} ${graph} ${part} ${dir} ${arch} 2 0_1 128
            done
        done
    ) | paste - $(echo $threads | sed -e 's/\b[a-zA-Z0-9_]*\b/-/g')

    echo
}

function lazy() {
    local vl=$1
    local threads=$2
    local part=$3
    local commit=$4
    local arch=$5
    local dir=$commit

    local lazy="0 -1 -2"
    local variant="itrim_sort4_trav3_geabove_vl${vl}"
    echo "VL=$vl commit=$commit part=$part"
    (
        echo -ne "\"\" "
        echo " $lazy" | tr ' ' "\n"
        for graph in $graphs ; do
            echo $graph
            for lz in $lazy ; do
		local ep="128.lz$lz"
		if [ $lz -eq -1 ] ; then ep=128 ; fi
                one $commit $threads MC "" ${variant} ${graph} ${part} ${dir} ${arch} 2 0_1 $ep
            done
        done
    ) | paste - $(echo $lazy | sed -e 's/-//g;s/\b[a-zA-Z0-9_\.+]*\b/-/g')

    echo
}

function vc() {
    local vl=$1
    local threads=$2
    local part=$3
    local commit=$4
    local arch=$5
    local dir=$commit

    #local density="0.5 0.7 0.9 0.95 1"
    local density="0.1 0.3 0.5 0.7 0.9 1.0"
    local variant="itrim_sort4_trav3_geabove_vl${vl}"
    echo "VL=$vl commit=$commit part=$part"
    (
        echo -ne "\"\" "
        echo " $density 0.1.ep0" | tr ' ' "\n"
        for graph in $graphs ; do
            echo $graph
            for d in $density ; do
		local de=`echo $d | tr . _`
                one $commit $threads MC "" ${variant} ${graph} ${part} ${dir} ${arch} 2 $de 128
            done
            for d in 0.1 ; do
		local de=`echo $d | tr . _`
                one $commit $threads MC "" ${variant} ${graph} ${part} ${dir} ${arch} 2 $de 0
            done
        done
    ) | paste - $(echo $density | sed -e 's/\b[a-zA-Z0-9_\.+]*\b/-/g') -

    echo
}

function steps() {
    local vl=$1
    local threads=$2
    local part=$3
    local commit=$4
    local arch=$5
    local dir=$commit

    local metrics="early remap hashed heuristic enumeration complete"
    local variant="itrim_sort4_trav3_geabove_vl${vl}"
    echo "VL=$vl commit=$commit part=$part"
    (
        echo -ne "\"\" "
        echo " $metrics" | tr ' ' "\n"
        for graph in $graphs ; do
            echo $graph
            get_steps $commit $threads MC "" ${variant} ${graph} ${part} ${dir} ${arch} 2 0_1 128
        done
    ) | paste - $(echo $metrics | sed -e 's/\b[a-zA-Z0-9_\.+]*\b/-/g')

    echo
}


function stats() {
    local vl=vl$1
    local threads=$2
    local commit=$3
    local part=$4
    local arch=$5
    local dir=$6
    local early=$7

    local variant="itrim_sort4_trav3_geabove_$vl"
    local stats="vertices edges density maxdegree avgdegree degeneracy maxclique earlyclique heurclique"

    echo "VL=$vl threads=$threads commit=$commit part=$part early=$early"
    (
	echo -ne "\"\" "
	echo " $stats" | tr ' ' "\n"
	for graph in $graphs ; do
	    echo $graph
	    get_stats $commit $threads MC "" ${variant} ${graph} ${part} ${dir} ${arch} ${early}
	done
    ) | paste - $(echo $stats | sed -e 's/\b[a-zA-Z0-9_]*\b/-/g')

    echo
}

function maymust() {
    local vl=vl$1
    local threads=$2
    local commit=$3
    local part=$4
    local arch=$5
    local dir=$6

    if [ x$dir = x ] ; then
	dir=$commit
    fi

    local variant="itrim_sort4_trav3_geabove_$vl"
    local stats="may_vertices may_edges may_density must_vertices must_edges must_density may_attached_vertices may_attached_edges may_attached_useful"

    echo "VL=$vl threads=$threads commit=$commit part=$part"
    (
	echo -ne "\"\" "
	echo " $stats" | tr ' ' "\n"
	for graph in $graphs ; do
	    echo $graph
	    get_maymust $commit $threads MC "" ${variant} ${graph} ${part} ${dir} ${arch}
	done
    ) | paste - $(echo $stats | sed -e 's/\b[a-zA-Z0-9_]*\b/-/g')

    echo
}


#commit=b3d0aed7
#commit=9319a5a3
#commit=d0d47f7d
#commit=093ea1c1
#commit=66356e5c
commit=28c5ddae

#mce 16 128 $commit 1024 avx512 $commit
#mce 16 1 $commit 1024 avx512 $commit

scalability 16 1024 $commit avx512

#vc 16 128 1024 $commit avx512

#lazy 16 128 1024 $commit avx512

#steps 16 128 1024 $commit avx512

#stats 16 128 $commit 1024 avx512 $commit 128
#stats 16 128 $commit 1024 avx512 $commit 0

#maymust 16 128 $commit 1024 avx512 $commit
