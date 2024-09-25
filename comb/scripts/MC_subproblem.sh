#!/bin/bash

function get_avg() {
    local commit=$1
    local threads=$2
    local bench=$3
    local version=$4
    local opts=$5
    local graph=$6
    local subprob=$7
    local dir=$8

    local avg_build=0;
    local avg_proc=0;
    local avg_call=0;
    local count=0
    local fail=0

    for file in `ls -1 graptor_avx512/$dir/output.${commit}.t${threads}.c1024.r*.h2.d0_1.${bench}_${version}_${opts}.${graph}_undir 2> /dev/null` ; do
	if [ -e $file ] ; then
	    if grep FAIL $file > /dev/null 2>&1 
	       -o ! grep "Enumeration:" $file > /dev/null 2>&1 ; then
		fail=1
	    else
		local val=`grep "^$subprob:" $file 2> /dev/null | cut -d: -f2- | cut -d' ' -f2,5,13 | sed -e s/nan/0/`
		avg_proc=`echo $val | cut -d' ' -f1 | perl -ne 'chomp; my $a='$avg_proc'+$_; print "$a\n";'`
		avg_build=`echo $val | cut -d' ' -f3 | perl -ne 'chomp; my $a='$avg_build'+$_; print "$a\n";'`
		avg_call=`echo $val | cut -d' ' -f2 | perl -ne 'chomp; my $a='$avg_call'+$_; print "$a\n";'`
		count=$(( $count + 1 ))
	    fi
	else
	    fail=1
	fi
    done

    if [ $count -eq 0 ] ; then
    	echo ABSENT 
    	echo ABSENT 
    	echo ABSENT
    elif [ $fail -eq 0 -a $count -ne 0 ] ; then
	echo | perl -ne 'my ($a,$b,$c)=('$avg_build/$count,$avg_proc/$count,$avg_call/$count'); print "$a\n$b\n$c\n";'
    else
	echo FAIL
	echo FAIL
	echo FAIL
    fi
}

function one() {
    local commit=$1
    local threads=$2
    local bench=$3
    local version=$4
    local opts=$5
    local graph=$6
    local subprob=$7
    local dir=$8

    get_avg "$commit" "$threads" "$bench" "$version" "$opts" "$graph" "$subprob" "$dir"
}

function mce() {
    local vl=$1
    local subprob=$2
    local commit=$3
    local dir=$4
    
    local graphs="mawi USAroad sinaweibo friendster_full webcc dimacs cit-patents CAroad sx-stackoverflow wiki-talk LiveJournal hudong flickr Yahoo_mem warwiki wiki-topcats pokec dblp2012 orkut ppminer it-2004 hollywood2009 higgs-twitter uk-2005 bio-WormNet-v3 bio-HS-CX bio-human-gene2 keller4"


    local variants="itrim_sort4_trav3_vl$vl"
    echo VL=$vl
    echo $subprob
    (
	echo "\"\" "
	echo "$variants" | sed -E 's/([a-zA-Z0-9_]*)/\1-build \1-proc \1-call/g' | tr ' ' "\n"
	for graph in $graphs ; do
	    echo $graph
	    for variant in $variants ; do
		one $commit 128 MC "" ${variant} ${graph} "${subprob}" $dir
	    done
	done
    ) | paste - $(echo $variants | sed -e 's/\b[a-zA-Z0-9_]*\b/- - -/g')

    echo
}

#commit=174e5cf6
commit=9319a5a3

for algo in BK VC
do
    mce 16 "generic $algo" $commit $commit

    mce 16 "32-bit dense $algo" $commit $commit
    mce 16 "64-bit dense $algo" $commit $commit
    mce 16 "128-bit dense $algo" $commit $commit
    mce 16 "256-bit dense $algo" $commit $commit
    mce 16 "512-bit dense $algo" $commit $commit

    mce 16 "32-bit dense leaf $algo" $commit $commit
    mce 16 "64-bit dense leaf $algo" $commit $commit
    mce 16 "128-bit dense leaf $algo" $commit $commit
    mce 16 "256-bit dense leaf $algo" $commit $commit
    mce 16 "512-bit dense leaf $algo" $commit $commit
done

mce 16 filter0 $commit $commit
mce 16 filter1 $commit $commit
mce 16 filter2 $commit $commit

mce 16 heuristic $commit $commit
