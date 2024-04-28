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
    
    for file in `ls -1 graptor_avx512/$dir/output.${commit}.t${threads}.c1024.r*.${bench}_${version}_${opts}.${graph} 2> /dev/null` ; do
	if [ -e $file ] ; then
	    if grep FAIL $file > /dev/null 2>&1 
	       -o ! grep "Enumeration:" $file > /dev/null 2>&1 ; then
		fail=1
	    else
		local val=`grep "^$subprob" $file 2> /dev/null | cut -d: -f2- | cut -d' ' -f2,5,13 | sed -e s/nan/0/`
		avg_proc=`echo $val | cut -d' ' -f1 | perl -ne 'chomp; my $a='$avg_proc'+$_; print "$a\n";'`
		avg_build=`echo $val | cut -d' ' -f3 | perl -ne 'chomp; my $a='$avg_build'+$_; print "$a\n";'`
		avg_call=`echo $val | cut -d' ' -f2 | perl -ne 'chomp; my $a='$avg_call'+$_; print "$a\n";'`
		count=`bc <<< "$count + 1"`
	    fi
	else
	    fail=1
	fi
    done

    if [ $fail -eq 0 -a $count -ne 0 ] ; then
	echo | perl -ne 'my ($a,$b,$c)=('$avg_build/$count,$avg_proc/$count,$avg_call/$count'); print "$a\n$b\n$c\n";'
    else
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
    
    local graphs="USAroad_undir CAroad_undir wiki-talk_undir cit-patents_undir sx-stackoverflow_undir Yahoo_mem_undir warwiki_undir pokec_undir wiki-topcats_undir friendster_full_undir bio-HS-CX_undir higgs-twitter_undir orkut_undir bio-WormNet-v3_undir"

    #local variants="vl8 abBc_abDc_abGc_vl8 abTy_abTd_abL_vl8 abDxph_abBxph_abAxph_vl8 abPxph_vl8 abxpv_vl8 abpdeg_vl8"
    local variants="vl$vl"
    echo VL=$vl
    echo $subprob
    (
	echo "\"\" "
	echo "$variants" | sed -E 's/([a-zA-Z0-9_]*)/\1-build \1-proc \1-call/g' | tr ' ' "\n"
	for graph in $graphs ; do
	    echo $graph
	    for variant in $variants ; do
		one $commit 1 MCE "" ${variant} ${graph} ${subprob} $dir
	    done
	done
    ) | paste - $(echo $variants | sed -e 's/\b[a-zA-Z0-9_]*\b/- - -/g')

    echo
}

mce 16 generic f4e4c4c5 f4e4c4c5
mce 16 tiny f4e4c4c5 f4e4c4c5

mce 16 32-bit f4e4c4c5 f4e4c4c5
mce 16 64-bit f4e4c4c5 f4e4c4c5
mce 16 128-bit f4e4c4c5 f4e4c4c5
mce 16 256-bit f4e4c4c5 f4e4c4c5
mce 16 512-bit f4e4c4c5 f4e4c4c5
mce 16 32,512-bit f4e4c4c5 f4e4c4c5
mce 16 64,512-bit f4e4c4c5 f4e4c4c5
mce 16 128,512-bit f4e4c4c5 f4e4c4c5
mce 16 256,512-bit f4e4c4c5 f4e4c4c5
mce 16 512,32-bit f4e4c4c5 f4e4c4c5
mce 16 512,64-bit f4e4c4c5 f4e4c4c5
mce 16 512,128-bit f4e4c4c5 f4e4c4c5
mce 16 512,256-bit f4e4c4c5 f4e4c4c5
mce 16 512,512-bit f4e4c4c5 f4e4c4c5

mce 16 leaf-32-bit f4e4c4c5 f4e4c4c5
mce 16 leaf-64-bit f4e4c4c5 f4e4c4c5
mce 16 leaf-128-bit f4e4c4c5 f4e4c4c5
mce 16 leaf-256-bit f4e4c4c5 f4e4c4c5
mce 16 leaf-512-bit f4e4c4c5 f4e4c4c5
mce 16 leaf-32,512-bit f4e4c4c5 f4e4c4c5
mce 16 leaf-64,512-bit f4e4c4c5 f4e4c4c5
mce 16 leaf-128,512-bit f4e4c4c5 f4e4c4c5
mce 16 leaf-256,512-bit f4e4c4c5 f4e4c4c5
mce 16 leaf-512,32-bit f4e4c4c5 f4e4c4c5
mce 16 leaf-512,64-bit f4e4c4c5 f4e4c4c5
mce 16 leaf-512,128-bit f4e4c4c5 f4e4c4c5
mce 16 leaf-512,256-bit f4e4c4c5 f4e4c4c5
mce 16 leaf-512,512-bit f4e4c4c5 f4e4c4c5

