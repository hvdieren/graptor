#!/bin/bash

#what="Enumeration"
#what="Removed self-edges"
what="Calculating coreness"
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
    for file in `ls -1 graptor_$arch/$dir/output.${commit}.t${threads}.c${part}.r*.${bench}_${version}_${opts}.${graph} 2> /dev/null` ; do
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

function mce() {
    local vl=vl$1
    local threads=$2
    local commit=$3
    local part=$4
    local arch=$5
    local dir=$6
    local graphs="USAroad_undir CAroad_undir wiki-talk_undir cit-patents_undir sx-stackoverflow_undir Yahoo_mem_undir warwiki_undir pokec_undir wiki-topcats_undir friendster_full_undir bio-HS-CX_undir higgs-twitter_undir orkut_undir bio-WormNet-v3_undir"

    if [ x$dir = x ] ; then
	dir=$commit
    fi

    local variants="$vl"
    #local variants="$vl abDc_abBc_abGc_$vl abTy_abTd_abL_$vl abTy_$vl abTd_abL_$vl abDxph_abBxph_abAxph_$vl abPxph_$vl abxpv_$vl abDi_abBi_$vl"
    ##local variants="$vl abDc_abBc_abGc_$vl abTy_abTd_abL_$vl abTy_$vl abTd_abL_$vl abDxph_$vl abBxph_$vl abAxph_$vl abDxph_abBxph_abAxph_$vl abPxph_$vl abxpv_$vl abDi_abBi_$vl abGc_$vl"
    #local variants="$vl ${vl}_yes512_noavx512f ${vl}_no512 ${vl}_no512_noavx512f"
    #local variants="$vl ${vl}_yes512_noavx512f ${vl}_no512 ${vl}_no512_noavx512f abGc_${vl} ${vl}_nopopc ${vl}_yes512_noavx512f_nopopc abGc_$vl"
    #local variants="$vl abDc_abBc_abGc_$vl abTy_abTd_abL_$vl abDxph_abBxph_abAxph_$vl abPxph_$vl abxpv_$vl abDi_abBi_$vl Dq32_Bq32_$vl Dq64_Bq64_$vl Dq128_Bq128_$vl"
    #local variants="$vl par2_$vl par4_$vl"
    #local variants="$vl par0_${vl}_papi par1_${vl}_papi"
    #local variants="DQ64_BQ64_vl8 DQ64_BQ64_Dq32_Bq32_vl8 DQ32_BQ32_Dq32_Bq32_vl8 vl8 abBc_abDc_abGc_vl8 abTy_abTd_abL_vl8 abDxph_abBxph_abAxph_vl8 abPxph_vl8 abxpv_vl8 Dq3_Bq3_vl8 Dq10_Bq10_vl8 Dq20_Bq20_vl8 Dq50_Bq50_vl8 Dd20_Bd20_vl8 Dd80_Bd80_vl8 Dd99_Bd99_vl8"
    echo "VL=$vl threads=$threads commit=$commit part=$part"
    (
	echo -ne "\"\" "
	echo " $variants" | tr ' ' "\n"
	for graph in $graphs ; do
	    echo $graph
	    for variant in $variants ; do
		one $commit $threads MCE "" ${variant} ${graph} ${part} ${dir} ${arch}
	    done
	done
    ) | paste - $(echo $variants | sed -e 's/\b[a-zA-Z0-9_]*\b/-/g')

    echo
}

#mce 8 128 7ad08053 128
#mce 8 128 dfa6086d 128
#mce 8 128 8689e7c8 128
#mce 8 128 6f9329ed 512
#mce 8 128 3ae3ceec 512
#mce 8 128 773ac7f2 512
#mce 8 128 c90aa41a 512
#mce 8 128 f67848ff 512 epyc f67848ff_try1 
#mce 8 128 f67848ff 512
#mce 8 128 c5e03980 512 epyc
#mce 8 128 c5e03980 1024 epyc
#mce 8 128 c5e03980 2048 epyc
#mce 8 128 a241fc39 1024 epyc a241fc39_himem
#mce 8 128 a241fc39 1024 epyc
#mce 8 128 0f66b914 1024 epyc 0f66b914 
#mce 8 128 0f66b914 1024 epyc 0f66b914_numa8 
#mce 8 128 eb33192e 1024 epyc eb33192e
#mce 8 128 12343215 1024 epyc 12343215
#mce 8 128 12343215 1024 epyc 12343215_numa8
#mce 16 32 0f66b914 512 avx512
#mce 16 32 3ae3ceec 512 avx512
#mce 8 1 f0192005 128

#mce 16 128 0f20dd74 1024 avx512 0f20dd74
#mce 8 128 0f20dd74 1024 avx512 0f20dd74
#mce 8 128 f4e4c4c5 1024 avx512 f4e4c4c5
mce 16 128 f4e4c4c5 1024 avx512 f4e4c4c5
#mce 16 1 f4e4c4c5 1024 avx512 f4e4c4c5
#mce 8 1 f4e4c4c5 1024 avx512 f4e4c4c5
