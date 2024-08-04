#!/bin/bash

#what="filter"
#what="Enumeration"
#what="Calculting coreness"
#what="Completed MC in"
what="Completed search in"
#what="filter0"
#what="filter1"
#what="filter2"
whatn=$((`echo "$what" | sed -e 's/[^ ]//g' | wc -c` + 1))

function parse_file() {
    local file=$1
    local preamble=$2

    local field=$((`echo "$preamble" | sed -e 's/[^ ]//g' | wc -c` + 1))

    local fail=0

    if [ -e $file ] ; then
	if grep FAIL $file > /dev/null 2>&1 ; then
	    echo FAIL
	else
	    ( grep "$preamble" $file 2> /dev/null || echo $preamble NOMETRIC ) | cut -d' ' -f$field
	fi
    else
	echo ABSENT
    fi
}

function get_avg() {
    local commit=$1
    local threads=$2
    local bench=$3
    local graph=$4
    local dir=$5
    local arch=$6
    local preamble="$7"

    local sum=0
    local count=0
    local fail=0
    
    for file in `ls -1 base_$arch/$dir/output.${commit}.t${threads}.r*.${bench}.${graph}_undir 2> /dev/null` ; do
	local result=`parse_file $file "$preamble"`
	case "$result" in
	    FAIL) fail=1 ;;
	    NOMETRIC) fail=3 ;;
	    ABSENT) fail=2 ;;
	    *) 
		sum=`echo | perl -ne "END { printf \"%g\n\", ($sum+$result); }"`
		count=$(( $count + 1 ))
	esac
    done

    if [ $fail -eq 0 -a $count -ne 0 ] ; then
    	echo | perl -ne "END { printf \"%g\n\", ($sum/$count); }"
    elif [ $fail -eq 1 ] ; then
	echo FAIL
    elif [ $fail -eq 3 ] ; then
	echo NOMETRIC
    else
	echo ABSENT
    fi
}

function one() {
    local commit=$1
    local threads=$2
    local bench=$3
    local graph=$4
    local dir=$5
    local arch=$6

    local preamble="nowt"

    case "$bench" in
	pmc) preamble="^Time taken:" ;;
	cliquer) preamble="^Maximum clique search:" ;;
	dOmega_BS) ;& # fall-through
	dOmega_LS) preamble="^Total running time:" ;;
    esac

    get_avg "$commit" "$threads" "$bench" "$graph" "$dir" "$arch" "$preamble"
}

function base() {
    local threads=$1
    local commit=$2
    local arch=$3
    local graphs="CAroad LiveJournal xxx USAroad Yahoo_mem bio-HS-CX bio-WormNet-v3 cit-patents dblp2012 xxx friendster_full higgs-twitter hollywood2009 hudong it-2004 xxx orkut pokec sinaweibo sx-stackoverflow warwiki webcc wiki-talk wiki-topcats"

    if [ x$dir = x ] ; then
	dir=$commit
    fi

    local algo="pmc cliquer dOmega_LS dOmega_BS"

    echo "base threads=$threads commit=$commit"
    (
	echo -ne "\"\" "
	echo " $algo" | tr ' ' "\n"
	for graph in $graphs ; do
	    echo $graph
	    for a in $algo ; do
		one $commit $threads $a ${graph} ${dir} ${arch}
	    done
	done
    ) | paste - $(echo $algo | sed -e 's/\b[a-zA-Z0-9_]*\b/-/g')

    echo
}

commit=v0

base 128 $commit avx512
