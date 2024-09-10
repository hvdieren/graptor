#!/bin/bash

#GRAPHPATH=/mnt/scratch2/users/3047056/graphs/tree
#LOCALPATH=/tmp/users/3047056
#SRCPATH=hpdc:/var/shared/projects/asap/graphs/adj/realworld/recent
#BINDIR=$HOME/graptor/exec-vl/bin

GRAPHPATH=
LOCALPATH=
SRCPATH=
BINDIR=./tools

# cleanup handler
#declare -ga cleanup_files
#cleanup_files=()

function cleanup()
{
#    declare -ga cleanup_files
#    echo "checking cleanup (${#cleanup_files})..."
#    if [ ${#cleanup_files} -eq 0 ] ; then
#	for f in $cleanup_files ; do
#	    echo "Removing $f"
#	    echo rm -f $f
#	done
#	cleanup_files=()
#    fi
     #rm $LOCALPATH/*
    echo cleaning up...
}

#trap cleanup EXIT

function set_cleanup_file()
{
    #declare -ga cleanup_files
    #echo "Recording for cleanup: '$1' / was '${#cleanup_files}'" >&2
    #cleanup_files+=('$1')
    echo "Recording cleanup (mute)" >&2
}

# arguments: path tmp
function unzip_file()
{
    local path=$1
    local tmp=$2

    local gfile=${path}
    if [ ! -e ${path} -a -e ${path}.gz ] ; then
	gfile=$tmp

	if [ ! -e $tmp ] ; then
	    echo "unzip ${path}.gz into $tmp" >&2
	    if gunzip -c ${path}.gz > $tmp ;
	    then
		set_cleanup_file $tmp
	    else
		echo "unzip ${path}.gz into $tmp failed" >&2
		if [ ! -e ${path} ] ; then
		    echo "unzip ${path}.gz into ${path}" >&2
		    gunzip -c ${path}.gz > ${path}
		    set_cleanup_file ${path}
		fi
		gfile=${path}
	    fi
	fi
    fi

    echo $gfile
}

# arguments: path filename srcpath
function download_file_if_missing()
{
    local path=$1
    local filename=$2
    local srcpath=$3

    if [ ! -e ${path}/${filename} -a ! -e ${path}/${filename}.gz ] ; then
	echo "$path/$filename missing" >&2
	if [ -z "${srcpath}" ] ; then
	    echo "Cannot download file: remote not specified" >&2
	    exit 1
	fi

	local dirname=`dirname $filename`
	if [ ! -e $path/$dirname ] ; then
	    echo Create directory: $path/$dirname >&2
	    mkdir -p $path/$dirname
	    lfs setstripe -c 32 -S 128m $path/$dirname
	fi

	echo Download data: scp $srcpath/${filename}* ${path}/${dirname}/ >&2
    	scp $srcpath/${filename} $srcpath/${filename}.gz ${path}/${dirname}/
    fi
}

# arguments: path graph (un)dir format weights-spec srcpath
function create_ligra_weights_from_graptor()
{
    local path=$1
    local graph=$2
    local un_dir=$3
    local format=$4
    local wspec=$5
    local spec=adjacency_${wspec}
    local srcpath=$6

    local ligra_file=`graph_file $path $graph $un_dir ligra $spec`
    if [ ! -e ${path}/${ligra_file} -a ! -e ${path}/${ligra_file}.gz ] ; then
	local graptor_weights_path=`graph_path $path $graph $un_dir b2 $wspec`
	local graptor_graph_path=`graph_path $path $graph $un_dir b2 b2`

	local dirname=`dirname $ligra_file`
	if [ ! -e $path/$dirname ] ; then
	    echo Create directory: $path/$dirname >&2
	    mkdir -p $path/$dirname
	    lfs setstripe -c 32 -S 128m $path/$dirname
	fi

	echo "Converting weighted graph to ligra format" >&2
	local sym=""
	if [[ "$un_dir" == undir* ]]; then
	    sym="-s"
	fi
	$BINDIR/CvtBinToText $sym -i $graptor_graph_path \
			     -weights $graptor_weights_path -o $path/$ligra_file >&2
	echo "Compressing $path/$ligra_file" >&2
	#gzip -9 $path/$ligra_file
    fi
}

# arguments: path graph (un)dir format srcpath
function create_ligra_from_graptor()
{
    local path=$1
    local graph=$2
    local un_dir=$3
    local format=$4
    local spec=adjacency
    local srcpath=$5

    local ligra_file=`graph_file $path $graph $un_dir ligra $spec`
    if [ ! -e ${path}/${ligra_file} -a ! -e ${path}/${ligra_file}.gz ] ; then
	local graptor_graph_path=`graph_path $path $graph $un_dir b2 b2`

	local dirname=`dirname $ligra_file`
	if [ ! -e $path/$dirname ] ; then
	    echo Create directory: $path/$dirname >&2
	    mkdir -p $path/$dirname
	    lfs setstripe -c 32 -S 128m $path/$dirname
	fi

	echo "Converting graph to ligra format" >&2
	local sym=""
	if [[ "$un_dir" == undir* ]]; then
	    sym="-s"
	fi
	$BINDIR/CvtBinToText $sym -i $graptor_graph_path -o $path/$ligra_file >&2
	echo "Compressing $path/$ligra_file" >&2
	#gzip -9 $path/$ligra_file
    fi
}

# arguments: path graph (un)dir format weights-spec srcpath
function create_gapbs_weights_from_graptor()
{
    local path=$1
    local graph=$2
    local un_dir=$3
    local format=$4
    local wspec=$5
    local spec=edgelist_${wspec}.wel
    local srcpath=$6

    local gapbs_file=`graph_file $path $graph $un_dir el $spec`
    if [ ! -e ${path}/${gapbs_file} -a ! -e ${path}/${gapbs_file}.gz ] ; then
	local graptor_weights_path=`graph_path $path $graph $un_dir b2 $wspec`
	local graptor_graph_path=`graph_path $path $graph $un_dir b2 b2`

	local dirname=`dirname $gapbs_file`
	if [ ! -e $path/$dirname ] ; then
	    echo Create directory: $path/$dirname >&2
	    mkdir -p $path/$dirname
	    lfs setstripe -c 32 -S 128m $path/$dirname
	fi

	echo "Converting weighted graph to edge-list format" >&2
	local sym=""
	if [[ "$un_dir" == undir* ]]; then
	    sym="-s"
	fi
	$BINDIR/CvtBinToEdgeList $sym -i $graptor_graph_path \
				 -weights $graptor_weights_path \
				 -o $path/$gapbs_file >&2
	echo "Compressing $path/$gapbs_file" >&2
	#gzip -9 $path/$gapbs_file
    fi
}


# arguments: path graph (un)dir format weights-spec srcpath
function create_gapbs_from_graptor()
{
    local path=$1
    local graph=$2
    local un_dir=$3
    local format=$4
    local spec=edgelist.el
    local srcpath=$5

    local gapbs_file=`graph_file $path $graph $un_dir el $spec`
    if [ ! -e ${path}/${gapbs_file} -a ! -e ${path}/${gapbs_file}.gz ] ; then
	local graptor_graph_path=`graph_path $path $graph $un_dir b2 b2`

	local dirname=`dirname $gapbs_file`
	if [ ! -e $path/$dirname ] ; then
	    echo Create directory: $path/$dirname >&2
	    mkdir -p $path/$dirname
	    lfs setstripe -c 32 -S 128m $path/$dirname
	fi

	echo "Converting graph to edge-list format" >&2
	local sym=""
	if [[ "$un_dir" == undir* ]]; then
	    sym="-s"
	fi
	$BINDIR/CvtBinToEdgeList $sym -i $graptor_graph_path \
				 -o $path/$gapbs_file >&2
	echo "Compressing $path/$gapbs_file" >&2
	#gzip -9 $path/$gapbs_file
    fi
}

# arguments: path graph (un)dir format weights-spec srcpath
function create_mtxmkt_from_graptor()
{
    local path=$1
    local graph=$2
    local un_dir=$3
    local format=$4
    local spec=pattern.mtx
    local srcpath=$5

    local mtx_file=`graph_file $path $graph $un_dir mtx $spec`
    if [ ! -e ${path}/${mtx_file} -a ! -e ${path}/${mtx_file}.gz ] ; then
	local graptor_graph_path=`graph_path $path $graph $un_dir b2 b2`

	local dirname=`dirname $mtx_file`
	if [ ! -e $path/$dirname ] ; then
	    echo Create directory: $path/$dirname >&2
	    mkdir -p $path/$dirname
	    lfs setstripe -c 32 -S 128m $path/$dirname
	fi

	echo "Converting graph to edge-list format" >&2
	local sym=""
	if [[ "$un_dir" == undir* ]]; then
	    sym="-s"
	fi
	$BINDIR/CvtBinToEdgeList $sym -i $graptor_graph_path \
				 -m -o $path/$mtx_file >&2
	echo "Compressing $path/$mtx_file" >&2
	#gzip -9 $path/$mtx_file
    fi
}


# arguments: path graph (un)dir format spec srcpath
function check_graph()
{
    local path=$1
    local graph=$2
    local un_dir=$3
    local format=$4
    local spec=$5
    local srcpath=$6
    
    local filename=`graph_file $path $graph $un_dir $format $spec`
    download_file_if_missing $path $filename $srcpath
}

# arguments: path graph (un)dir format spec
function graph_file()
{
    local path=$1
    local graph=$2
    local un_dir=$3
    local format=$4
    local spec=$5

    echo $graph/$un_dir/$format/$spec
}    

# arguments: path graph (un)dir format spec
function graph_path()
{
    local path=$1
    local graph=$2
    local un_dir=$3
    local format=$4
    local spec=$5

    echo $path/$graph/$un_dir/$format/$spec
}    

# arguments: graph un_dir weights
function ligra_graph_path()
{
    local graph=$1
    local un_dir=$2
    local weights=$3

    if [ x$weights == xnone ] ; then
	graph_path $GRAPHPATH $graph $un_dir ligra adjacency
    else
	graph_path $GRAPHPATH $graph $un_dir ligra adjacency_$weights
    fi
}

# arguments: graph un_dir weights
function ligra_graph()
{
    local graph=$1
    local un_dir=$2
    local weights=$3

    if [ x$weights == xnone ] ; then
	create_ligra_from_graptor $GRAPHPATH $graph $un_dir ligra $SRCPATH
    else
	create_ligra_weights_from_graptor $GRAPHPATH $graph $un_dir \
					  ligra $weights $SRCPATH
    fi
}

# arguments: graph un_dir weights
function gapbs_graph_path()
{
    local graph=$1
    local un_dir=$2
    local weights=$3

    if [ x$weights == xnone ] ; then
	graph_path $GRAPHPATH $graph $un_dir el edgelist.el
    else
	graph_path $GRAPHPATH $graph $un_dir el edgelist_${weights}.wel
    fi
}

# arguments: graph un_dir weights
function gapbs_graph()
{
    local graph=$1
    local un_dir=$2
    local weights=$3

    if [ x$weights == xnone ] ; then
	create_gapbs_from_graptor $GRAPHPATH $graph $un_dir el $SRCPATH
    else
	create_gapbs_weights_from_graptor $GRAPHPATH $graph $un_dir \
					  el $weights $SRCPATH
    fi
}

# arguments: graph un_dir weights
function mtxmkt_graph_path()
{
    local graph=$1
    local un_dir=$2
    local weights=$3

    if [ x$weights == xnone ] ; then
	graph_path $GRAPHPATH $graph $un_dir mtx pattern.mtx
    else
	graph_path $GRAPHPATH $graph $un_dir mtx pattern_${weights}.mtx
    fi
}

# arguments: graph un_dir weights
function mtxmkt_graph()
{
    local graph=$1
    local un_dir=$2
    local weights=$3

    if [ x$weights == xnone ] ; then
	create_mtxmkt_from_graptor $GRAPHPATH $graph $un_dir mtx $SRCPATH
    else
	echo "Not yet implemented - weighted matrix market file" >&2
	exit 1
    fi
}


# arguments: graph un_dir 
function graptor_graph_path()
{
    local graph=$1
    local un_dir=$2

    graph_path $GRAPHPATH $graph $un_dir b2 b2
}

# arguments: graph un_dir weights
function graptor_weights_file()
{
    local graph=$1
    local un_dir=$2
    local weights=$3

    graph_path $GRAPHPATH $graph $un_dir b2 $weights
}

# arguments: graph-name weights
function graptor_weights_file_from_name()
{
    local graph_name=$1
    local weights=$2
    # Remove shortest substring ending in _*dir
    local graph=${graph_name%_*dir}
    # Remove longest substring before the final _
    local un_dir=${graph_name##*_}

    graptor_weights_file $graph $un_dir $weights
}

# arguments: graph un_dir [weights]
function graptor_graph()
{
    local graph=$1
    local un_dir=$2
    local weights=$3

    # The graph topology
    check_graph $GRAPHPATH $graph $un_dir b2 b2 $SRCPATH

    # The edge weights
    if [ x"$weights" != x -a x"$weights" != xnone ] ; then
	check_graph $GRAPHPATH $graph $un_dir b2 $weights $SRCPATH
    fi
}

# arguments: graph un_dir 
function graptorv4_graph_path()
{
    local graph=$1
    local un_dir=$2

    graph_path $GRAPHPATH $graph $un_dir v4 v4
}

# arguments: graph un_dir weights
function graptorv4_weights_file()
{
    local graph=$1
    local un_dir=$2
    local weights=$3

    graph_path $GRAPHPATH $graph $un_dir v4 $weights
}

# arguments: graph-name weights
function graptorv4_weights_file_from_name()
{
    local graph_name=$1
    local weights=$2
    # Remove shortest substring ending in _*dir
    local graph=${graph_name%_*dir}
    # Remove longest substring before the final _
    local un_dir=${graph_name##*_}

    graptorv4_weights_file $graph $un_dir $weights
}

# arguments: graph un_dir [weights]
function graptorv4_graph()
{
    local graph=$1
    local un_dir=$2
    local weights=$3

    # The graph topology
    check_graph $GRAPHPATH $graph $un_dir v4 v4 $SRCPATH

    # The edge weights
    if [ x"$weights" != x -a x"$weights" != xnone ] ; then
	check_graph $GRAPHPATH $graph $un_dir v4 $weights $SRCPATH
    fi
}


# aux
function list_include_item()
{
  local list="$1"
  local item="$2"
  if [[ $list =~ (^|[[:space:]])"$item"($|[[:space:]]) ]] ; then
    # yes, list include item
    result=0
  else
    result=1
  fi
  return $result
}

# argument: program name
function help_message()
{
    echo "Usage: $(basename $1) [-l local-root] [-r remote-root] action format graph direction [weights]"
    echo "Supported commands"
    echo "    materialise    create file if required and return path"
    echo "    path           determine path of file"
    echo "    cleanup        cleanup temporary file"
}

# main code
getopt --test 2> /dev/null
if [[ $? -ne 4 ]]; then
    echo "GNU enhanced getopt is required. Available version too old."
    exit 1
fi

# All supported options
SHORT="-l:r:h"
LONG="local-tree,remote-tree,help"

OPTS=$(getopt -o ${SHORT} --longoptions ${LONG} --name "$0" -- "$@")
if [[ $? -ne 0 ]] ; then
    # getopt will have printed error messages already
    exit 1
fi

eval set -- "${OPTS}"

declare -a command

while [[ $# -gt 0 ]] ; do
    case "$1" in
	-l|--local-tree)
	    shift
	    GRAPHPATH="$1"
	    ;;
	-r|--remote-tree)
	    shift
	    SRCPATH="$1"
	    ;;
	-h|--help)
	    help_message $0
	    exit 1
	    ;;
	--)
	    shift
	    ;;
	*) command+=("$1")
	   ;;
    esac
    shift
done

if [ ${#command[@]} -ne 4 -a ${#command[@]} -ne 5 ] ; then
    help_message $0
    exit 1
fi

if ! `list_include_item "graptor graptorv4 ligra gapbs mtxmkt" "${command[1]}"`
then
   echo "Format ${command[1]} is not a recognised format" >2&
   exit 1;
fi

if [ x${command[0]} = xmaterialise ] ; then
    eval "${command[1]}_graph ${command[2]} ${command[3]} ${command[4]}"
    eval "${command[1]}_graph_path ${command[2]} ${command[3]} ${command[4]}"
elif [ x${command[0]} = xpath ] ; then
    eval "${command[1]}_graph_path ${command[2]} ${command[3]} ${command[4]}"
elif [ x${command[0]} = xcleanup ] ; then
    eval "echo cleanup ..."
else
    echo "Unrecognised command '${command[0]}'"
    exit 1
fi

exit 0
