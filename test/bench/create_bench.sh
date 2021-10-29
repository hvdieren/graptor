#!/bin/sh

cfg=$1
bench=$2
target=$3

echo "" > $target

for c in $cfg ; do
    cat <<INC >> $target
#include "config/${c}.h"
INC
done

cat <<EOF >> $target
#include "$bench"
EOF

