#!/bin/bash

hours=24
node=smp05
prio=himem

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name bl:skl:$1
#PBS -r 1
#SBATCH --mem-per-cpu=23G
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --nodelist=${node}
#SBATCH -p $prio
#SBATCH --time=${hours}:00:00

. ./mce_blanusa_avx512_lib.sh

case $1 in
ork*) for_gg orkut undir 100 $2
	;;
tw*) for_gg twitter_xl undir 100 $2
	;;
dirtw*) for_gg twitter_xl dir 100 $2
	;;
fr*) for_gg friendster_full undir 5000 $2
	;;
dirfr*) for_gg friendster_full dir 5000 $2
	;;
CA*) for_gg CAroad undir 100 $2
	;;
US*) for_gg USAroad undir 100 $2
	;;
RM*) for_gg RMAT27 undir 1000 $2
	;;
Yah*) for_gg Yahoo_mem undir 100 $2
	;;
Liv*) for_gg LiveJournal undir 100 $2
	;;
dirLiv*) for_gg LiveJournal dir 100 $2
	;;
war*) for_gg warwiki undir 100 $2
	;;
dirwar*) for_gg warwiki dir 100 $2
	;;
pok*) for_gg pokec undir 100 $2
	;;
web*) for_gg webcc undir 100 $2
	;;
dirweb*) for_gg webcc dir 100 $2
	;;
uk*) for_gg uk_union_06 undir 100 $2
	;;
diruk*) for_gg uk_union_06 dir 100 $2
	;;
clu*) for_gg clueweb09 undir 100 $2
	;;
dirclu*) for_gg clueweb09 dir 100 $2
	;;
dim*) for_gg dimacs undir 100 $2
	;;
alta*) for_gg altavista undir 100 $2
	;;
diralta*) for_gg altavista dir 100 $2
	;;
direu15*) for_gg eu15 dir 100 $2
	;;
dirclueweb12*) for_gg clueweb12 dir 100 $2
	;;
dirgsh15*) for_gg gsh15 dir 100 $2
	;;
diruk14*) for_gg uk14 dir 100 $2
	;;
dirmclust*) for_gg mclust dir 100 $2
	;;
dirwdc12*) for_gg wdc12 dir 100 $2
	;;
dirwdc14*) for_gg wdc14 dir 100 $2
	;;
hs-cx*) for_gg bio-HS-CX undir 1 $2
	;;
mouse*) for_gg bio-mouse-gene undir 1 $2
	;;
human-gene1*) for_gg bio-human-gene1 undir 1 $2
	;;
human-gene2*) for_gg bio-human-gene2 undir 1 $2
	;;
worm*) for_gg bio-WormNet-v3 undir 1 $2
	;;
talk*) for_gg wiki-talk undir 1 $2
	;;
topcats*) for_gg wiki-topcats undir 1 $2
	;;
patents*) for_gg cit-patents undir 1 $2
	;;
higgs*) for_gg higgs-twitter undir 1 $2
	;;
sx*) for_gg sx-stackoverflow undir 1 $2
	;;
esac

true
EOT
