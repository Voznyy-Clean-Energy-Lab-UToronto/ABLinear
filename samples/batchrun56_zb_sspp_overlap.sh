#!/bin/bash

outfold=sspp_split_overlap
infold=sspp/overlap
offset=9

declare -a targets=( "ovro" )

for j in "${!targets[@]}"
do
	for i in `seq 0 4`
	do
		#current best
		( jobname=$((${j} + ${offset}))_zb_${targets[$j]}_Herelorb_1hgr_overlap_w1600lr6e-3wd2e-4d1e-1_${i}_for; echo $jobname; python3 ABlinear_nn.py . --id-prop-t $infold/5_fold_zb_${targets[$j]}_train_${i}.csv --id-prop-v $infold/5_fold_zb_${targets[$j]}_val_${i}.csv --out $outfold/$jobname --width 1600 --funnel 2 -m 0 -e 50000 --ari "1hgroup_1hrow_ionic" --lr 6e-3 --wd 2e-4 -d 0.1 --normalize > ${jobname}.txt; python train_outputs/fastplot.py train_outputs/$outfold/$jobname ) &
	done
	wait $(jobs -p)
done