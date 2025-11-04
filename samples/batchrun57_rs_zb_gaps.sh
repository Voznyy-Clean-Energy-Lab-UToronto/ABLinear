#!/bin/bash

outfold=rs_zb_compare
infold=sspp/rszb_comp
offset=8

declare -a targets=( "Mg_O" "Na_F" "Rb_Br" "Li_I" )
declare -a trainset=( "zb_only" "zb_both" "both_both" )

for j in "${!targets[@]}"
do
	for k in "${!trainset[@]}"
	do
		for i in `seq 0 4`
		do
			#current best
			( jobname=$((${j} + ${offset}))_zb_${trainset[$k]}_Herelorb_1hgr_gap_ban_${targets[$j]}_w1600lr6e-3wd2e-4d1e-1_${i}_for; echo $jobname; python3 ABlinear_nn.py . --id-prop-t $infold/5_fold_${targets[$j]}_${trainset[$k]}_train_${i}.csv --id-prop-v $infold/5_fold_${targets[$j]}_${trainset[$k]}_val_${i}.csv --out $outfold/$jobname --width 1600 --funnel 2 -m 0 -e 20000 --ari "1hgroup_1hrow_ionic" --lr 6e-3 --wd 2e-4 -d 0.1 --normalize > ${jobname}.txt; python train_outputs/fastplot.py train_outputs/$outfold/$jobname ) &
		done
		wait $(jobs -p)
		
		for i in `seq 0 4`
		do
			#current best
			( jobname=$((${j} + ${offset}))_zb_${trainset[$k]}_Herelorb_1hgr_gap_ban_${targets[$j]}_w1600lr6e-3wd2e-4d1e-1_${i}_for; echo $jobname; python3 ABlinear_pred.py . --id-prop-p $infold/5_fold_${targets[$j]}_${trainset[$k]}_test.csv --out $outfold/$jobname --width 1600 --funnel 2 -m 0 --ari "1hgroup_1hrow_ionic" --lr 6e-3 --wd 2e-4 -d 0.1 > ${jobname}.txt; python train_outputs/fastplot_pred.py train_outputs/$outfold/$jobname ) &
		done
		wait $(jobs -p)
	done
done



#( jobname=ban_MgO_w1600lr6e-3wd2e-4d1e-1_${i}_for; echo $jobname; python3 ABlinear_nn.py . --id-prop-t sspp/rszb_comp/allbut1_MgO.csv --id-prop-v sspp/rszb_comp/allbut1_MgO_val.csv --out rs_zb_compare/$jobname --width 1600 --funnel 2 -m 0 -e 20000 --ari "1hgroup_1hrow_ionic" --lr 6e-3 --wd 2e-4 -d 0.1 --normalize > ${jobname}.txt; python train_outputs/fastplot.py train_outputs/rs_zb_compare/$jobname ) &