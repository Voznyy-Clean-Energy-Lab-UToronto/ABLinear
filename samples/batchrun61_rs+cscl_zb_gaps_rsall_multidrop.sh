#!/bin/bash

outfold=cscl+rs_zb_compare_3
infold=sspp/rszbcc_comp_2
offset=0

declare -a targets=( "mdrp" )
declare -a trainset=( "csclrsallbackr" "csclrsallbacku" )
declare -a valset=( "csclrsallback" "csclrsallback" )

for j in "${!targets[@]}"
do
	for k in "${!trainset[@]}"
	do
		for i in `seq 0 4`
		do
			#current best
			( jobname=$((${j} + ${offset}))_zb_${trainset[$k]}_Herelorb_1hgr_gap_ban_${targets[$j]}_w1600lr6e-3wd2e-4d1e-1_${i}_for; echo $jobname; python3 ABlinear_nn.py . --id-prop-t $infold/5_fold_${targets[$j]}_${trainset[$k]}_train_${i}.csv --id-prop-v $infold/5_fold_${targets[$j]}_${valset[$k]}_val_${i}.csv --out $outfold/$jobname --width 1600 --funnel 2 -m 0 -e 20000 --ari "1hgroup_1hrow_ionic" --lr 6e-3 --wd 2e-4 -d 0.1 --normalize > ${jobname}.txt; python train_outputs/fastplot.py train_outputs/$outfold/$jobname ) &
		done
		wait $(jobs -p)
		
		for i in `seq 0 4`
		do
			#current best
			( jobname=$((${j} + ${offset}))_zb_${trainset[$k]}_Herelorb_1hgr_gap_ban_${targets[$j]}_w1600lr6e-3wd2e-4d1e-1_${i}_for; echo $jobname; python3 ABlinear_pred.py . --id-prop-p $infold/5_fold_${targets[$j]}_${valset[$k]}_test.csv --out $outfold/$jobname --width 1600 --funnel 2 -m 0 --ari "1hgroup_1hrow_ionic" --lr 6e-3 --wd 2e-4 -d 0.1 > ${jobname}.txt; python train_outputs/fastplot_pred.py train_outputs/$outfold/$jobname ) &
		done
		wait $(jobs -p)
	done
done

declare -a trainset=( "rsallbackr" "rsallbacku" )
declare -a valset=( "rsallback" "rsallback" )

for j in "${!targets[@]}"
do
	for k in "${!trainset[@]}"
	do
		for i in `seq 0 4`
		do
			#current best
			( jobname=$((${j} + ${offset}))_zb_${trainset[$k]}_Herelorb_1hgr_gap_ban_${targets[$j]}_w1600lr6e-3wd2e-4d1e-1_${i}_for; echo $jobname; python3 ABlinear_nn.py . --id-prop-t $infold/5_fold_${targets[$j]}_${trainset[$k]}_train_${i}.csv --id-prop-v $infold/5_fold_${targets[$j]}_${valset[$k]}_val_${i}.csv --out $outfold/$jobname --width 1600 --funnel 2 -m 0 -e 20000 --ari "1hgroup_1hrow_ionic" --lr 6e-3 --wd 2e-4 -d 0.1 --normalize > ${jobname}.txt; python train_outputs/fastplot.py train_outputs/$outfold/$jobname ) &
		done
		wait $(jobs -p)
		
		for i in `seq 0 4`
		do
			#current best
			( jobname=$((${j} + ${offset}))_zb_${trainset[$k]}_Herelorb_1hgr_gap_ban_${targets[$j]}_w1600lr6e-3wd2e-4d1e-1_${i}_for; echo $jobname; python3 ABlinear_pred.py . --id-prop-p $infold/5_fold_${targets[$j]}_${valset[$k]}_test.csv --out $outfold/$jobname --width 1600 --funnel 2 -m 0 --ari "1hgroup_1hrow_ionic" --lr 6e-3 --wd 2e-4 -d 0.1 > ${jobname}.txt; python train_outputs/fastplot_pred.py train_outputs/$outfold/$jobname ) &
		done
		wait $(jobs -p)
	done
done

declare -a trainset=( "zbonlyr" "zbonlyu" )
declare -a valset=( "zbonly" "zbonly" )

for j in "${!targets[@]}"
do
	for k in "${!trainset[@]}"
	do
		for i in `seq 0 4`
		do
			#current best
			( jobname=$((${j} + ${offset}))_zb_${trainset[$k]}_Herelorb_1hgr_gap_ban_${targets[$j]}_w1600lr6e-3wd2e-4d1e-1_${i}_for; echo $jobname; python3 ABlinear_nn.py . --id-prop-t $infold/5_fold_${targets[$j]}_${trainset[$k]}_train_${i}.csv --id-prop-v $infold/5_fold_${targets[$j]}_${valset[$k]}_val_${i}.csv --out $outfold/$jobname --width 1600 --funnel 2 -m 0 -e 20000 --ari "1hgroup_1hrow_ionic" --lr 6e-3 --wd 2e-4 -d 0.1 --normalize > ${jobname}.txt; python train_outputs/fastplot.py train_outputs/$outfold/$jobname ) &
		done
		wait $(jobs -p)
		
		for i in `seq 0 4`
		do
			#current best
			( jobname=$((${j} + ${offset}))_zb_${trainset[$k]}_Herelorb_1hgr_gap_ban_${targets[$j]}_w1600lr6e-3wd2e-4d1e-1_${i}_for; echo $jobname; python3 ABlinear_pred.py . --id-prop-p $infold/5_fold_${targets[$j]}_${valset[$k]}_test.csv --out $outfold/$jobname --width 1600 --funnel 2 -m 0 --ari "1hgroup_1hrow_ionic" --lr 6e-3 --wd 2e-4 -d 0.1 > ${jobname}.txt; python train_outputs/fastplot_pred.py train_outputs/$outfold/$jobname ) &
		done
		wait $(jobs -p)
	done
done

#( jobname=ban_MgO_w1600lr6e-3wd2e-4d1e-1_${i}_for; echo $jobname; python3 ABlinear_nn.py . --id-prop-t sspp/rszb_comp/allbut1_MgO.csv --id-prop-v sspp/rszb_comp/allbut1_MgO_val.csv --out rs_zb_compare/$jobname --width 1600 --funnel 2 -m 0 -e 20000 --ari "1hgroup_1hrow_ionic" --lr 6e-3 --wd 2e-4 -d 0.1 --normalize > ${jobname}.txt; python train_outputs/fastplot.py train_outputs/rs_zb_compare/$jobname ) &