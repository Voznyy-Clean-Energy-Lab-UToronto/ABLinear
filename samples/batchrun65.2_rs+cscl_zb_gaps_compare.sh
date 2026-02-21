#!/bin/bash

outfold=cscl+rs_zb_compare_VASP_rng4
infold=sspp/crz_compare
offset=0

declare -a targets=( "zb" "zb_rs" "zb_rs_cscl" )
declare -a trainset=( "exclusive" "inclusive" )

for j in "${!targets[@]}"
do
	for k in "${!trainset[@]}"
	do
		for i in `seq 0 4`
		do
			#current best
			( jobname=$((${j} + ${offset}))_${targets[$j]}_w1600lr6e-3wd2e-4d1e-1_${trainset[$k]}_${i}_for; echo $jobname; python3 ABlinear_nn.py . --id-prop-t $infold/${targets[$j]}_${trainset[$k]}_train.csv --id-prop-v $infold/${targets[$j]}_${trainset[$k]}_val.csv --out $outfold/$jobname --width 860 --funnel 2 -m 0 -e 50000 --ari "+eneg+hard" --lr 0.004488486592394007 --wd 0.0001616679778585697 -d 0.1  --rngseed ${i} > ${jobname}.txt; python train_outputs/fastplot.py train_outputs/$outfold/$jobname ) &

		done
		wait $(jobs -p)
		
		for i in `seq 5 9`
		do
			#current best
			( jobname=$((${j} + ${offset}))_${targets[$j]}_w1600lr6e-3wd2e-4d1e-1_${trainset[$k]}_${i}_for; echo $jobname; python3 ABlinear_nn.py . --id-prop-t $infold/${targets[$j]}_${trainset[$k]}_train.csv --id-prop-v $infold/${targets[$j]}_${trainset[$k]}_val.csv --out $outfold/$jobname --width 860 --funnel 2 -m 0 -e 50000 --ari "+eneg+hard" --lr 0.004488486592394007 --wd 0.0001616679778585697 -d 0.1  --rngseed ${i} > ${jobname}.txt; python train_outputs/fastplot.py train_outputs/$outfold/$jobname ) &

		done
		wait $(jobs -p)
	
		for i in `seq 0 9`
		do
		#current best
		( jobname=$((${j} + ${offset}))_${targets[$j]}_w1600lr6e-3wd2e-4d1e-1_${trainset[$k]}_${i}_for; echo $jobname; python3 ABlinear_pred.py . --id-prop-p $infold/${targets[$j]}_${trainset[$k]}_test.csv --out $outfold/$jobname --width 860 --funnel 2 -m 0 -e 20000 --ari "+eneg+hard"  >> ${jobname}.txt; python train_outputs/fastplot.py train_outputs/$outfold/$jobname ) &
		done
		wait $(jobs -p)
	done
done
