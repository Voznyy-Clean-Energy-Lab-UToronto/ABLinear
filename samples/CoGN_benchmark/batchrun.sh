#!/bin/bash

#zincblende test comparisons
for i in `seq 1 5`
do
	#zincblendes only
	( jobname=zbov_${i}; python customrun.py mb_trainset_zbo.csv mb_valset.csv mb_testset.csv batchout/${jobname}.csv > batchout/${jobname}.txt )
	#materials project gammas tested on all zincblendes
	( jobname=matproj_${i}; python customrun.py mb_mpgammatrain.csv mb_mpgammaval.csv mb_zbset.csv batchout/${jobname}.csv > batchout/${jobname}.txt )
	#materials project with zincblendes tested on zincblendes (same split as zincblende only)
	( jobname=matproj+zbv2_${i}; python customrun.py mb_mpgammatrain_zb.csv mb_mpgammaval_zb.csv mb_testset.csv batchout/${jobname}.csv > batchout/${jobname}.txt )
	
done

#UMAP models
#all gammas
( jobname=mpgammas; python customrun.py mb_mpgammaflt.csv mb_mpgammaflt.csv mb_mpgammaflt.csv mpgamma/${jobname}.csv ALLCIFS > mpgamma/${jobname}.txt )
#all formation energies
( jobname=All_mb_formation; python customrun.py mb_mb_formationenergy.csv mb_mb_formationenergy.csv mb_mb_formationenergy.csv mb_percents/${jobname}.csv efcifs > mb_percents/${jobname}.txt )	
#all bandgaps > 0.1 eV
( jobname=All_mb; python customrun.py mb_gaps_0.1.csv mb_gaps_0.1.csv mb_gaps_0.1.csv mb_percents/${jobname}.csv mbcifs > mb_percents/${jobname}.txt )	