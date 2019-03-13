#!/bin/bash

for i in {1..10}
do
((seed=$i + 167))
python ../run_semisuper_vae_training.py \
			--epochs 100 \
			--seed ${seed} \
			--eval_test_set True \
			--save_every 1000 \
			--print_every 5 \
			--outdir '../mnist_vae_results/'\
			--outfilename ss_vae_relax_adapt_cv_trial${i}\
			--learning_rate 5e-4 \
			--rebar_eta 1.0 \
			--topk 0 \
			--grad_estimator 'relax' \
			--use_vae_init True \
			--vae_init_file '../mnist_vae_results/warm_starts/warm_start_vae_final' \
			--use_classifier_init True \
			--classifier_init_file '../mnist_vae_results/warm_starts/warm_start_classifier_final'
done
