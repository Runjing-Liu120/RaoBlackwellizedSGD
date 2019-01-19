#!/bin/bash

for seed in {1..10}
do
python ../run_semisuper_vae_training.py \
			--epochs 100 \
			--seed 22314 + ${seed} \
			--save_every 1000 \
			--print_every 5 \
			--outdir '../mnist_vae_results/'\
			--outfilename ss_vae_reinforce_topk1_trial${seed} \
			--propn_sample 1.0 \
			--learning_rate 1e-3 \
			--topk 1 \
			--grad_estimator 'reinforce' \
			--use_vae_init True \
			--vae_init_file '../mnist_vae_results/warm_starts/warm_start_vae_final' \
			--use_classifier_init True \
			--classifier_init_file '../mnist_vae_results/warm_starts/warm_start_classifier_final'
done
