#!/bin/bash

for lr in 5e-3 1e-3 5e-4 1e-4 5e-5
do
python ../run_semisuper_vae_training.py \
			--epochs 10 \
			--seed 901 \
			--save_every 1000 \
			--print_every 10 \
			--outdir '../mnist_vae_results/tuning_results/'\
			--outfilename ss_vae_reinforce_lr$lr \
			--learning_rate $lr \
			--topk 0 \
			--grad_estimator 'reinforce' \
			--use_vae_init True \
			--vae_init_file '../mnist_vae_results/warm_starts/warm_start_vae_final' \
			--use_classifier_init True \
			--classifier_init_file '../mnist_vae_results/warm_starts/warm_start_classifier_final'
done
