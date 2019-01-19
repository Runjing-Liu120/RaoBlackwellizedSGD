#!/bin/bash
for lr in 5e-3 1e-3 5e-4 1e-4 5e-5
do 
python ../run_semisuper_vae_training.py \
			--epochs 10 \
			--seed 901 \
			--save_every 1000 \
			--print_every 1000 \
			--outdir '../mnist_vae_results/tuning_results/'\
			--outfilename ss_vae_fully_marg_lr$lr \
			--propn_sample 1.0 \
			--propn_labeled 0.1 \
			--learning_rate $lr \
			--topk 10 \
			--grad_estimator 'reinforce' \
			--use_vae_init True \
			--vae_init_file '../mnist_vae_results/warm_starts/warm_start_vae_final' \
			--use_classifier_init True \
			--classifier_init_file '../mnist_vae_results/warm_starts/warm_start_classifier_final'
done
