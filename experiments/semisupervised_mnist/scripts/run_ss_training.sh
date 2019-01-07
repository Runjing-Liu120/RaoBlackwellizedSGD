#!/bin/bash

python ../run_semisuper_vae_training.py \
			--epochs 100 \
			--seed 901 \
			--save_every 20 \
			--outdir '../mnist_vae_results/'\
			--outfilename 'ss_vae_topk10' \
			--propn_sample 1.0 \
			--learning_rate 1e-3 \
			--topk 10 \
			--use_baseline False \
			--use_vae_init True \
			--vae_init_file '../mnist_vae_results/warm_startsvae_final' \
			--use_classifier_init True \
			--classifier_init_file '../mnist_vae_results/warm_startsclassifier_final'
