#!/bin/bash

python ../run_gumbel_vae_training.py \
			--epochs 10 \
			--seed 901 \
			--save_every 20 \
			--print_every 5 \
			--outdir '../mnist_vae_results_propn_labeled_sandbox/'\
			--outfilename 'gumbel_vae' \
			--propn_sample 1.0 \
			--propn_labeled 0.05 \
			--learning_rate 1e-3 \
			--use_vae_init True \
			--vae_init_file '../mnist_vae_results_propn_labeled_sandbox/warm_starts_vae_final' \
			--use_classifier_init True \
			--classifier_init_file '../mnist_vae_results_propn_labeled_sandbox/warm_starts_classifier_final'