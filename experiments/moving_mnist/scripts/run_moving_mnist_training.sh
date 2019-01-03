#!/bin/bash

python ../run_moving_mnist_training.py \
			--use_vae_init False \
			--vae_init_file ../mnist_vae_results/moving_mnist_vae_true_loc_final\
			--train_attn_only False \
			--epochs 80 \
			--seed 901 \
			--outdir '../mnist_vae_results/'\
			--outfilename 'moving_mnist_vae_nsamples6' \
			--propn_sample 1.0 \
			--learning_rate 1e-3 \
			--save_every 20 \
			--topk 0 \
			--n_samples 6
