#!/bin/bash

python ../run_moving_mnist_training.py \
			--use_vae_init True \
			--vae_init_file ../mnist_vae_results/moving_mnist_vae_true_loc_final\
			--train_attn_only True \
			--epochs 50 \
			--seed 901 \
			--outdir '../mnist_vae_results/'\
			--outfilename 'moving_mnist_vae_warm_start_attn_only' \
			--propn_sample 1.0 \
			--learning_rate 1e-3 \
			--save_every 10
