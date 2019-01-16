#!/bin/bash

python ../run_moving_mnist_training.py \
			--epochs 50 \
			--batch_size 64 \
			--seed 992 \
			--outdir '../mnist_vae_results/'\
			--outfilename 'test' \
			--propn_sample 0.1 \
			--learning_rate 1e-3 \
			--save_every 20 \
			--print_every 10 \
			--topk 0 \
			--n_samples 1 \
			--use_vae_init False \
			--vae_init_file '../mnist_vae_results/moving_mnist_vae_topk5_final' 
