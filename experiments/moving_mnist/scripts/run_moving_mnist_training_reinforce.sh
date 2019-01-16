#!/bin/bash

python ../run_moving_mnist_training.py \
			--epochs 100 \
			--batch_size 64 \
			--seed 901 \
			--outdir '../mnist_vae_results/'\
			--outfilename 'moving_mnist_vae_reinforce_double_bs' \
			--grad_estimator 'reinforce_double_bs'
			--propn_sample 0.1 \
			--learning_rate 1e-3 \
			--save_every 20 \
			--print_every 5 \
			--topk 0 \
			--n_samples 1 \
