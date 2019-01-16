#!/bin/bash

python ../run_moving_mnist_training.py \
			--epochs 50 \
			--seed 901 \
			--set_true_loc True \
			--outdir '../mnist_vae_results/'\
			--outfilename 'moving_mnist_vae_set_true_loc' \
			--propn_sample 0.1 \
			--learning_rate 1e-3 \
			--save_every 20 
