#!/bin/bash

python ../run_moving_mnist_training.py \
			--epochs 20 \
			--seed 901 \
			--set_true_loc True \
			--outdir '../mnist_vae_results/'\
			--outfilename 'moving_mnist_vae_set_true_loc' \
			--propn_sample 1.0 \
			--learning_rate 1e-3 \
			--save_every 5 
