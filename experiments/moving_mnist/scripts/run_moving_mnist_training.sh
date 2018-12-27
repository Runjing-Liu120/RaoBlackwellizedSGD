#!/bin/bash

python ../libraries/run_moving_mnist_training.py \
			--set_true_loc True \
			--epochs 50 \
			--seed 901 \
			--outdir '../mnist_vae_results/'\
			--outfilename 'moving_mnist_vae' \
			--propn_sample 1.0 \
			--learning_rate 1e-3 \
			--save_every 10
