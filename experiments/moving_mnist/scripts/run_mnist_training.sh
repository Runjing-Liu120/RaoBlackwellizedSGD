#!/bin/bash

python ../libraries/run_mnist_training.py \
			--epochs 50 \
			--seed 901 \
			--outdir '../mnist_vae_results/'\
			--outfilename 'mnist_vae' \
			--propn_sample 1.0 \
			--learning_rate 1e-3 \
			--save_every 10
