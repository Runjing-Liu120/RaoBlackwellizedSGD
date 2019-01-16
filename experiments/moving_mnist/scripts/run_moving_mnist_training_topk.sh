#!/bin/bash

python ../run_moving_mnist_training.py \
			--epochs 80 \
			--batch_size 64 \
			--seed 901 \
			--outdir '../mnist_vae_results/'\
			--outfilename 'moving_mnist_vae_topk5' \
			--propn_sample 0.01 \
			--learning_rate 1e-3 \
			--save_every 20 \
			--print_every 10 \
			--topk 5 
