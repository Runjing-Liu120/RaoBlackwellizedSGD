#!/bin/bash

python ../run_moving_mnist_training.py \
			--epochs 50 \
			--batch_size 64 \
			--seed 901 \
			--outdir '../mnist_vae_results/'\
			--outfilename 'moving_mnist_vae_nvil' \
			--grad_estimator 'nvil' \
			--propn_sample 0.1 \
			--learning_rate 1e-4 \
			--save_every 20 \
			--print_every 10 \
			--topk 0
