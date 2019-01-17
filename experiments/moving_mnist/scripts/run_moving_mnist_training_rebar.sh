#!/bin/bash

python ../run_moving_mnist_training.py \
			--epochs 50 \
			--batch_size 64 \
			--seed 901 \
			--outdir '../mnist_vae_results/'\
			--outfilename 'moving_mnist_vae_rebar' \
			--grad_estimator 'rebar' \
			--propn_sample 0.1 \
			--learning_rate 1e-3 \
			--save_every 20 \
			--print_every 2 \
			--topk 0
