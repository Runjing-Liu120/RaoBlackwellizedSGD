#!/bin/bash

python ../run_semisuper_vae_training.py \
			--epochs 100 \
			--seed 901 \
			--outdir '../mnist_vae_results/warm_starts/'\
			--outfilename 'warm_start' \
			--learning_rate 1e-3 \
			--save_every 200 \
			--train_labeled_only True
