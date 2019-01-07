#!/bin/bash

python ../run_semisuper_vae_training.py \
			--epochs 100 \
			--seed 901 \
			--outdir '../mnist_vae_results/'\
			--outfilename 'warm_starts' \
			--propn_sample 1.0 \
			--propn_labeled 0.1 \
			--learning_rate 1e-3 \
			--save_every 20 \
			--train_labeled_only True
