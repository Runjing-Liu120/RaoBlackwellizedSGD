#!/bin/bash

python ../run_semisuper_vae_training.py \
			--epochs 100 \
			--seed 901 \
			--outdir '../mnist_vae_results_propn_labeled_sandbox/'\
			--outfilename 'warm_starts' \
			--propn_sample 1.0 \
			--propn_labeled 0.05 \
			--learning_rate 1e-3 \
			--save_every 200 \
			--train_labeled_only True
