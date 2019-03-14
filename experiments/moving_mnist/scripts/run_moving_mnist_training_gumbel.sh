#!/bin/bash

for i in {1..10}
do
	((seed=$i + 234))
	python ../run_moving_mnist_training.py \
				--epochs 100 \
				--batch_size 64 \
				--seed $seed \
				--outdir '../mnist_vae_results/'\
				--outfilename moving_mnist_vae_gumbel_trial${i} \\
				--grad_estimator 'gumbel' \
				--propn_sample 0.1 \
				--learning_rate 1e-3 \
				--save_every 20 \
				--print_every 100 \
				--topk 0
done
