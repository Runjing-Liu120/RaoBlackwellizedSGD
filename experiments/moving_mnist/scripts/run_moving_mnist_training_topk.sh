#!/bin/bash

for i in {1..10}
do
	((seed=$i + 8143))
	python ../run_moving_mnist_training.py \
				--epochs 50 \
				--batch_size 64 \
				--seed $seed \
				--outdir '../mnist_vae_results/'\
				--outfilename moving_mnist_vae_reinforce_double_bs_topk5_trial${i} \
				--grad_estimator 'reinforce_double_bs'\
				--propn_sample 0.1 \
				--learning_rate 1e-3 \
				--save_every 50000 \
				--print_every 2 \
				--topk 5
done
