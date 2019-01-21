#!/bin/bash

for i in {1..10}
do
	((seed=$i + 119))
	python ../run_moving_mnist_training.py \
				--epochs 50 \
				--batch_size 64 \
				--seed $seed \
				--outdir '../mnist_vae_results/'\
				--outfilename moving_mnist_vae_nvil_trial${i} \
				--grad_estimator 'nvil' \
				--propn_sample 0.1 \
				--learning_rate 1e-3 \
				--save_every 6000 \
				--print_every 2 \
				--topk 0
done
