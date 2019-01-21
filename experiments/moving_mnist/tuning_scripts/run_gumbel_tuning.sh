#!/bin/bash

for lr in 5e-3 1e-3 5e-4 1e-4 5e-5
do
	for annealr in 1e-5 5e-5 1e-4 5e-4
	do
		python ../run_moving_mnist_training.py \
					--epochs 20 \
					--batch_size 64 \
					--seed 901 \
					--outdir '../mnist_vae_results/tuning_results/'\
					--outfilename moving_mnist_vae_gumbel_lr${lr}_annealr${annealr} \
					--grad_estimator 'gumbel' \
					--propn_sample 0.1 \
					--eval_test_set False \
					--learning_rate ${lr} \
					--gumbel_anneal_rate ${annealr} \
					--save_every 2000 \
					--print_every 20 \
					--topk 0
	done
done
