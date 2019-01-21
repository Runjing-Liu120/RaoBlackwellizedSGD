#!/bin/bash

for lr in 5e-3 1e-3 5e-4 1e-4 5e-5
do
	for topk in 1 3 5
	do
		python ../run_moving_mnist_training.py \
					--epochs 20 \
					--batch_size 64 \
					--seed 901 \
					--outdir '../mnist_vae_results/tuning_results/'\
					--outfilename moving_mnist_vae_reinforce_doublebs_topk${topk}_lr${lr} \
					--grad_estimator 'reinforce_double_bs'\
					--eval_test_set False \
					--propn_sample 0.1 \
					--learning_rate ${lr} \
					--save_every 2000 \
					--print_every 20 \
					--topk $topk
	done
done
