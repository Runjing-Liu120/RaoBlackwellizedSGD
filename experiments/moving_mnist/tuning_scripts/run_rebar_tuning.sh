#!/bin/bash

for lr in 5e-3 1e-3 5e-4 1e-4 5e-5
do
	for eta in 0.2 0.4 0.6 0.8 1.0
	do

		python ../run_moving_mnist_training.py \
					--epochs 20 \
					--batch_size 64 \
					--seed 901 \
					--outdir '../mnist_vae_results/tuning_results/'\
					--outfilename moving_mnist_vae_rebar_lr${lr}_eta${eta} \
					--grad_estimator 'rebar' \
					--propn_sample 0.1 \
					--learning_rate ${lr} \
					--rebar_eta ${eta} \
					--save_every 200 \
					--print_every 20 \
					--topk 0
	done
done
