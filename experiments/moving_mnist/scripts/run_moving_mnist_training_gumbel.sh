#!/bin/bash

for i in {1..10}
do
((seed=$i + 11129))

python ../run_moving_mnist_training.py \
			--epochs 50 \
			--batch_size 64 \
			--seed $seed \
			--outdir '../mnist_vae_results/'\
			--outfilename moving_mnist_vae_gumbel_trial$i \
			--grad_estimator 'gumbel' \
			--propn_sample 0.1 \
			--learning_rate 5e-5 \
			--gumbel_anneal_rate 5e-4 \
			--save_every 233 \
			--print_every 2 \
			--topk 0
done
