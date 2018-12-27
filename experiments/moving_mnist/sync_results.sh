rsync -avL --progress -e 'ssh -i ./../../../bryans_key_oregon.pem' \
   ubuntu@ec2-18-237-239-60.us-west-2.compute.amazonaws.com:/home/ubuntu/astronomy/discrete_vae_experimentation/moving_mnist_experiments/mnist_vae_results/.\
   /home/runjing_liu/Documents/astronomy/discrete_vae_experimentation/moving_mnist_experiments/mnist_vae_results/.

