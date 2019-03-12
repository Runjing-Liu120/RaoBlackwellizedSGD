rsync -avL --progress -e 'ssh -i ./../../../../bryans_key_oregon.pem' \
   ubuntu@ec2-54-70-65-134.us-west-2.compute.amazonaws.com:/home/ubuntu/astronomy/RaoBlackwellizedSGD/experiments/semisupervised_mnist/mnist_vae_results2/. \
   /home/runjing_liu/Documents/astronomy/RaoBlackwellizedSGD/experiments/semisupervised_mnist/mnist_vae_results2/.

