This folder contains the experiments on the moving MNIST problem. 

The folder `./scripts/` contains the scripts to fit the VAE, using the different gradient estimators discussed in our paper. The fits are saved to `./mnist_vae_results/`.  

Our VAE is implemented in `mnist_vae_lib.py`. 

After fitting the models, the results are parsed by jupyter notebooks in the `./jupyter/` folder. Reproductions of our figures can be found there. 
