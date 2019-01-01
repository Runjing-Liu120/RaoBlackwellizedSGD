import numpy as np
import os

import torch
print('torch version', torch.__version__)
import torch.optim as optim

import time

import mnist_data_utils
import mnist_vae_lib
import vae_training_lib

import distutils.util
import argparse

parser = argparse.ArgumentParser(description='VAE')

# Training parameters
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--weight_decay', type = float, default = 1e-5)
parser.add_argument('--learning_rate', type = float, default = 0.001)

parser.add_argument('--topk', type = int, default = 0)
parser.add_argument('--nsamples', type = int, default = 1)

# whether to use true location
parser.add_argument('--set_true_loc',
                    type=distutils.util.strtobool, default='False')

# saving vae
parser.add_argument('--outdir', type = str,
                    default='./', help = 'directory for saving encoder and decoder')
parser.add_argument('--outfilename', type = str,
                    default='moving_mnist_vae',
                    help = 'filename for saving the encoder and decoder')
parser.add_argument('--save_every', type = int, default = 50,
                    help='save encoder ever how ___ epochs (default = 50)')

# Whether to just work with subset of data
parser.add_argument('--propn_sample', type = float,
                    help='proportion of dataset to use',
                    default = 1.0)

# warm start parameters
parser.add_argument('--use_vae_init',
                    type=distutils.util.strtobool, default='False',
                    help='whether to initialize the mnist vae (but not the pixel attn)')
parser.add_argument('--vae_init_file',
                    type=str,
                    help='file to initialize the mnist vae')
parser.add_argument('--train_attn_only',
                    type=distutils.util.strtobool, default='False')

# Other params
parser.add_argument('--seed', type=int, default=4254,
                    help='random seed')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

def validate_args():
    assert os.path.exists(args.outdir)

    if args.use_vae_init:
        assert os.path.isfile(args.vae_init_file)

validate_args()

np.random.seed(args.seed)
_ = torch.manual_seed(args.seed)

# LOAD DATA
data_dir = '../mnist_data/'
propn_sample = args.propn_sample
train_set, test_set = \
    mnist_data_utils.get_moving_mnist_dataset(data_dir, propn_sample)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=args.batch_size,
                 shuffle=True)

test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=args.batch_size,
                shuffle=False)

# SET UP VAE
print('setting up VAE: ')
vae = mnist_vae_lib.MovingHandwritingVAE()

if args.use_vae_init:
    vae.load_state_dict(torch.load(args.vae_init_file,
                                   map_location=lambda storage, loc: storage))

vae.to(device)

# set up optimizer
if args.train_attn_only:
    optimizer = optim.Adam([
                    {'params': vae.pixel_attention.parameters(),
                    'lr': args.learning_rate,
                    'weight_decay': args.weight_decay}])
else:
    optimizer = optim.Adam([
                    {'params': vae.parameters(),
                    'lr': args.learning_rate,
                    'weight_decay': args.weight_decay}])

# TRAIN!
print('training vae')

t0_train = time.time()

outfile = os.path.join(args.outdir, args.outfilename)

vae_training_lib.train_vae(vae, train_loader, test_loader, optimizer,
                topk = args.topk,
                nsamples = args.nsamples,
                set_true_loc = args.set_true_loc,
                outfile = outfile,
                n_epoch = args.epochs,
                print_every = 10,
                save_every = args.save_every)


print('done. Total time: {}secs'.format(time.time() - t0_train))
