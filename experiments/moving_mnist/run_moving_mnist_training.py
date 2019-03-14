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

import sys
sys.path.insert(0, '../../../rb_utils/')
sys.path.insert(0, '../../rb_utils/')
import baselines_lib as bs_lib

parser = argparse.ArgumentParser(description='VAE')

# Training parameters
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--weight_decay', type = float, default = 1e-5)
parser.add_argument('--learning_rate', type = float, default = 0.001)

parser.add_argument('--topk', type = int, default = 0)
parser.add_argument('--n_samples', type = int, default = 1)

parser.add_argument('--grad_estimator',
                    type=str, default='reinforce',
                    help='type of gradient estimator. One of reinforce, reinforce_double_bs, ')

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
parser.add_argument('--print_every', type = int, default = 10)

# Whether to just work with subset of data
parser.add_argument('--propn_sample', type = float,
                    help='proportion of dataset to use',
                    default = 1.0)
# whether to evaluate on test set
parser.add_argument('--eval_test_set',
                    type=distutils.util.strtobool, default='False')

# warm start parameters
parser.add_argument('--use_vae_init',
                    type=distutils.util.strtobool, default='False',
                    help='whether to initialize the mnist vae')
parser.add_argument('--vae_init_file',
                    type=str,
                    help='file to initialize the mnist vae')
parser.add_argument('--train_attn_only',
                    type=distutils.util.strtobool, default='False')

# Other params
parser.add_argument('--seed', type=int, default=4254,
                    help='random seed')

# Gradient parameters
parser.add_argument('--rebar_eta', type = float, default = 1e-5)
parser.add_argument('--gumbel_anneal_rate', type = float, default = 5e-5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

def validate_args():
    assert os.path.exists(args.outdir)

    if args.use_vae_init:
        assert os.path.isfile(args.vae_init_file)

validate_args()

np.random.seed(901)
_ = torch.manual_seed(901)
print('seed: ', args.seed)
# LOAD DATA
data_dir = '../mnist_data/'

train_set = mnist_data_utils.MovingMNISTDataSet(data_dir = data_dir,
                        indices = np.load('../train_indx.npy'),
                        train_set = True)

if args.eval_test_set:
    test_set = mnist_data_utils.MovingMNISTDataSet(data_dir = data_dir,
                            propn_sample = 0.6,
                            train_set = False)
else:
    test_set = mnist_data_utils.MovingMNISTDataSet(data_dir = data_dir,
                            indices = np.load('../val_indx.npy'),
                            train_set = True)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=args.batch_size,
                 shuffle=True)

test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=args.batch_size,
                shuffle=False)

print('num train: ', len(train_loader.dataset))

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
                    {'params': vae.pixel_attention.parameters(),
                    'lr': args.learning_rate,
                    'weight_decay': args.weight_decay},
		{'params': vae.mnist_vae.parameters(),
		'lr': args.learning_rate,
		'weight_decay': args.weight_decay}])
print('lr: ', args.learning_rate)

### Gradient estimator
if args.grad_estimator == 'reinforce':
    grad_estimator = bs_lib.reinforce
    grad_estimator_kwargs = {'grad_estimator_kwargs': None}
elif args.grad_estimator == 'reinforce_double_bs':
    grad_estimator = bs_lib.reinforce_w_double_sample_baseline
    grad_estimator_kwargs = {'grad_estimator_kwargs': None}
elif args.grad_estimator == 'relax':
    grad_estimator = bs_lib.relax
    temperature_param = torch.Tensor([1]).to(device).requires_grad_(True)
    c_phi = bs_lib.RELAXBaseline(68**2).to(device)
    grad_estimator_kwargs = {'temperature': temperature_param,
                            'eta': args.rebar_eta,
                            'c_phi': c_phi}
    bs_optimizer = optim.Adam([{'params': [temperature_param]},
                            {'params': c_phi.parameters()}], lr = 1e-2)

elif args.grad_estimator == 'gumbel':
    print('gumbel anneal rate: ', args.gumbel_anneal_rate)
    grad_estimator = bs_lib.gumbel
    grad_estimator_kwargs = {'annealing_fun': lambda t : \
                        np.maximum(0.5, \
                        np.exp(- args.gumbel_anneal_rate* float(t) * \
                            len(train_loader.sampler) / args.batch_size)),
                            'straight_through': True}


elif args.grad_estimator == 'nvil':
    grad_estimator = bs_lib.nvil
    baseline_nn = bs_lib.BaselineNN(slen = train_set[0]['image'].shape[-1]);
    grad_estimator_kwargs = {'baseline_nn': baseline_nn.to(device)}

    optimizer = optim.Adam([
                    {'params': vae.pixel_attention.parameters(),
                    'lr': args.learning_rate,
                    'weight_decay': args.weight_decay},
                {'params': vae.mnist_vae.parameters(),
                'lr': args.learning_rate,
                'weight_decay': args.weight_decay},
                {'params': baseline_nn.parameters(),
                'lr': args.learning_rate,
                'weight_decay': args.weight_decay}])


else:
    print('invalid gradient estimator')
    raise NotImplementedError


# TRAIN!
print('training vae')

t0_train = time.time()

outfile = os.path.join(args.outdir, args.outfilename)

np.random.seed(args.seed)
_ = torch.manual_seed(args.seed)

vae_training_lib.train_vae(vae, train_loader, test_loader, optimizer,
                grad_estimator = grad_estimator,
                grad_estimator_kwargs = grad_estimator_kwargs,
                topk = args.topk,
                n_samples = args.n_samples,
                set_true_loc = args.set_true_loc,
                outfile = outfile,
                n_epoch = args.epochs,
                print_every = args.print_every,
                save_every = args.save_every,
                baseline_optimizer = bs_optimizer)


print('done. Total time: {}secs'.format(time.time() - t0_train))
