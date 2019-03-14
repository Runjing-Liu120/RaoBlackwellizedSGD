import argparse
import os
import distutils.util

import numpy as np

import torch

import torch.optim as optim

from copy import deepcopy

import mnist_data_lib
import mnist_vae_lib

import semisuper_vae_training_lib as ss_lib

import sys
sys.path.insert(0, '../../../rb_utils/')
sys.path.insert(0, '../../rb_utils/')
import baselines_lib as bs_lib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
                    help='type of gradient estimator. One of reinforce, reinforce_double_bs, relax, gumbel, nvil')

# whether to only train on labeled data
parser.add_argument('--train_labeled_only',
                    type=distutils.util.strtobool, default='False')

# whether to evaluate on test set
parser.add_argument('--eval_test_set',
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

# warm start parameters
parser.add_argument('--use_vae_init',
                    type=distutils.util.strtobool, default='False',
                    help='whether to initialize the mnist vae')
parser.add_argument('--vae_init_file',
                    type=str,
                    help='file to initialize the mnist vae')
parser.add_argument('--use_classifier_init',
                    type=distutils.util.strtobool, default='False',
                    help='whether to initialize the mnist vae')
parser.add_argument('--classifier_init_file',
                    type=str,
                    help='file to initialize the classifier vae')

# Other params
parser.add_argument('--seed', type=int, default=4254,
                    help='random seed')


# Gradient parameters
parser.add_argument('--rebar_eta', type = float, default = 1e-5)
parser.add_argument('--gumbel_anneal_rate', type = float, default = 5e-5)

args = parser.parse_args()
print('learning rate', args.learning_rate)

assert os.path.exists(args.outdir)

print('seed: ', args.seed)
np.random.seed(args.seed)
_ = torch.manual_seed(args.seed)

# get data
# train sets
train_set_labeled, train_set_unlabeled, test_set = \
        mnist_data_lib.get_mnist_dataset_semisupervised(
                            data_dir = '../mnist_data/',
                            train_test_split_folder = '../test_train_splits/',
                            eval_test_set = args.eval_test_set)

train_loader_labeled = torch.utils.data.DataLoader(
                 dataset=train_set_labeled,
                 batch_size=args.batch_size,
                 shuffle=True)

if args.train_labeled_only:
    train_loader_unlabeled = deepcopy(train_loader_labeled)
else:
    train_loader_unlabeled = torch.utils.data.DataLoader(
                     dataset=train_set_unlabeled,
                     batch_size=args.batch_size,
                     shuffle=True)

test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=args.batch_size,
                shuffle=False)

# setup VAE
slen = train_set_labeled[0]['image'].shape[0]

vae, classifier = mnist_vae_lib.get_mnist_vae_and_classifier(
                        latent_dim = 8,
                        n_classes = 10,
                        slen = slen)

# get warm starts
if args.use_vae_init:
    assert os.path.isfile(args.vae_init_file)
    vae.load_state_dict(torch.load(args.vae_init_file,
                        map_location=lambda storage, loc: storage))

if args.use_classifier_init:
    assert os.path.isfile(args.classifier_init_file)
    classifier.load_state_dict(torch.load(args.classifier_init_file,
                        map_location=lambda storage, loc: storage))

vae.to(device)
classifier.to(device)


# set up optimizer
optimizer = optim.Adam([
                {'params': classifier.parameters(), 'lr': args.learning_rate}, #1e-3},
                {'params': vae.parameters(), 'lr': args.learning_rate}],
                weight_decay=args.weight_decay)
bs_optimizer = None

if args.grad_estimator == 'reinforce':
    grad_estimator = bs_lib.reinforce
    grad_estimator_kwargs = {'grad_estimator_kwargs': None}
elif args.grad_estimator == 'reinforce_double_bs':
    grad_estimator = bs_lib.reinforce_w_double_sample_baseline
    grad_estimator_kwargs = {'grad_estimator_kwargs': None}
elif args.grad_estimator == 'relax':
    grad_estimator = bs_lib.relax
    print('eta: ', args.rebar_eta)
    temperature_param = torch.Tensor([1]).to(device).requires_grad_(True)
    c_phi = bs_lib.RELAXBaseline(10).to(device)
    grad_estimator_kwargs = {'temperature': temperature_param,
                            'eta': args.rebar_eta,
                            'c_phi': c_phi}

    bs_optimizer = optim.Adam([{'params': [temperature_param]},
                            {'params': c_phi.parameters()}], lr = 1e-2)

elif args.grad_estimator == 'gumbel':
    grad_estimator = bs_lib.gumbel
    print('annealing rate: ', args.gumbel_anneal_rate)
    grad_estimator_kwargs = {'annealing_fun': lambda t : \
                        np.maximum(0.5, \
                        np.exp(- args.gumbel_anneal_rate* float(t) * \
                    len(train_loader_labeled.sampler) / args.batch_size)),
                    'straight_through': False}

elif args.grad_estimator == 'nvil':
    grad_estimator = bs_lib.nvil
    baseline_nn = bs_lib.BaselineNN(slen = slen)
    grad_estimator_kwargs = {'baseline_nn': baseline_nn.to(device)}

    optimizer = optim.Adam([
                    {'params': classifier.parameters(), 'lr': args.learning_rate},
                    {'params': vae.parameters(), 'lr': args.learning_rate},
                    {'params': baseline_nn.parameters(), 'lr': args.learning_rate}],
                    weight_decay=args.weight_decay)

else:
    print('invalid gradient estimator')
    raise NotImplementedError

# train!
outfile = args.outdir + args.outfilename
ss_lib.train_semisuper_vae(vae, classifier,
                train_loader_unlabeled,
                test_loader,
                optimizer,
                train_loader_labeled,
                topk = args.topk,
                n_samples = args.n_samples,
                grad_estimator = grad_estimator,
                grad_estimator_kwargs = grad_estimator_kwargs,
                epochs=args.epochs,
                outfile = outfile,
                save_every = args.save_every,
                print_every = args.print_every,
                train_labeled_only = args.train_labeled_only,
                baseline_optimizer = bs_optimizer)
