import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Categorical

import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval_vae(vae, loader, \
                grad_estimator,
                grad_estimator_kwargs = {'grad_estimator_kwargs': None},
                optimizer = None,
                train = False,
                set_true_loc = False,
                topk = 0,
                n_samples = 1,
                epoch = 0,
                baseline_optimizer = None):
    if train:
        vae.train()
        assert optimizer is not None
    else:
        n_samples = 0
        vae.eval()

    avg_loss = 0.0

    num_images = len(loader.dataset); # print('num images', num_images)

    for batch_idx, data in enumerate(loader):

        if optimizer is not None:
            optimizer.zero_grad()

        image = data['image'].to(device)
        if set_true_loc:
            true_pixel_2d = data['pixel_2d'].to(device)
            n_samples = 1
            use_baseline = False
            topk = 0
        else:
            true_pixel_2d = None

        pm_loss, loss = vae.get_rb_loss(image,
                                topk = topk,
                                grad_estimator = grad_estimator,
                                grad_estimator_kwargs = grad_estimator_kwargs,
                                epoch = epoch,
                                n_samples = n_samples,
                                true_pixel_2d = true_pixel_2d)

        if train:
            pm_loss.backward(retain_graph = True)
            optimizer.step()

            if baseline_optimizer is not None:
                # This is used for RELAX, where we have a learned
                # baseline that needs to be optimized
                optimizer.zero_grad()
                baseline_optimizer.zero_grad()
                loss_grads = grad(total_ps_loss, classifier.parameters(),
                                    create_graph=True)
                gn2 = sum([grd.norm()**2 for grd in loss_grads])
                gn2.backward()
                baseline_optimizer.step()

        avg_loss += loss.data  / num_images

    return avg_loss

def train_vae(vae, train_loader, test_loader, optimizer,
                    grad_estimator,
                    grad_estimator_kwargs = {'grad_estimator_kwargs': None},
                    set_true_loc = False,
                    topk = 0,
                    n_samples = 1,
                    outfile = './mnist_vae_semisupervised',
                    n_epoch = 200, print_every = 10, save_every = 20,
                    baseline_optimizer = None):

    batch_losses = []
    train_losses = []
    test_losses = []

    # get a few images for debugging
    for batch_idx, d in enumerate(train_loader):
        data_train = d
        break

    images_debugging = data_train['image'].to(device)
    np.save(outfile + 'images_debugging', images_debugging.cpu().numpy())

    # get losses
    train_loss = eval_vae(vae, train_loader, grad_estimator,
                            grad_estimator_kwargs = grad_estimator_kwargs,
                            train = False,
                            set_true_loc = set_true_loc)
    test_loss = eval_vae(vae, test_loader, grad_estimator,
                            grad_estimator_kwargs = grad_estimator_kwargs,
                            train = False,
                            set_true_loc = set_true_loc)
    _, debugging_images_loss = vae.get_rb_loss(images_debugging,
                                    grad_estimator,
                                    grad_estimator_kwargs = grad_estimator_kwargs,
                                    n_samples = 0)

    print('  * init train recon loss: {:.10g};'.format(train_loss))
    print('  * init test recon loss: {:.10g};'.format(test_loss))
    print('  * debugging_images loss: {:.10g};'.format(debugging_images_loss / \
                                                    images_debugging.shape[0]))

    batch_losses.append(train_loss)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    t0 = time.time()
    batch_timing = [0.]
    test_timing = [t0]


    for epoch in range(1, n_epoch + 1):
        t0 = time.time()

        loss = eval_vae(vae, train_loader,
                                optimizer = optimizer,
                                train = True,
                                set_true_loc = set_true_loc,
                                topk = topk,
                                grad_estimator = grad_estimator,
                                grad_estimator_kwargs = grad_estimator_kwargs,
                                epoch = epoch,
                                n_samples = n_samples,
                                baseline_optimizer = None)

        elapsed = time.time() - t0
        print('[{}] unlabeled_loss: {:.10g}  \t[{:.1f} seconds]'.format(\
                    epoch, loss, elapsed))
        batch_losses.append(loss)
        np.save(outfile + '_batch_losses', np.array(batch_losses))

        batch_timing.append(elapsed)
        np.save(outfile + '_batch_timing', np.array(batch_timing))

        if epoch % print_every == 0:
            train_loss = eval_vae(vae, train_loader, grad_estimator,
                                grad_estimator_kwargs = grad_estimator_kwargs,
                                train = False,
                                set_true_loc = set_true_loc)
            test_loss = eval_vae(vae, test_loader, grad_estimator,
                                grad_estimator_kwargs = grad_estimator_kwargs,
                                train = False,
                                set_true_loc = set_true_loc)
            _, debugging_images_loss = vae.get_rb_loss(images_debugging,
                                grad_estimator,
                                grad_estimator_kwargs = grad_estimator_kwargs,
                                n_samples = 0)

            print('  * train recon loss: {:.10g};'.format(train_loss))
            print('  * test recon loss: {:.10g};'.format(test_loss))
            print('  * debugging_images loss: {:.10g};'.format(debugging_images_loss / \
                                                    images_debugging.shape[0]))

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            np.save(outfile + '_train_losses', np.array(train_losses))
            np.save(outfile + '_test_losses', np.array(test_losses))

            test_timing.append(time.time())
            np.save(outfile + '_test_timing', np.array(test_timing))


        if epoch % save_every == 0:
            outfile_every = outfile + '_epoch' + str(epoch)
            print("writing the parameters to " + outfile_every + '\n')
            torch.save(vae.state_dict(), outfile_every)

    outfile_final = outfile + '_final'
    print("writing the parameters to " + outfile_final + '\n')
    torch.save(vae.state_dict(), outfile_final)
