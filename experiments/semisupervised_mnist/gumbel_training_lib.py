import torch
from torch import optim

from itertools import cycle

import time

import numpy as np

import vae_utils
from semisuper_vae_training_lib import get_supervised_loss

import sys
sys.path.insert(0, '../../../rb_utils/')
sys.path.insert(0, '../../rb_utils/')
import gumbel_softmax_lib as gs_lib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval_gumbel_vae(vae, classifier, loader_unlabeled,
                            annealing_fun = None, init_step = 0,
                            loader_labeled = [None],
                            train = False, optimizer = None,
                            train_labeled_only = False):

    if train:
        assert optimizer is not None
        vae.train()
        classifier.train()

    else:
        vae.eval()
        classifier.eval()

    sum_loss = 0.0
    num_images = 0.0

    for labeled_data, unlabeled_data in zip(cycle(loader_labeled), \
                                                loader_unlabeled):

        unlabeled_image = unlabeled_data['image'].to(device)

        if labeled_data is not None:
            labeled_image = labeled_data['image'].to(device)
            true_labels = labeled_data['label'].to(device)

            # get labeled portion of loss
            supervised_loss = \
                get_supervised_loss(vae, classifier, labeled_image, true_labels)

            num_labeled = len(loader_labeled.sampler)

        else:
            supervised_loss = torch.Tensor([0.0])
            num_labeled = 0.0

        # run through classifier
        log_q = classifier.forward(unlabeled_image)

        if train:
            # flush gradients
            optimizer.zero_grad()

            if not train_labeled_only:
                # get unlabeled pseudoloss

                # sample from gumbel
                temperature = annealing_fun(init_step)
                softmax_sample = gs_lib.gumbel_softmax(log_q, temperature)
                init_step += 1

                # get loss
                unlabeled_gumbel_loss =\
                    vae_utils.get_loss_from_one_hot_label(vae, unlabeled_image,
                                                    softmax_sample)

                kl_q = torch.sum(torch.exp(log_q) * log_q, dim = 1)

                unlabeled_gumbel_loss += kl_q

            else:
                unlabeled_gumbel_loss = 0.0

            total_ps_loss = \
                unlabeled_gumbel_loss.mean() * len(loader_unlabeled.sampler) + \
                supervised_loss.mean() * num_labeled

            # backprop gradients from pseudo loss
            total_ps_loss.backward()
            optimizer.step()

        # loss at MAP value of z
        loss = \
            vae_utils.get_labeled_loss(vae, unlabeled_image,
                                torch.argmax(log_q, dim = 1)).detach().sum()

        sum_loss += loss
        num_images += unlabeled_image.shape[0]

    return sum_loss / num_images, init_step


def train_gumbel_vae(vae, classifier,
                train_loader, test_loader,
                optimizer, annealing_fun,
                loader_labeled = [None],
                train_labeled_only = False,
                epochs=10,
                save_every = 10,
                print_every = 10,
                outfile='./ss_mnist'):

    # initial losses
    init_train_loss = eval_gumbel_vae(vae, classifier, train_loader)[0]
    init_train_accuracy = vae_utils.get_classification_accuracy(classifier, train_loader)
    print('init train loss: {} || init train accuracy: {}'.format(
                init_train_loss, init_train_accuracy))

    train_loss_array = [init_train_loss]
    batch_losses = [init_train_loss]
    train_accuracy_array = [init_train_accuracy]

    init_test_loss = eval_gumbel_vae(vae, classifier, test_loader)[0]
    init_test_accuracy = vae_utils.get_classification_accuracy(classifier, test_loader)
    print('init test loss: {} || init test accuracy: {}'.format(
                init_test_loss, init_test_accuracy))

    test_loss_array = [init_test_loss]
    test_accuracy_array = [init_test_accuracy]

    epoch_start = 1
    step = 0
    for epoch in range(epoch_start, epochs+1):

        t0 = time.time()

        loss, step = eval_gumbel_vae(vae, classifier, train_loader,
                            annealing_fun = annealing_fun,
                            init_step = step,
                            loader_labeled = loader_labeled,
                            train = True,
                            optimizer = optimizer,
                            train_labeled_only = train_labeled_only)

        elapsed = time.time() - t0
        print('[{}] unlabeled_loss: {:.10g}  \t[{:.1f} seconds]'.format(\
                    epoch, loss, elapsed))
        batch_losses.append(loss)
        np.save(outfile + '_batch_losses', np.array(batch_losses))

        # print stuff
        if epoch % print_every == 0:
            # save the checkpoint.
            train_loss = eval_gumbel_vae(vae, classifier, train_loader)[0]
            test_loss = eval_gumbel_vae(vae, classifier, test_loader)[0]

            print('train loss: {}'.format(train_loss) + \
                    ' || test loss: {}'.format(test_loss))

            train_loss_array.append(train_loss)
            test_loss_array.append(test_loss)
            np.save(outfile + '_train_losses', np.array(train_loss_array))
            np.save(outfile + '_test_losses', np.array(test_loss_array))


            train_accuracy = vae_utils.get_classification_accuracy(classifier, train_loader)
            test_accuracy = vae_utils.get_classification_accuracy(classifier, test_loader)

            print('train accuracy: {}'.format(train_accuracy) + \
                    ' || test accuracy: {}'.format(test_accuracy))

            train_accuracy_array.append(train_accuracy)
            test_accuracy_array.append(test_accuracy)
            np.save(outfile + '_train_accuracy', np.array(train_accuracy_array))
            np.save(outfile + '_test_accuracy', np.array(test_accuracy_array))

        if epoch % save_every == 0:
            outfile_epoch = outfile + '_vae_epoch' + str(epoch)
            print("writing the vae parameters to " + outfile_epoch + '\n')
            torch.save(vae.state_dict(), outfile_epoch)

            outfile_epoch = outfile + '_classifier_epoch' + str(epoch)
            print("writing the classifier parameters to " + outfile_epoch + '\n')
            torch.save(classifier.state_dict(), outfile_epoch)

    outfile_final = outfile + '_vae_final'
    print("writing the vae parameters to " + outfile_final + '\n')
    torch.save(vae.state_dict(), outfile_final)

    outfile_final = outfile + '_classifier_final'
    print("writing the classifier parameters to " + outfile_final + '\n')
    torch.save(classifier.state_dict(), outfile_final)
