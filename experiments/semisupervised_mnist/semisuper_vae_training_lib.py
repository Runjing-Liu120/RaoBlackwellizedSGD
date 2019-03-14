import torch
from torch import optim
from torch.autograd import grad

from itertools import cycle

import time

import numpy as np

import vae_utils

import sys
sys.path.insert(0, '../../../rb_utils/')
sys.path.insert(0, '../../rb_utils/')
import rao_blackwellization_lib as rb_lib
import baselines_lib as bs_lib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_correct_classifier(image, true_labels, n_classes, fudge_factor = 1e-12):
    # for debugging only: returns q with mass on the correct label

    batch_size = image.shape[0]
    q = torch.zeros((batch_size, n_classes)) + fudge_factor
    seq_tensor = torch.LongTensor([i for i in range(batch_size)])
    q[seq_tensor, true_labels] = 1 - fudge_factor * (n_classes - 1)

    assert np.all((q > 0).detach().numpy())

    return torch.log(q).to(device)

def get_supervised_loss(vae, classifier, labeled_image, true_labels):
    # get labeled loss
    labeled_loss = \
        vae_utils.get_labeled_loss(vae, labeled_image, true_labels)

    # cross entropy term
    labeled_log_q = classifier.forward(labeled_image)

    cross_entropy = \
        vae_utils.get_class_label_cross_entropy(labeled_log_q,
                                true_labels)

    return labeled_loss + cross_entropy

def eval_semisuper_vae(vae, classifier, loader_unlabeled,
                        loader_labeled = [None],
                        train = False, optimizer = None,
                        topk = 0,
                        grad_estimator = bs_lib.reinforce,
                        grad_estimator_kwargs = {'grad_estimator_kwargs': None},
                        n_samples = 1,
                        train_labeled_only = False, epoch = 0,
                        baseline_optimizer = None):

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
                get_supervised_loss(vae, classifier, labeled_image,
                                                        true_labels).sum()

            num_labeled = len(loader_labeled.sampler)
            num_labeled_batch = labeled_image.shape[0]

        else:
            supervised_loss = 0.0
            num_labeled = 0.0
            num_labeled_batch = 1.0

        # run through classifier
        log_q = classifier.forward(unlabeled_image)

        if train:

            train_labeled_only_bool = 1.
            if train_labeled_only:
                n_samples = 0
                train_labeled_only_bool = 0.

            # flush gradients
            optimizer.zero_grad()

            # get unlabeled pseudoloss
            f_z = lambda z : vae_utils.get_loss_from_one_hot_label(vae,
                                unlabeled_image, z)
            unlabeled_ps_loss = 0.0
            for i in range(n_samples):
                unlabeled_ps_loss_ = rb_lib.get_raoblackwell_ps_loss(f_z, log_q,
                                topk = topk,
                                epoch = epoch,
                                data = unlabeled_image,
                                grad_estimator = grad_estimator,
                                grad_estimator_kwargs = grad_estimator_kwargs)

                unlabeled_ps_loss += unlabeled_ps_loss_

            unlabeled_ps_loss = unlabeled_ps_loss / max(n_samples, 1)

            kl_q = torch.sum(torch.exp(log_q) * log_q)

            total_ps_loss = \
                (unlabeled_ps_loss + kl_q) * train_labeled_only_bool * \
                len(loader_unlabeled.sampler) / unlabeled_image.shape[0] + \
                supervised_loss * num_labeled / labeled_image.shape[0]

            # backprop gradients from pseudo loss
            total_ps_loss.backward(retain_graph = True)
            optimizer.step()

            if baseline_optimizer is not None:
                # flush gradients
                optimizer.zero_grad()
                # for params in classifier.parameters():
                baseline_optimizer.zero_grad()
                loss_grads = grad(total_ps_loss, classifier.parameters(),
                                    create_graph=True)
                gn2 = sum([grd.norm()**2 for grd in loss_grads])
                gn2.backward()
                baseline_optimizer.step()

        # loss at MAP value of z
        loss = \
            vae_utils.get_labeled_loss(vae, unlabeled_image,
                                torch.argmax(log_q, dim = 1)).detach().sum()

        sum_loss += loss
        num_images += unlabeled_image.shape[0]

    return sum_loss / num_images


def train_semisuper_vae(vae, classifier,
                train_loader, test_loader,
                optimizer,
                loader_labeled = [None],
                train_labeled_only = False,
                topk = 0, n_samples = 1,
                grad_estimator = bs_lib.reinforce,
                grad_estimator_kwargs = {'grad_estimator_kwargs': None},
                epochs=10,
                save_every = 10,
                print_every = 10,
                outfile='./ss_mnist',
                baseline_optimizer = None):

    # initial losses
    init_train_loss = eval_semisuper_vae(vae, classifier, train_loader)
    init_train_accuracy = vae_utils.get_classification_accuracy(classifier, train_loader)
    print('init train loss: {} || init train accuracy: {}'.format(
                init_train_loss, init_train_accuracy))

    train_loss_array = [init_train_loss]
    batch_losses = [init_train_loss]
    train_accuracy_array = [init_train_accuracy]

    init_test_loss = eval_semisuper_vae(vae, classifier, test_loader)
    init_test_accuracy = vae_utils.get_classification_accuracy(classifier, test_loader)
    print('init test loss: {} || init test accuracy: {}'.format(
                init_test_loss, init_test_accuracy))

    test_loss_array = [init_test_loss]
    test_accuracy_array = [init_test_accuracy]

    epoch_start = 1
    t0 = time.time()
    batch_timing = [0.0]
    test_timing = [t0]
    for epoch in range(epoch_start, epochs+1):

        t0 = time.time()
        loss = eval_semisuper_vae(vae, classifier, train_loader,
                            loader_labeled = loader_labeled,
                            topk = topk,
                            n_samples = n_samples,
                            grad_estimator = grad_estimator,
                            grad_estimator_kwargs = grad_estimator_kwargs,
                            train = True,
                            optimizer = optimizer,
                            train_labeled_only = train_labeled_only,
                            epoch = epoch,
                            baseline_optimizer = baseline_optimizer)

        elapsed = time.time() - t0
        print('[{}] unlabeled_loss: {:.10g}  \t[{:.1f} seconds]'.format(\
                    epoch, loss, elapsed))
        batch_losses.append(loss)
        batch_timing.append(elapsed)

        np.save(outfile + '_batch_losses', np.array(batch_losses))
        np.save(outfile + '_batch_timing', np.array(batch_timing))

        # print stuff
        if epoch % print_every == 0:
            # save the checkpoint.
            train_loss = eval_semisuper_vae(vae, classifier, train_loader)
            test_loss = eval_semisuper_vae(vae, classifier, test_loader)

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

            test_timing.append(time.time())
            np.save(outfile + '_test_timing', np.array(test_timing))

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
