import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_one_hot_encoding_from_int(z, n_classes):

    z_one_hot = torch.zeros(len(z), n_classes).to(device)
    z_one_hot.scatter_(1, z.view(-1, 1), 1)
    z_one_hot = z_one_hot.view(len(z), n_classes)

    return z_one_hot

# def get_one_hot_encoding_from_int(z, n_classes):
#     # z is a sequence of integers in {0, ...., n_classes}
#     #  corresponding to categorires
#     # we return a matrix of shape len(z) x n_classes
#     # corresponding to the one hot encoding of z
#
#     assert (torch.max(z) + 1) <= n_classes
#
#     batch_size = len(z)
#     one_hot_z = torch.zeros((batch_size, n_classes)).to(device)
#
#     for i in range(n_classes):
#         one_hot_z[z == i, i] = 1.
#
#     return one_hot_z
