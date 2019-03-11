import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_one_hot_encoding_from_int(z, n_classes):

    z_one_hot = torch.zeros(len(z), n_classes).to(device)
    z_one_hot.scatter_(1, z.view(-1, 1), 1)
    z_one_hot = z_one_hot.view(len(z), n_classes)

    return z_one_hot
