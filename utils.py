import torch
import torchvision
import numpy as np
from torchvision import transforms

def random_batch_gen(batch_size):
    real_data_iter = get_train_set(batch_size)
    ep = 0
    while True:
        for batch in real_data_iter:
            yield batch, ep
        ep += 1

def get_train_set(batch_size):
    '''
    Returns a DataLoader for the dataset with the specified batch size

    Parameters
    ----------
    batch_size : int
        Batch size of train loader

    Returns
    -------
    trainloader : DataLoader

    '''
    transform = transforms.Compose([transforms.Resize(32),
        transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(
            root='../data', train=True, download=True, 
            transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    return trainloader
                                                                
def latent_samples(batch_size, latent_dim):
    '''Returns iid samples from the standard normal distribution in the shape 
    (batch size, latent_dim)'''

    samples =  torch.from_numpy(np.random.randn(batch_size, latent_dim, 1, 1))
    return samples.float()

def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False
    
def unfreeze_params(model):
    for param in model.parameters():
        param.requires_grad = True

def preprocess_batch(batch, flatten=True):
    '''
    Flattens a batch of images and scales their values to the range [-1, 1]

    Parameters
    ----------
    batch : Tensor of shape (N, ...)
        Batch of images. Elements are in the range[0,1]

    Returns
    -------
    batch : Tensor of shape (N, M)

    '''
    if flatten:
        batch = batch.view(batch.shape[0], -1) #flatten
    batch = 2*batch-1 #output between -1,1
    return batch

def deprocess_batch(batch, image_shape, unflatten=True):
    '''
    Inverse of preprocess_batch function. Reshapes batch of flattened images 
    to be of shape (batch_size, image_shape)
    and scales their output to the range [0, 1]

    Parameters
    ----------
    batch : Tensor of shape (N, M)
        Batch of flattened images. Elements in the range [-1, 1]
    image_shape : tuple of ints

    Returns
    -------
    batch : Tensor of shape (N, image_shape)

    '''
    if unflatten:
        batch = batch.view(batch.shape[0], *image_shape)
    batch = (batch+1)/2
    return batch


