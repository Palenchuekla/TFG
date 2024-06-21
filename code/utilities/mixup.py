import torch
from torch.distributions import Beta
from torch.utils.data import DataLoader
from torchvision.transforms import v2

class CustomMixUP(torch.nn.Module):
    '''
    Custom PyTorch Data Transformation.
    Personal implementation of MixUp data augmentation technique https://arxiv.org/abs/1710.09412.
    To summarize it, images and labels of a batch are transformed by linary combinig them with each other.
    Every image of the batch is paired with another image. A new image is created by making a weighted sum for every pixel component. Same for new label. 
    The coefficient (called lambda) for the weighted combination is extracted from a beta distribution where alpha=beta.
    '''
    def __init__(self, alpha = 1.0, first = False, lambdas = None, index = None):
        '''
         Parameters
        ----------
        - alpha: 
            Defines the beta distribution where coefficients are extracted from.
        - first:
            Wheter this transformation will be applied after (False) or before (True) other transformations. This is a special transformation because is applied per batch, not per iamge (as usual data transformations do).
        - lambdas:
            Internal use only. PyTorch tensor. Must match dataloader size. If specified, alpha is ignored and lambdas are taken from this tensor.
        - index:
            Internal use only. PyTorch tensor. Defines the pairing of the images. The i-th image of the batch will be paired with the image of the batch corresponding to the i-th index.

        Returns
        ----------
        A PyTorch trasnform object.

        '''
        self.alpha = alpha
        self.beta_dist = Beta(self.alpha, self.alpha)
        self.lambdas = lambdas
        self.index = index
        self.first = first
        super().__init__()
        
    def forward(self, batch_images, batch_labels):
        with torch.no_grad():
            # Lambdas
            batch_size = len(batch_images)
            if (self.lambdas == None):
                lambdas   = self.beta_dist.sample(sample_shape=[batch_size])
            else:
                lambdas = self.lambdas
            # Images and labels will be paired: { [0, index[0]], [1, index[1]], ... , [n, index[n]] } 
            # An image can be combined with itself.
            if self.index == None:
                index = torch.randperm(batch_size)
            else:
                index = self.index
            # Images
            lambdas_images = lambdas.view(len(lambdas), 1, 1, 1)
            t_images = lambdas_images*batch_images + (1-lambdas_images)*batch_images[index]
            # Labels
            lambdas_labels = lambdas.view(len(lambdas), 1)
            one_hot_labels = torch.nn.functional.one_hot(batch_labels)
            t_labels = lambdas_labels*one_hot_labels + (1-lambdas_labels)*one_hot_labels[index]
            return t_images, t_labels