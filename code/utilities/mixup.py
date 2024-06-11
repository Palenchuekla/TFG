import torch
from torch.distributions import Beta
from torch.utils.data import DataLoader
from torchvision.transforms import v2

class CustomMixUP(torch.nn.Module):
    def __init__(self, alpha = 1.0, first = False, lambdas = None, index = None):
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