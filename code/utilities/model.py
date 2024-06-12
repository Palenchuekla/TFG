import torch
from torch import nn

class SingleLogitResNet(torch.nn.Module):
    '''
    Convolutional Neural Net where the feature extractor is a ResNet and the fully connected head is a linear layer with a single output.
    For a classification problem, this single logit would be the score of the positive class.
    For a regression problem, this single logit would be the predicted age.
    Constructor requires as a parameter a torchvision resnet model: from torchvision import models.resnet18(weights='IMAGENET1K_V1') or models.resnet50(weights='IMAGENET1K_V1').
    '''
    def __init__(self, resnet):
        '''
        Instantiates a CNN where the feature extractor is taken from the specified resnet and the fully conected head has no hidden layers and the last layers ouputs a single value.
        Parameters
        ----------
        - resnet
            torchvision resnet model.
        '''
        super(SingleLogitResNet, self).__init__()
        # Pretrained model
        pretrained_model = resnet
        # Feature Extractor
        self.feature_extractor = torch.nn.Sequential()
        for name, module in pretrained_model.named_children():
            if name != 'fc':
                self.feature_extractor.add_module(name=name, module=module)
        # Fully Connected Head
        ending_features = pretrained_model.fc.in_features
        self.fc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=ending_features, out_features=1, bias=True),
            )
        
    def get_n_params(self):
        '''
        Returns the number of parameters of the model.
        '''
        return sum([p.numel() for p in self.parameters()])

    def forward(self, x, mixup = False):
        x = self.feature_extractor(x)
        x = self.fc_head(x)
        return x