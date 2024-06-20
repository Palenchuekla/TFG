import torch

class BCEL_from_logit_positive_class(torch.nn.Module):
    '''
    Custom loss function to compute the binary cross entropy loss form the logit of the positive class.
    '''
    def __init__(
        self,
        reduction: str = "mean",
    ):
        super(BCEL_from_logit_positive_class, self).__init__()
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ):
        # Cross Entropy Loss of every image of the batch --> (N)
        criterion = torch.nn.BCEWithLogitsLoss(reduction=self.reduction)
        loss = criterion(logits, labels.unsqueeze(dim=-1).float())

        return loss