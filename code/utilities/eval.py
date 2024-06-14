import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from utilities.common import specifity_score
from utilities.common import sensitivity_score

def evaluate(
        model : torch.nn, 
        inputs : torch.tensor, 
        labels : torch.tensor, 
        metrics : dict ={
        'acc':accuracy_score,
        'sen':sensitivity_score,
        'spe':specifity_score,
        'cm':confusion_matrix
        }
    ):
    '''
    Function to evaluate a PyTorch binary classifier.

    Parameters
    ----------
    - model:
        PyTorch model. Must output the logit of the positive class.
    - input:
        Tensor of N instances with dimension (N, ...), where "..." are a single instance dimmesions.
    - labels:
        Tensor of N labels. Labels must be either 0 or 1.
    - metrics:
        Dictionary to specificy the metrics to be evalued. The key is the identifier of the metric. 
        The associated value is the name of the funtion to call to compute the metric. 
        The metric function must accept two parameters: true labels and predicted labels.
    Returns
    -------
    - pd.DataFrame:
      pd.DataFrame containing the evaluation results for all emtrics specified.
    '''
    # Modify the behavior of certain model layers (generally, dropout and normalization).
    model.eval()
    # Computational graph is not computed (required_grad = false). No need to compute gradients through the chain rule.
    results = {}
    with torch.no_grad():
        logits = model(inputs)
        pred_labels = torch.nn.Sigmoid()(logits).round().to(int).squeeze().tolist()
        for m in metrics.keys():
            results[m] = [metrics[m](labels, pred_labels)]
    return results
