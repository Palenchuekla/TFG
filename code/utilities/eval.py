import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from utilities.common import specifity_score
from utilities.common import sensitivity_score
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2

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
    Function to evaluate a SingleLogitResNet classifier.

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

def pred(
        model : torch.nn, 
        inputs : torch.tensor, 
    ):
    '''
    Returns predictions of a SingleLogitResnet (un-normalized probability of the positive class) for a batch of images.

    Parameters
    ----------
    - model:
        SingleLogitResnet. Binary classfier. Must output the logit of the positive class.
    - input:
        Tensor of N instances with dimension (N, ...), where "..." are a single instance dimmesions.
    - labels:
        Tensor of N labels. Labels must be either 0 or 1.
    Returns
    -------
    - labels:
      1-D tensor containing the predicted labels.
    '''
    # Modify the behavior of certain model layers (generally, dropout and normalization).
    model.eval()
    # Computational graph is not computed (required_grad = false). No need to compute gradients through the chain rule.
    results = {}
    with torch.no_grad():
        logits = model(inputs)
        pred_labels = torch.nn.Sigmoid()(logits).round().to(int).squeeze().tolist()
    return pred_labels

def gradCAM(img, model, layer, target=None):
    '''
    Generates a activation map to visualize the most influential regions of an image for predicting a class based on the GradCAM technique.
    Parameters
    ----------
    - model:
        PyTroch Model.
    - img:
        PyTorch tensor. Image to predict and justify.
    - layer:
        PyTorch Module. Layer to perform GradCAM at.
    - target:
        Class to performance GradCAM for.
    Returns
    -------
    - p_filter:
      Generated CAM.
    - n_filer:
      Inverted CAM (usefull for SingleLogitResnet classifiers).
    '''
    # Set everything up
    layer_gc = LayerGradCam(forward_func=model, layer=layer)
    # Generate the CAM
    batch = img.unsqueeze(0).to('cuda')
    attr = layer_gc.attribute(batch, target=target)
    attr = attr[0]
    # Normalize the CAM
    attr = (attr - attr.min()) / (attr.max()-attr.min())
    attr = attr.detach().cpu()
    p_filter = attr
    # Generate inverted CAM
    n_filter = attr*-1
    # Normalize the inverted CAM
    n_filter = (n_filter - n_filter.min()) / (n_filter.max()-n_filter.min())
    # Plot CAMs
    return p_filter, n_filter


def justify(
        model,
        t,
        layer,
        img_path,
        target = None
        ):
        '''
        Predict and justify the predicted label of an image. 
        Plots a "justification" using the GradCAM (G_p=1) and inverted GradCAM (G_p=0) of the positive class.
        Keep in mind the the inverted GradCAM only matches the probability of the negative class only for SingleLogitResnet models.
        Parameters
        ----------
        - model:
            PyTorch Model.
        - img_path:
            Path of the iamge to predict.
        - layer:
            PyTorch Module. Layer to perform GradCAM at.
        - target:
            Integer. Class to performance GradCAM for.
        - t:
            PyTorch Transformation. Preprocessing done to images by the model.
        Returns
        -------
        - p_filter:
            Generated CAM. Usefull to detect regions that push the model to predict the target class.
        - n_filer:
            Inverted CAM. Usefull to detect regions that push the model to NOT predict the target class. Matches de GradCAM of the negative class for SingleLogitResnet models.
        '''
        # ------------- Predictoin ----------
        p = pred(
                model = model,
                inputs = t(Image.open(img_path)).unsqueeze(0).to('cuda'),
        )
        # ------------- GradCAM -------------
        p_filter, n_filter = gradCAM(
                img     = t(Image.open(img_path)),
                model   = model,
                layer   = layer
        )
        # ------------- Final Figure -------------
        # Figure
        pil_image = Image.open(img_path)
        original_width, original_height = pil_image.size
        resize = v2.Resize(size=(original_height, original_width),  max_size=None, antialias='warn')
        fig, axs = plt.subplots(2, 2)
        fig.set_facecolor("w")
        fig.set_figheight(fig.get_figheight()*1.5)
        fig.set_figwidth(fig.get_figwidth()*2)

        # Original Image
        axs[0,0].imshow(pil_image, alpha=1.0)
        axs[0,0].axis('off')
        axs[0,0].set_title("Imagen Original")

        # GradCAM Result 
        axs[0,1].imshow(v2.functional.to_pil_image(resize(p_filter)), alpha=1.0)
        axs[0,1].axis('off')
        axs[0,1].set_title("$G_{p=1}$")

        # p==1 
        axs[1,0].imshow(pil_image, alpha=1.0)
        axs[1,0].imshow(v2.functional.to_pil_image(resize(p_filter)), alpha=0.5)
        axs[1,0].axis('off')
        axs[1,0].set_title("$Imagen + G_{p=1}$")

        # p==0
        axs[1,1].imshow(pil_image, alpha=1.0)
        axs[1,1].imshow(v2.functional.to_pil_image(resize(n_filter)), alpha=0.5)
        axs[1,1].axis('off')
        axs[1,1].set_title("$Imagen + G_{p=0}$")

        # Figure
        plt.suptitle(f"Predicted label. p= {p}.")
        plt.show()

        return p_filter, n_filter