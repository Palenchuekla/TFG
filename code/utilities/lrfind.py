import torch
from tqdm.auto import tqdm
import pandas as pd

def initial_lr_find_Adam(
    model : torch.nn.Module,
    loss : torch.nn.Module,
    dataloader : torch.utils.data.DataLoader,
    mixup_t = None,
    data_augmentation_t = None,
    min_exp = -10,
    max_exp = 0,
    n_iters = 1000,
    save_path = None
):
  '''
  Custom interpretation of the LR-Find method proposed in https://arxiv.org/abs/1506.01186.
  It trains a model by fitting 1000 minii-batches/iterations. 
  After every batch is processed, the loss is saved and the learning rate increased exponentially.
  The slope of the stored loss, allows the expert to set a proper learning rate.
  The optimizer used is ADAM.
  Parameters
  ----------
  - model: 
      PyTorch model whose initial learning rate is trying to be computed.
  - loss:
      PyTorch loss object.
  - dataloader:
      Data to fit.
  - mixup_t:
    Custom PyTorch transform object (check utilities/mixup.py). If defined, MixUp is applied, either before or after the rest of data augmentation techniques.
  - data_augmentation_t:
    Dictionary with two keys, one for every class: '0' and '1'. Every key as assigned a Pytorch transform object that it is applied to all images whose label matches the key.
  - min_exp
    Initial learning rate to evaluate = 10^min_exp.
  - max_exp
    Final learning rate to evaluate = 10^max_exp.
  - n_iters
    Number of learning rates/batches to evaluate within the minimum and maximum defined.
  - save_path
    Path to store results (pnadas dataframe containing the loss per iteration/learning rate candidate).
  Returns
  ----------
  - lres
    Learning rate exponents evalued.
  - lrs
    Learning rates evalued.
  - losses
    Loss per processed batch / learning rate evalued.
  '''
  # lrs to evaluate (candidates)
  lres  = torch.linspace(min_exp, max_exp, n_iters)
  lrs   = 10**lres

  # Loss for every candidate
  losses = []
  for lr in tqdm(lrs, unit="lr_candidate", desc ='LR-Finder Progress: '):
    # Optimizer with new learning rate.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr.item())
    for images, labels in dataloader:
    # Data Augmentation
      # If no mix-up
      if mixup_t == None:
        if data_augmentation_t != None:
          images = torch.stack([*[data_augmentation_t[labels[i].item()](image) if data_augmentation_t[labels[i].item()].p > 0.0 else image for i,image in enumerate(images)]])
      # If mixup
      else:
        # Ff 1st mixup
        if mixup_t.first == True:
          images, labels = mixup_t(images, labels)
          labels = labels[:,1]
          if data_augmentation_t != None:
              images = torch.stack([*[data_augmentation_t[0](image) if data_augmentation_t[0].p > 0.0 else image for i,image in enumerate(images)]])
        # If 2nd mixup
        else:
          if data_augmentation_t != None:
            images = torch.stack([*[data_augmentation_t[labels[i].item()](image) if data_augmentation_t[labels[i].item()].p > 0.0 else image for i,image in enumerate(images)]])
          images, labels = mixup_t(images, labels)
          labels = labels[:,1]
      images = images.to('cuda')
      labels = labels.to('cuda')
      # Restart gradients. If not, they accumulate (addition).
      optimizer.zero_grad()
      # Forward pass. The model processes the "mini_batch" images. Preditcs their labels and constructs the computational graph ("autograd").
      logits = model(images)
      # Computing loss function for the whole "mini-batch".
      minibatch_loss = loss(logits, labels)
      # Store results
      losses.append(minibatch_loss.item())
      # Backward pass. Computing (model parameters) gradients applying the chain rule thanks to the computational graph and the loss function value.
      minibatch_loss.backward()
      # Optimizing/Actualizing model parameters
      optimizer.step()
      # Only one batch
      break

  if save_path != None:
    aux_df = pd.DataFrame(
        {'lres': lres,
        'lrs': lrs,
        'loss': losses
        }
    )
    aux_df.to_csv(save_path)

  return lres, lrs, losses