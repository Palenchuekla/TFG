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