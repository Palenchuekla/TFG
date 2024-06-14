import time
import torch
import torch.nn
import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm #https://pypi.org/project/tqdm/#manual
from IPython import display
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from utilities.common import specifity_score
from utilities.common import sensitivity_score
import matplotlib.pyplot as plt
import operator

def train(
    model : torch.nn.Module,
    optimizer : torch.optim,
    loss : torch.nn.Module,
    n_epochs : int,
    dataloaders : dict,
    mixup_t = None,
    data_augmentation_t : dict = None,
    metrics : dict ={
        'acc':accuracy_score,
        'sen':sensitivity_score,
        'spe':specifity_score,
        'cm':confusion_matrix
    },
    es_patience : float = np.inf,
    es_tolerance : int = 0,
    es_metric = 'loss',
    best_model_save_load : bool = True,
    scheduler: torch.optim.lr_scheduler = None,
    results_dir_path: str = '../results/modelo_X',
    verbosity : int = True,
):
  '''
  Function to train and validate a PyTorch model.

  Parameters
  ----------
  - model:
    PyTorch model. Must output the logit of the positive class.
  - optimizer:
    PyTorch optimizer.
  - loss:
    PyTroch loss function.
  - n_epochs:
    Maximum number of epochs.
  - dataloaders:
    Dictionary with one key ('train') or two keys ('train' and 'val'). Each key must have a torch.utils.data.DataLoader associated. These are the dataloaders used for training and validation.
  - mixup_t:
    Custom PyTorch transform object (check utilities/mixup.py). If defined, MixUp is applied, either before or after the rest of data augmentation techniques.
  - data_augmentation_t:
    Dictionary with two keys, one for every class: '0' and '1'. Every key as assigned a Pytorch transform object that it is applied to all images whose label matches the key.
  - metrics:
    Dictionary to specificy the metrics to be evalued (appart from the specified loss) during evaluation and training. The key is the identifier of the metric. The associated value is the name of the funtion to call to compute the metric.
  - es_patience:
    Early stopping patience.
  - es_tolerance:
    Early stopping tolerance.
  - es_metric:
    Early stopping evalued metric. Must be a metrics key.
  - best_model_save_load
    If "True", the weights of the best found model is loaded at the end of the training. Same metric as the Early Stopping.
  - scheduler:
    PyTorch scheduler to set the policy of actualization of the learning rate (adaptative, decaying, different for some parameter groups ...).
  - results_dir_path:
    Directory path to store resutls and metadata. Error if already exists to avoid deletaing precious data.
  - verbosity:
    If executing outside a Jupyter Notebook, set verbosity to 0.
    if == 1 : Early Stopping and best model hyperparameters are printed at the begining. Progress bars and table with results are printed and actalized during training (Jupyter Notebooks tested, don't really now how it will behave in terminal). Training and validation curves are printed at the end. 
    if >= 2 : Extra information about best model printed.

  Returns
  -------
  - pd.DataFrame
      pd.DataFrame containing the train and evaluation results (one row per epoch).
  '''

  # Results Folder 
  if not os.path.exists(results_dir_path):
    os.makedirs(results_dir_path)
  else:
    raise ValueError(f'Folder {results_dir_path} already exists.')
  
  # Control variables to store results: loss, performance metrics and time per every epoch.
  metrics_training_values = []
  aux = metrics
  metrics = {'loss':loss}
  for k in aux.keys(): metrics[k] = aux[k]
  metrics_df_col_names =[ cjto + '_' +m for cjto in dataloaders.keys() for m in metrics.keys()] # = [train_loss, train_acc, ..., val_loss, val_acc ...]
  metrics_df_col_names.append('epoch_time') # = [train_loss, train_acc, ..., val_loss, val_acc ..., epoch_time]
  metrics_df = pd.DataFrame([], columns=metrics_df_col_names)

  if len(dataloaders) != 1:
    # Control varibales for early-stopping
    if es_metric not in metrics.keys():
      raise ValueError(f'Invalid metric for early stopping (\"{es_metric}\"). Must be in {metrics.keys()}.')
    es_counter = 0 # Counter for the number of epochs without a significant improrvent on 'es_metric' for the validation set
    es_best_epoch = -1 # Epoch when the model reached the best 'es_metric' for the validation set
    if es_metric == 'loss':
      es_best_value = np.inf # Initial best value for 'loss'
      operator_best_value = np.less
      operator_tolerance = operator.sub
    else:
      es_best_value = -1 # Initial best value for a metric like accuracy, recall or senicison
      operator_best_value = np.greater
      operator_tolerance  = operator.add
    if verbosity > 0:
      print(f'EARLY STOPPING. Metric = \'val_{es_metric}\'. Patience = {es_patience}. Tolerance = {es_tolerance}.')

    # Control varibales for storing the best model
    bm_best_epoch = -1 # Epoch when the model reached the best 'bm_metric' for the validation set
    bm_metric = es_metric # Not too much sense any other way
    bm_best_value = es_best_value # Not too much sense any other way
    if best_model_save_load:
      if verbosity > 0:
        print(f'STORING BEST MODEL FOR \'val_{bm_metric}\' AT {results_dir_path}/best_model_params.pt .')
      torch.save(model.state_dict(), f"{results_dir_path}/best_model_params.pt")



  # Training Loop.
  global_pbar=range(n_epochs)
  if verbosity > 0:
    global_pbar = tqdm(iterable=range(n_epochs), leave=True, unit="epoch", desc ='Training Progress: ') # Progress bar for the whole process
  for epoch in global_pbar:
    # ·) Training Phase. Where the "magic" happens. Make the model "learn". Optimize model parameters.

    # Modify the behavior of certain model layers ... --> https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    # --> Activate dropout layers 
    # --> Change the stadistics used in normalization layers. This can fool you into thinking that there's no reproducibility. --> https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    model.train()
    # Epoch start time
    t_start = time.time()
    # For every "mini-batch" ("batch_size" images and labels) ...
    epoch_train_pbar = dataloaders['train']
    if verbosity > 0:
      epoch_train_pbar = tqdm(iterable=dataloaders['train'], leave=False, unit="mini_batch", desc ='Epoch {} Progress: '.format(epoch)) # Progress bar for the epoch's training phase
    for images, labels in epoch_train_pbar:
        # Apply data augmentation
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
        # Load "mini-batch" to GPU.
        images = images.to('cuda')
        labels = labels.to('cuda')
        # Restart gradients. If not, they accumulate (addition).
        optimizer.zero_grad()
        # Forward pass. The model processes the "mini_batch" images. Preditcs their labels and constructs the computational graph ("autograd").
        logits = model(images)
        # Computing loss function for the whole "mini-batch".
        minibatch_loss = loss(logits, labels)
        # Backward pass. Computing (model parameters) gradients applying the chain rule thanks to the computational graph and the loss function value.
        minibatch_loss.backward()
        # Optimizing/Actualizing model parameters
        optimizer.step()
    # Increase epoch in scheduler
    if scheduler != None:
          scheduler.step()


    # ·) Evaluation Phase   A) Compute loss function and performance metrics over training and validation set.
    #                       B) Early Stopping
    #                       C) Store best model parameters

    
    # A) Compute loss function and performance metrics over training and validation set.
    # Modify the behavior of certain model layers (generally, dropout and normalization).
    model.eval()
    # Computational graph is not computed (required_grad = false). No need to compute gradients through the chain rule.
    with torch.no_grad():
      # Container for this epoch metrics and loss values.
      metrics_epoch_values={}
      # For the training and validation set ...
      for cjto in dataloaders.keys():
        metrics_epoch_values[cjto] = {}
        # Still using "mini-batches" to not collapse GPU memory ...
        for i, (images, labels) in enumerate(dataloaders[cjto]):
            # Apply data augmentation (if not mix up)
            if cjto == 'train':
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
            # Load "mini-batch" to GPU
            images = images.to('cuda')
            labels = labels.to('cuda')
            # Forward pass. No computatinoal graph is built.
            logits = model(images)
            # Store labels and preditions for the whole set.
            if i == 0:
              all_labels = labels
              all_logits = logits
            else:
              all_labels = torch.concat([all_labels, labels],dim=0)
              all_logits = torch.concat([all_logits, logits],dim=0)
        # Computing and storing loss function
        metrics_epoch_values[cjto]['loss'] = loss(all_logits, all_labels).item()
        # Computing and storing performance metrics
        for m in metrics.keys():
          if m != 'loss':
              if m == 'prec':
                metrics_epoch_values[cjto][m] = metrics[m](all_labels.to(int).tolist(), (torch.nn.Sigmoid().forward(all_logits).round().to(int)).tolist(), zero_division=0.0)
              else:
                metrics_epoch_values[cjto][m] = metrics[m](all_labels.to(int).tolist(), (torch.nn.Sigmoid().forward(all_logits).round().to(int)).tolist())
        # Computing and storing epoch execution time
        # Epoch end time
        t_end = time.time()
        metrics_epoch_values['time'] = t_end-t_start
    # Storing results
    metrics_training_values.append(
        [metrics_epoch_values[cjto][m] for cjto in dataloaders.keys() for m in metrics.keys()] + [metrics_epoch_values['time']]
        )
    metrics_df = pd.DataFrame(metrics_training_values[:], columns=metrics_df_col_names)
    #metrics_df.to_csv(f'{results_dir_path}/results.csv')
    # Visulaizing results. Table.
    if verbosity > 0:
      if epoch == 0 :
        id = display.display(metrics_df, display_id=True)
      else:
        id.update(metrics_df)

    # B) Storing Best Model
    # If the evaluated metric is better than the actual best ...
    if len(dataloaders) != 1:
      if best_model_save_load and operator_best_value(metrics_epoch_values['val'][bm_metric], bm_best_value):
        # Actualize to new best
        bm_best_value = metrics_epoch_values['val'][bm_metric]
        bm_best_epoch = epoch
        torch.save(model.state_dict(), f"{results_dir_path}/best_model_params.pt")
        if verbosity > 1:
          print(f"NEW ABSOLUTE BEST FOR \'val_{bm_metric}\' FOUND AT EPOCH {bm_best_epoch}. STORING BEST'S PARAMS AT {results_dir_path}/best_model_params.pt.")

    # C) Early Stopping
    # If the evaluated metric is tolerance-better than the actual best ...
      if operator_best_value(metrics_epoch_values['val'][es_metric], operator_tolerance(es_best_value, es_tolerance)):
        # Re-start the count
        es_counter = 0
        # Store new best
        es_best_value = metrics_epoch_values['val'][es_metric]
        es_best_epoch = epoch
        #print(f"NEW SIGNIFICANT BEST (tolerance={es_tolerance}) FOR \'val_{es_metric}\' FOUND AT EPOCH {es_best_epoch}. RE-STARTING EARLY-STOPPING.")

      # If not ...
      else:
        # Count another epoch without imporvement
        es_counter = es_counter + 1
      # If for es_patience epochs, the evaluated metric has not improved ...
      if es_counter > es_patience:
        # Stop the training process
        if verbosity > 1:
          print(f"TRAINING STOPED BY EARLY-STOPPING AT EPOCH {epoch}. NO SIGNIFICANT IMPORVEMENT (tolerance={es_tolerance}) ON \'val_{es_metric}\' SINCE EPOCH {es_best_epoch}.")
        break

  # Loading best model
  if len(dataloaders) != 1:
    if best_model_save_load == True:
      model.load_state_dict(torch.load(f"{results_dir_path}/best_model_params.pt"))
      if verbosity > 1:
        print(f'LOADED BEST MODEL FOR \'val_{bm_metric}\' FOUND AT EPOCH {bm_best_epoch} FROM {results_dir_path}/best_model_params.pt .')
  
  # Plots.
  num_subplots = len(metrics.keys())
  if 'cm' in metrics:
    num_subplots = num_subplots - 1
  l_aux = [*metrics]
  if 'cm' in metrics:
    l_aux.remove('cm')

  fig, axs = plt.subplots(1, num_subplots)
  fig.set_facecolor("w")
  fig.set_figheight(fig.get_figheight()*1)
  fig.set_figwidth(fig.get_figwidth()*num_subplots)

  for i, m in enumerate(l_aux):
    for cjto in dataloaders.keys():
      axs[i].plot(metrics_df[cjto+'_'+m], label=f'{cjto}')
      if m != 'loss':
        axs[i].set_ylim(bottom=-0.05, top=1.05)
    axs[i].set_title(m)
    axs[i].legend()

  # Storing training curves
  plt.savefig(f'{results_dir_path}/training_curves.png')
  # Showing training curves
  if verbosity > 0:
    plt.show()

  # Storing table of results
  metrics_df.to_csv(f'{results_dir_path}/results.csv')

  return metrics_df