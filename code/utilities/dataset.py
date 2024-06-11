import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from torchvision import transforms
from pdf2image import convert_from_path
from sklearn.model_selection import StratifiedKFold
import numpy as np

class RAMDataset(Dataset):
  '''
  Pytorch Dataset created from ...
  - Pandas.DataFrame containing the metadata of the problem.
  The metadata DataFrame must include one row per sample/image y at least two columns:
  one for the image path and one for the image label.
  - Two files ".pt" containing images (resized to 224x224) and labels as tensors. 
  
  All images are pre-processed (transformed) and loaded to memory (RAM) to increase computing velocity.
  
  Image formats accepted: JPG, JPGE, PNG and PDF. Consult features.pilinfo(supported_formats=True).
  '''
  def __init__(
      self, 
      metadata_df = None, 
      path_col = 'path', 
      label_col = 'label',
      transforms = None,
      verbose=True,
      tensored_imgs_labels=None,
      expert_da = None,
      ):
    '''
    Dataset constructor.
    '''
    if tensored_imgs_labels != None:
      self.imgs = tensored_imgs_labels[0]
      self.labels = tensored_imgs_labels[1]
      self.transforms = transforms.Compose(
          [
            # PIL.Image (int8 in [0, 255]) --> Pytorch.Tensor (float32 in [0.0, 1.0]).
            transforms.ToTensor(),
            # Changes the dimensions of the image.
            transforms.Resize(
                size = [224,224],
                interpolation = transforms.InterpolationMode.BILINEAR,
                max_size = None,
                antialias=False #'warn'
                ),
        ]
      )
    else:
      # Labels.
      self.labels = torch.tensor(metadata_df[label_col].tolist(), requires_grad=False)
      
      # Images.
      # Images paths.
      imgs_paths = metadata_df[path_col].tolist()
      # Image transformations (before loading it to memory).
      self.transforms = transforms
      if self.transforms == None:
          self.transforms = transforms.Compose(
            [
              # PIL.Image (int8 in [0, 255]) --> Pytorch.Tensor (float32 in [0.0, 1.0]).
              transforms.ToTensor(),
              # Changes the dimensions of the image.
              transforms.Resize(
                  size = [224,224],
                  interpolation = transforms.InterpolationMode.BILINEAR,
                  max_size = None,
                  antialias =False #'warn'
                  ),
          ]
          )
      # Transform and load every image from its path (p).
      self.imgs = []
      aux = imgs_paths
      if verbose:
        aux = tqdm(iterable=imgs_paths, leave=True, desc='Loading/Transforming Images')
      for p in aux:
        if p[-3:] in ['pdf']:
          pil_image = convert_from_path(p)[0]
        else:
          pil_image = Image.open(p)
        self.imgs.append(self.transforms(pil_image))
      self.imgs = torch.stack([t for t in self.imgs])

  def __len__(self):
    '''
    Returns the length of the dataset.
    '''
    return len(self.labels)

  def __getitem__(self, idx):
    '''
    Given an index, returns the corresponding and image, along with its label.
    '''
    img = self.imgs[idx]
    label = self.labels[idx].item()
    
    return img, label
  
  def cross_val_idxs(self, skf_seed, n_splits=5):
    '''
    Returns two lists of n_splits elements.
    The i-th element of the first one contains the train indices for the i-th split.
    The i-th element of the second one contains the validation indices for the i-th split.
    '''
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=skf_seed)
    train_idxs = [train_idxs for train_idxs, _ in kf.split(X=np.zeros(len(self)), y=self.labels)]
    val_idxs = [val_idxs for _, val_idxs in kf.split(X=np.zeros(len(self)), y=self.labels)]
    return train_idxs, val_idxs

class DiskDataset(Dataset):
  '''
  Pytorch Dataset created from Pandas.DataFrame containing the metadata of the problem.
  The metadata DataFrame must include one row per sample/image y at least two columns:
  one for the image path and one for the image label.

  Images are transformed and loaded into memory when needed.

  Image formats accepted: JPG, JPGE, PNG and PDF. Consult features.pilinfo(supported_formats=True).
  '''
  def __init__(
      self, 
      metadata_df, 
      path_col = 'path', 
      label_col = 'label', 
      transforms=None, 
      ):
    '''
    Dataset constructor.
    '''
    # Metadata.
    self.metadata_df = metadata_df
    self.path_col = path_col
    self.label_col = label_col
    # Labels.
    self.labels = metadata_df[label_col].tolist()
    # Images paths.
    self.imgs_paths = metadata_df[path_col].tolist()
    # Image transformations.
    self.transform = transforms
    if transforms == None:
      self.transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
      )

  def __len__(self):
    '''
    Returns the length of the dataset.
    '''
    return len(self.labels)

  def __getitem__(self, idx):
    '''
    Given an index, returns the corresponding and image, along with its label.
    '''
    # If the image is a PDF ...
    if self.imgs_paths[idx][-3:] in ['pdf']:
      pil_image = convert_from_path(self.imgs_paths[idx])[0]
    else:
      pil_image = Image.open(self.imgs_paths[idx])
    return self.transform(pil_image), self.labels[idx]