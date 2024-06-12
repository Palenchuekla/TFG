import os
import pandas as pd
import numpy as np
from IPython import display
from sklearn.metrics import confusion_matrix

def sensitivity_score(y_true, y_pred):
    '''
    Sensitivity for a binary classfier given predicted an real labels.
    Sensitivity reflects how well a model performs for the positive class (label = 1).
    Parameters
    ----------
    - y_true
        Real labels.
    - y_pred
        Predicted labels.
    Returns
    ----------
    - metric
        Sensitivity.
    '''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metric = 0
    if(tp != 0 or fn != 0):
        metric = tp*1.0 / (tp + fn)
    return metric

def specifity_score(y_true, y_pred):
    '''
    Specifity for a binary classfier given predicted an real labels.
    Specifity reflects how well a model performs for the negative class (label = 0).
    Parameters
    ----------
    - y_true
        Real labels.
    - y_pred
        Predicted labels.
    Returns
    ----------
    - metric
        Specifity.
    '''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metric = 0
    if(tn != 0 or fp != 0):
        metric = tn*1.0 / (tn + fp)
    return metric

def safeMkdir(dir_path):
  '''
  Creates a new directory (recursively) only if it doesn't already exists.
  Firewall for not erasing costly experimental results.
  Parameters
  ----------
  - dir_path
    New folder path.
  '''
  if not os.path.exists(dir_path):
      os.makedirs(dir_path)
  else:
      raise ValueError(f'Folder {dir_path} already exists.')
  
def M(alpha, sen, spe):
    '''
    Return the weighted mean of the sensibility and specifity of a model.
    Parameters
    ----------
    - sen
        Sensitivity of the model.
    - spe
        Specificty of the model.
    - alpha
        Weight for combining them
    Returns
    ----------
    - M
        alpha * sen + (1 - alpha) * spe.
    '''
    return (alpha*sen + (1-alpha)*spe)

def print_submodules_names(module, indent=0):
    '''
    Pretty-prints the layered structure of a PyTorch model.
    Usefull to see how layers are grouped and how can access them.
    '''
    for name, module in module.named_children():
        name = f"-> {name}"
        for i in range(indent):
            name = '\t' + name 
        print(f"{name}")
        print_submodules_names(module, indent+1)

def write_results(dir_path, alpha):
    '''
    Returns results of the cross validation and stores them in the specified folder.
    Parameters
    ----------
    - dir_path
        Folder where results will be stored.
    - alpha
        Coefficient for the weighted sum of sensibility and specifity.
    Returns
    ----------
    - pd_best_model_metrics
        Pandas dataframe containing best models metrics per iteration.
    - pd_means_stds
        Pandas dataframe containing mean/std metrics of best models.
    '''
    # Mean and STD for every partition
    best_model_metrics = [] # i-th element, metrics for the best model found on the i-th partition
    col_names = ['train_loss','train_acc','train_sen','train_spe', 'train_cm','val_loss','val_acc','val_sen','val_spe', 'val_cm']
    mean_std_col_names = ['train_loss','train_acc','train_sen','train_spe','val_loss','val_acc','val_sen','val_spe']
    for k in range(5):
        aux = pd.read_csv(f'{dir_path}/partition_{k}/results.csv')
        M_ = M(alpha, aux['val_sen'], aux['val_spe'])
        epoch_best = np.argmax(M_) # Epoch=Row of best model
        metrics = aux.iloc[epoch_best][col_names].to_list() # Metrics of best model
        best_model_metrics.append([epoch_best]+metrics)
    # Save results
    pd_best_model_metrics = pd.DataFrame(best_model_metrics, columns=['epoch']+col_names)
    pd_best_model_metrics.to_csv(f'{dir_path}/metrics_best_model_per_partition_{alpha}.csv', index=False)
    # Mean and stds
    means = np.array(pd_best_model_metrics[mean_std_col_names]).mean(axis=0)
    stds = np.array(pd_best_model_metrics[mean_std_col_names]).std(axis=0)
    row_indexes = ['Mean', 'std']
    pd_means_stds = pd.DataFrame([means,stds], columns=mean_std_col_names, index=row_indexes)
    pd_means_stds.to_csv(f'{dir_path}/metrics_mean_std_best_model_per_partition_{alpha}.csv', index=False)
    
    return pd_best_model_metrics, pd_means_stds

def print_report(exp_results_dir_path, alpha_m):
    alpha_m = [0.4, 0.5, 0.6]
    winner = 0.5
    winner_Mcv = 2
    latex_rows = {
        'train':{}, 
        'val':{}
    }
    for cjto in ['train', 'val']:
        for a in alpha_m:
            latex_rows[cjto][a] = ""
    for a in alpha_m:
        print(f"\n------------------- alpha = {a} -------------------\n")
        pd_best_model, pd_mean_std_cv = write_results(
                                                        exp_results_dir_path=exp_results_dir_path, 
                                                        alpha=a
                                                    )
        print("--> Best Model per partition:")
        display.display(pd_best_model)
        print("--> CV Metrics:")
        display.display(pd_mean_std_cv)
        for cjto in ['train', 'val']:
            print(f'\t--> {cjto}: ')
            print("\t\t* Loss = {:.4f}".format(pd_mean_std_cv[f'{cjto}_loss'].tolist()[0]))
            print("\t\t* Acc = {:.4f}".format(pd_mean_std_cv[f'{cjto}_acc'].tolist()[0]))
            print("\t\t* SEN = {:.4f}".format(pd_mean_std_cv[f'{cjto}_sen'].tolist()[0]))
            print("\t\t* SPE = {:.4f}".format(pd_mean_std_cv[f'{cjto}_spe'].tolist()[0]))
            Mcv = abs(pd_mean_std_cv[f'val_sen'].tolist()[0]-pd_mean_std_cv[f'val_spe'].tolist()[0])          
            print("\t\t* Mcv = |SEN-SPE| = {:.4f}".format(Mcv))
            l_row = "{:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} ".format(
                pd_mean_std_cv[f'{cjto}_loss'].tolist()[0],
                pd_mean_std_cv[f'{cjto}_acc'].tolist()[0],
                pd_mean_std_cv[f'{cjto}_sen'].tolist()[0],
                pd_mean_std_cv[f'{cjto}_spe'].tolist()[0],
                Mcv,
                )
            if cjto == 'val':
                Osen = pd_mean_std_cv[f'train_sen'].tolist()[0]-pd_mean_std_cv[f'val_sen'].tolist()[0]
                l_row += "& {:.4f} ".format(Osen)
            l_row += " \\\\\n"
            print(f"\t\t* Latex table row format: {l_row}")
            latex_rows[cjto][a] = latex_rows[cjto][a] + l_row
            if Mcv < winner_Mcv:
                winner = a
                winner_Mcv = Mcv
        print("--> Overfitting: ")
        print("\t * Latex table row: {:.4f} & {:.4f} & {:.4f} & {:.4f}".format(
            pd_mean_std_cv[f'val_loss'].tolist()[0]-pd_mean_std_cv[f'train_loss'].tolist()[0],
            pd_mean_std_cv[f'train_acc'].tolist()[0]-pd_mean_std_cv[f'val_acc'].tolist()[0],
            pd_mean_std_cv[f'train_sen'].tolist()[0]-pd_mean_std_cv[f'val_sen'].tolist()[0],
            pd_mean_std_cv[f'train_spe'].tolist()[0]-pd_mean_std_cv[f'val_spe'].tolist()[0],
        ))
    print(f"\n\n\n******************* Winner = {winner} *******************")
    return latex_rows