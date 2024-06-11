import os
import pandas as pd
import numpy as np
from IPython import display

def safeMkdir(dir_path):
  '''
  Creates a new directory (recursively) only if it doesn't already exists.
  Firewall for not erasing costly experimental results.
  '''
  if not os.path.exists(dir_path):
      os.makedirs(dir_path)
  else:
      raise ValueError(f'Folder {dir_path} already exists.')
  
def M(alpha, sen, spe):
    '''
    Return the weighted mean of the sensibility and specifity of a model.
    '''
    return (alpha*sen + (1-alpha)*spe)

def print_submodules_names(module, indent=0):
    for name, module in module.named_children():
        name = f"-> {name}"
    for i in range(indent):
        name = '\t' + name 
    print(f"{name}")
    print_submodules_names(module, indent+1)

def write_results(exp_results_dir_path, alpha):
    '''
    Returns results of the cross validation.
    - Pandas dataframe containing best models metrics per iteration.
    - Pandas dataframe containing mean/std metrics of best models.
    Parameters:
    - alpha_m: coeffcieint for computing the weighted sum of SEN and SPE.
    '''
    # Mean and STD for every partition
    best_model_metrics = [] # i-th element, metrics for the best model found on the i-th partition
    col_names = ['train_loss','train_acc','train_sen','train_spe', 'train_cm','val_loss','val_acc','val_sen','val_spe', 'val_cm']
    mean_std_col_names = ['train_loss','train_acc','train_sen','train_spe','val_loss','val_acc','val_sen','val_spe']
    for k in range(5):
        aux = pd.read_csv(f'{exp_results_dir_path}/partition_{k}/results.csv')
        M_ = M(alpha, aux['val_sen'], aux['val_spe'])
        epoch_best = np.argmax(M_) # Epoch=Row of best model
        metrics = aux.iloc[epoch_best][col_names].to_list() # Metrics of best model
        best_model_metrics.append([epoch_best]+metrics)
    # Save results
    pd_best_model_metrics = pd.DataFrame(best_model_metrics, columns=['epoch']+col_names)
    pd_best_model_metrics.to_csv(f'{exp_results_dir_path}/metrics_best_model_per_partition_{alpha}.csv', index=False)
    # Mean and stds
    means = np.array(pd_best_model_metrics[mean_std_col_names]).mean(axis=0)
    stds = np.array(pd_best_model_metrics[mean_std_col_names]).std(axis=0)
    row_indexes = ['Mean', 'std']
    pd_means_stds = pd.DataFrame([means,stds], columns=mean_std_col_names, index=row_indexes)
    pd_means_stds.to_csv(f'{exp_results_dir_path}/metrics_mean_std_best_model_per_partition_{alpha}.csv', index=False)
    
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