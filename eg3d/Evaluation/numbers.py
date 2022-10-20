from scipy.stats.stats import pearsonr 
import pandas as pd
import os
import numpy as np
import json

def normalize_ages(age, rmin = 5, rmax = 80, tmin = -1, tmax = 1):
    z = ((age - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
    return z

def save_correlation(scatter_data, file_name='numbers.csv', network_folder=None):
    root = os.path.join('Evaluation', 'Runs', scatter_data)
    scatter_path = os.path.join(root, 'age_scatter.csv')
    ages_df = pd.read_csv(scatter_path)
    age_true = np.array(ages_df['age_true'])
    age_hat = np.array(ages_df['age_hat'])
    columns = ['value']
    rows = ['correlation', 'corr_p_val', 'mae', 'num_samples', 'std', 'cs']
    corr = pearsonr(age_true, age_hat)
    
    age_min = 0
    age_max=75
    if network_folder:
        training_option_path = os.path.join(network_folder, "training_options.json")
        f = open(training_option_path)
        training_option = json.load(f)
        age_loss_fn = training_option['age_loss_fn']
        age_min = training_option['age_min']
        age_max = training_option['age_max']
    a = normalize_ages(age_true, rmin=-1, rmax=1, tmin=age_min, tmax=age_max)
    b = normalize_ages(age_hat, rmin=-1, rmax=1, tmin=age_min, tmax=age_max)
    mae = np.mean(np.abs(a - b))
    std = np.std(np.abs(a-b))
    n_samples = len(age_true)
    cs = np.sum(np.abs(a-b) > 5) / n_samples
    data = [[corr[0]], [corr[1]], [n_samples], [mae], [std], [cs]]

    df = pd.DataFrame(data, columns=columns)
    df.index=rows
    df.to_csv(os.path.join(root, 'numbers_eval.csv'))