from scipy.stats.stats import pearsonr 
import pandas as pd
import os
import numpy as np
import json
from training.training_loop import denormalize, normalize

def save_correlation(scatter_data, file_name='numbers.csv', network_folder=None):
    root = os.path.join('Evaluation', 'Runs', scatter_data)
    scatter_path = os.path.join(root, 'age_scatter.csv')
    ages_df = pd.read_csv(scatter_path)
    age_true = np.array(ages_df['age_true'])
    age_hat = np.array(ages_df['age_hat'])
    mag = np.array(ages_df['mag'])
    columns = ['type', 'value']

    rows = ['correlation', 'corr_p_val', 'mae', 'num_samples', 'std', 'cs5', 'cs10', 'cs15', 'mag_corr', 'mag_p_val']

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
    age_true_unnormalized = normalize(age_true, rmin=-1, rmax=1, tmin=age_min, tmax=age_max)
    age_hat_unnormalized = normalize(age_hat, rmin=-1, rmax=1, tmin=age_min, tmax=age_max)
    error = np.abs(age_true_unnormalized - age_hat_unnormalized)
    mae = np.mean(error)
    std = np.std(error)
    n_samples = len(age_true)

    cs5 = np.sum(error > 5) / n_samples
    cs10 = np.sum(error > 10) / n_samples
    cs15 = np.sum(error > 15) / n_samples

    error_mag_corr = pearsonr(error, mag)

    data = [[corr[0]], [corr[1]], [mae],[n_samples],  [std],  [cs5], [cs10], [cs15], [error_mag_corr[0]], [error_mag_corr[1]]]


    df = pd.DataFrame(data, columns=columns)
    df.index=rows
    df.to_csv(os.path.join(root, 'numbers_eval.csv'))
