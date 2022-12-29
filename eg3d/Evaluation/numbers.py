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
    age_hat_fpage = np.array(ages_df['age_hat2'])
    mag = np.array(ages_df['mag'])
    columns = ['yu4u', 'FPAge']

    rows = ['correlation', 'num_samples', 'MAE', 'std', 'CS_5', 'CS_10', 'CS_15']

    corr_yu4u = pearsonr(age_true, age_hat)
    corr_fpage = pearsonr(age_true, age_hat_fpage)

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
    error_yu4u = np.abs(age_true_unnormalized - age_hat_unnormalized)
    error_fpage = np.abs(age_true_unnormalized - age_hat_fpage)
    mae_yu4u = np.mean(error_yu4u)
    mae_fpage = np.mean(error_fpage)
    std_yu4u = np.std(error_yu4u)
    std_fpage = np.std(error_fpage)
    n_samples = len(age_true)

    cs5_yu4u = int(round((np.sum(error_yu4u > 5) / n_samples) * 100,0))
    cs10_yu4u = int(round((np.sum(error_yu4u > 10) / n_samples) * 100,0))
    cs15_yu4u = int(round((np.sum(error_yu4u > 15) / n_samples) * 100,0))
    cs5_fpage = int(round((np.sum(error_fpage > 5) / n_samples) * 100,0))
    cs10_fpage = int(round((np.sum(error_fpage > 10) / n_samples) * 100,0))
    cs15_fpage = int(round((np.sum(error_fpage > 15) / n_samples) * 100,0))

    error_mag_corr_yu4u = pearsonr(error_yu4u, mag)
    error_mag_corr_fpage = pearsonr(error_fpage, mag)

    data = [
            [corr_yu4u[0],corr_fpage[0]],[n_samples, n_samples], [mae_yu4u,mae_fpage],
            [std_yu4u,std_fpage],  [cs5_yu4u,cs5_fpage], [cs10_yu4u,cs10_fpage], 
            [cs15_yu4u, cs15_fpage]
            ]


    df = pd.DataFrame(data, columns=columns)
    df.index=rows
    df = df.round(3)
    df.to_csv(os.path.join(root, 'numbers_eval.csv'))
    print(df.to_latex())