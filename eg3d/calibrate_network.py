import click 
import os
import pandas as pd
from training.training_loop import normalize, denormalize
import numpy as np
@click.command()
@click.option('--network_pkl_path', '--n', help='Path to network pickle', required=True)
@click.option('--t', help='truncation_psi', default=1.0, required=False)
@click.option('--age_min', help='', default=0, required=False)
@click.option('--age_max', help='', default=75, required=False)
def calibrate(
    network_pkl_path: str,
    t: float,
    age_min: int,
    age_max: int
):
    network_pkl = network_pkl_path.split('/')[-1]
    run_folder = network_pkl_path.split('/')[-3]
    inner_folder = network_pkl_path.split('/')[-2]
    save_name = f"{run_folder}-{network_pkl.split('.')[0][8:]}-trunc-{t}"
    
    evaluation_path = os.path.join("Evaluation", "Runs", save_name)
    if os.path.isdir(evaluation_path):
        df = pd.read_csv(os.path.join(evaluation_path, "age_scatter.csv"))
        yu4u_hat = df.age_hat.to_numpy()
        age_true = df.age_true.to_numpy()
        yu4u_hat = denormalize(yu4u_hat, rmin=age_min, rmax=age_max)
        age_true = denormalize(age_true, rmin=age_min, rmax=age_max)
        a, b = np.polyfit(age_true, yu4u_hat,1)
        print(f"Best fit line for {len(age_true)} synthetic faces:\n y_hat(target) = {round(a,3)}*target + {round(b,3)}")
        save = os.path.join("training-runs", run_folder, inner_folder, f"calibrate-{t}")
        np.save(save, [a,b])

if __name__ == "__main__":
    calibrate()
