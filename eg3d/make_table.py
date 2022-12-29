import pandas as pd
import numpy as np
import os
import json 

def get_model(id, age, age_loss_fn):
    suffix = "" if age_loss_fn == "MSE" else "/CAT"
    if id != 0:
        if age != 0:
            return "C" + suffix
        else:
            return "B" + suffix
    return "A" + suffix

results = []

training_folder = r'./training-runs'
fid_name = 'metric-fid50k_full.jsonl'
stats_name = 'stats.jsonl'
training_option_name = 'training_options.json'
evaluation_folder = r'./Evaluation/Runs'

for training_run in os.listdir(training_folder):
    cur_dir = os.path.join(training_folder, training_run)
    stats_path = os.path.join(cur_dir, stats_name)
    fid_path = os.path.join(cur_dir, fid_name)
    stats = pd.read_json(stats_path, lines=True)
    fid = pd.read_json(fid_path, lines=True)
    fid_final = round(fid.results.iloc[-1]['fid50k_full'],3)
    folders = list(filter(os.path.isdir, [os.path.join(cur_dir, x) for x in os.listdir(cur_dir)]))
    folder = folders[-1]

    training_option_path = os.path.join(folder, training_option_name)
    f = open(training_option_path)
    training_option = json.load(f)
    id_scale = training_option['id_scale']
    age_scale = training_option['age_scale']
    snap = training_option['network_snapshot_ticks']
    kimg_per_tick = training_option['kimg_per_tick']
    network_snapshot_ticks = training_option['network_snapshot_ticks']
    initial_age_training = training_option['loss_kwargs']['initial_age_training']
    age_loss_fn = training_option['age_loss_fn']
    f.close()
    img_pr_tick=4000
    train_kimgs = int(((len(fid)-1)*img_pr_tick*snap)/1000)
    model_name = get_model(id_scale, age_scale, age_loss_fn)

    for run in os.listdir(evaluation_folder):
        if training_run in run:
            try:
                trunc = float(run.split('-')[-1])
                eval_run_dir = os.path.join(evaluation_folder, run)
                
                df = pd.read_csv(os.path.join(eval_run_dir, "numbers_eval.csv"), index_col=0)
                id_df = pd.read_csv(os.path.join(eval_run_dir, "id_evaluation.csv"), index_col=0)
                id_score = round(id_df.cosine_sim_arcface.mean(), 4)
                fpage_mae = round(df.FPAge.MAE,3)
                results.append([
                    model_name, age_scale, id_scale, initial_age_training, train_kimgs, trunc, id_score, fid_final, fpage_mae
                ])
            except:
                print(f"Folder {run} evaluation incomplete.")
columns = [
    'Model', '$\\alpha_{age}$', '$\\alpha_{ID}$', '$\text{kimg}_{age}$','kimgs', '$\psi$', '$\text{ID}_\text{score}$', 'FID', 'MAE'
]
df = pd.DataFrame(results, columns=columns)
print(df.sort_values('MAE').to_latex(index=False,escape=False))

