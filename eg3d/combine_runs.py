import os

def combine_runs(comb_dir, file_name):
    comb_dir = os.path.join('training-runs', comb_dir)
    runs = os.listdir(comb_dir)
    runs = [os.path.join(comb_dir, dir, file_name) for dir in os.listdir(comb_dir) if os.path.isdir(os.path.join(comb_dir, dir))]
    runs = sorted(runs)
    combined_json = []
    for i, run in enumerate(runs):
        with open(run) as f:
            if i == 0:
                combined_json += f.readlines()
            else:
                combined_json += f.readlines()[1:]
    with open(os.path.join(comb_dir, file_name), 'w') as f:
        [f.write(line) for line in combined_json]
    print(f'Wrote combined {file_name} file to', os.path.join(comb_dir, file_name))
