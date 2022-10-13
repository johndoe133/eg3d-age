template_text = """
#!/bin/sh
### General options
### export CUDA_VISIBLE_DEVICES={devices}

### -- specify queue --
#BSUB -q {queue_name}

### -- set the job Name --
#BSUB -J {job_name}

### -- ask for number of cores (default: 1) --
#BSUB -n 8

### -- Select the resources: {gpu_num} gpus -- 
#BSUB -gpu "num={gpu_num}:mode=shared:j_exclusive=yes"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# Request GPU resources
#BSUB -R "rusage[mem={gpu_mem}GB]"
#BSUB -R "select[gpu{gpu_mem}gb]"

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
### -- send notification at start --
###BSUB -B
### -- send notification at completion--
###BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --

#BSUB -o {logs_dir}/run_%J.out
#BSUB -e {logs_dir}/run_%J.err

# -- end of LSF options --

# Activate venv
source activate eg3d

# Load the cuda module
module load gcc/9.2.0
module load cuda/11.1

python train.py --outdir=./training-runs --cfg=ffhq --data={data} --gpus={gpu_num} --batch={batch_size} --gamma={gamma} --gen_pose_cond=True --age_scale={age_scale}  --age_loss_fn={age_loss_fn} --id_scale={id_scale} --snap={snap} --batch_division={batch_division} --freeze={freeze} --age_version={age_version} --age_min={age_min} --age_max={age_max} --neural_rendering_resolution_initial={neural_rendering_resolution_initial} --categories={categories} --id_model={id_model}"""