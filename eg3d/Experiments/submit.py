import subprocess
from pathlib import Path
import argparse

from config import template_text

parser = argparse.ArgumentParser(description='A script for submitting individual bsub scripts')

# default values
parser.add_argument("--queue_name", help="Specify the gpu queue. gpua100, gpuv100 (4gpu), xx",default = 'gpua100')
parser.add_argument("--job_name", help="Give job name",default = 'eg3d')
parser.add_argument("--gpu_num", help="",default = 2, type=int)
parser.add_argument("--gpu_mem", help="",default = 40, type=int)
parser.add_argument("--logs_dir", help="",default = 'out')
parser.add_argument("--batch_size", help="",default = 8, type=int)
parser.add_argument("--gamma", help="",default = 5)
parser.add_argument("--resume", help="",default = './networks/ffhqrebalanced512-128.pkl')
parser.add_argument("--devices", help="",default = '0,1')
parser.add_argument("--data", help="",default = '/work3/morbj/FFHQ')
parser.add_argument("--age_scale", help="",default = '8')
parser.add_argument("--age_loss_fn", help="", default="MSE")
parser.add_argument("--id_scale", help="", default=10)
parser.add_argument("--snap", help="How many ticks between saving a snapshot", default=50)
parser.add_argument("--freeze", help="", default=False)
parser.add_argument("--age_version", help="", default='v2')
parser.add_argument("--age_min", help="",default=0)
parser.add_argument("--age_max", help="",default=75)
parser.add_argument("--neural_rendering_resolution_initial", help="", default=128)
parser.add_argument("--id_model", help="", default="MagFace")
parser.add_argument("--alternate_losses", help="", default=False)
parser.add_argument("--alternate_after", help="", default=100000)
parser.add_argument("--initial_age_training", help="", default=0)
parser.add_argument('--crop_before_estimate_ages', help="", default=False)
parser.add_argument('--training_time', help='', default='24')
parser.add_argument('--description', help="", default="")


args = parser.parse_args()

args.logs_dir = Path(args.logs_dir)

if args.queue_name not in ['gpua100', 'gpuv100', 'gpuvoltasxm2']:
    raise Exception('Invalid queue name! Queue must be gpua100, gpuv100, or xx')
elif args.queue_name == 'gpuv100':
    if args.gpu_mem > 32:
        raise Exception('gpu_mem cannot exceed 32 for the gpuv100 queue')
    if args.gpu_num > 4:
        raise Exception('gpu_num cannot exceed 4 for gpuv100 queue')
else:
    if args.gpu_mem > 40:
        raise Exception(f'gpu_mem cannot exceed 40gb for the {args.queue_name} queue')
    if args.gpu_num > 2:
        raise Exception(f'gpu_num cannot exceed 2 for {args.queue_name} queue')

if not args.devices:
    args.devices = ''
    for i in range(args.gpu_num):
        args.devices += f'{i},'
    args.devics = args.devices[:-1]

if len(args.devices.split(',')) != args.gpu_num:
    raise Exception('Number of cuda visible devices must be equal to number of gpus specified')

if args.data == 'datasets/FFHQ_128' and args.resume:
    args.data = 'datasets/FFHQ_512'

submit_script = template_text.format(
                    **{
                        "queue_name": args.queue_name, 
                        "job_name": args.job_name, 
                        "gpu_num": args.gpu_num, 
                        "gpu_mem": args.gpu_mem, 
                        "logs_dir": args.logs_dir, 
                        "batch_size": args.batch_size, 
                        "gamma": args.gamma, 
                        "devices": args.devices,
                        "data": args.data,
                        "age_scale": args.age_scale,
                        "age_loss_fn": args.age_loss_fn,
                        "id_scale": args.id_scale,
                        "snap": args.snap,
                        "freeze": args.freeze,
                        "age_version": args.age_version,
                        "age_min": args.age_min,
                        "age_max": args.age_max,
                        "neural_rendering_resolution_initial" : args.neural_rendering_resolution_initial,
                        "id_model": args.id_model,
                        "alternate_losses": args.alternate_losses,
                        "alternate_after": args.alternate_after,
                        "initial_age_training": args.initial_age_training,
                        "crop_before_estimate_ages": args.crop_before_estimate_ages,
                        'training_time': args.training_time,
                        "description": args.description,

                    }
                )

if args.resume:
    submit_script += f" --resume={args.resume}"
    

submit_file = args.logs_dir / "submit.sh"

with open(submit_file, "w") as f:
    f.write(submit_script)

print(f"Submitting job: {submit_file} with job name {args.job_name}")

normal = subprocess.run(
                    f"bsub < {submit_file}",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    shell=True,
                )

