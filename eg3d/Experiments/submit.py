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
parser.add_argument("--resume", help="",default = None)
parser.add_argument("--devices", help="",default = '0,1')
parser.add_argument("--data", help="",default = 'datasets/FFHQ_128')
parser.add_argument("--age_scale", help="",default = '1')

args = parser.parse_args()

args.logs_dir = Path(args.logs_dir)

if args.queue_name not in ['gpua100', 'gpuv100']:
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

