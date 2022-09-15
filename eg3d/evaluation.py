import click 
import torch
import dnnlib
import legacy


# Evaluation pipeline

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)

def evaluate(
    network_pkl: str,
):
    print(f'Loading networks from {network_pkl}...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)