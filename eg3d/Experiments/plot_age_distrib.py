import click
import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np

@click.command()
@click.option('--data', 'data', help='Dataset directory', required=True, metavar='DIR')
def main(data):
    f = open('/zhome/d1/9/127646/Documents/eg3d-age/eg3d/datasets/' + data)
    d = json.load(f)
    ages = [c[1][25] for c in d['labels']]
    name = data.split("/")[0]
    plt.hist(ages, bins=20)
    plt.title(f'min age {np.min(ages)}, max age {np.max(ages)}, dataset {name}')
    plt.savefig(f'histogram-{name}.png')
    print(f'saved figure as histogram-{name}.png')


if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter