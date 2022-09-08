## TO HOLD THE ENTIRE PTI PIPELINE

from pti_optimization import run
from preprocess_images import pre_process_images
from visualize_pti import visualize
import click
import numpy as np

def normalize(x, rmin = 5, rmax = 80, tmin = -1, tmax = 1):
    """Defined in eg3d.training.training_loop
    """
    z = ((x - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
    return np.round(z, 4)

@click.command()
@click.option('--age', help='Target age of the image (not normalized)', required=True, type=int)
@click.option('--image_name', help="", required=True)
@click.option('--preprocess', help="Whether to preprocess images", default=False)
@click.option('--model_path', help="Relative path to the model", required=True)
def pti_pipeline(
    age: int,
    image_name: str,
    preprocess: bool,
    model_path: str,
):
    age = normalize(age)

    if preprocess:
        pre_process_images()

    run(age, model_path)

    visualize(image_name)

    print(f"See output in folder eg3d/PTI/output/{image_name}")

if __name__ == "__main__":
    pti_pipeline()