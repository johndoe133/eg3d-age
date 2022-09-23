## TO HOLD THE ENTIRE PTI PIPELINE

from edit_age import edit_age
from pti_optimization import run
from preprocess_images import pre_process_images
from visualize_pti import visualize
import click
import numpy as np
from configs import paths_config, hyperparameters, global_config

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
@click.option('--w_iterations', help="How many iterations the program does to find w inversion", required=False, default=500, type=int)
@click.option('--pti_iterations', help="PTI inversion iterations", required=False, default=350, type=int)
@click.option('--run_pti_inversion', help="Whether to run the inversion", required=False, default=True, type=bool)
def pti_pipeline(
    age: int,
    image_name: str,
    preprocess: bool,
    model_path: str,
    w_iterations: int,
    pti_iterations: int,
    run_pti_inversion: bool,
):
    hyperparameters.first_inv_steps = w_iterations
    hyperparameters.max_pti_steps = pti_iterations

    age = normalize(age)

    if preprocess:
        pre_process_images()

    c = run(model_path, image_name, run_pti_inversion)

    visualize(image_name, c)
    print(f"See output in folder eg3d/PTI/output/{image_name}")

    edit_age(image_name, model_path, c)

if __name__ == "__main__":
    pti_pipeline()