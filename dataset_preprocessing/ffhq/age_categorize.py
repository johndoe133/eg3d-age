import os
import click
import json
import numpy as np
from tqdm import tqdm

def normalize(x, rmin = 5, rmax = 80, tmin = -1, tmax = 1):
    z = ((x - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
    return round(z, 4)

def calc_age_category(network, dataset):
    return 0

def calc_age_category_from_json(dataset, categories=[5,10,20,30,40,50,60,80], normalized=True):
    f = open(dataset)
    data = json.load(f)['labels']
    data_category = dict({'labels': []})
    if normalized:
        categories = np.array([normalize(x) for x in categories])
    print(f'Calculating categorized age groups from {dataset}')
    for image in tqdm(data):
        age = (image[1][-1])

        # age_category gives the FIRST category for which age > category is true
        # for example, for age 25, it would give the following array:
        # [False, False, True, False, False, False, False, False]
        # which is then mapped to [0,0,1,0,0,0,0,0]
        # in order to have it one hot encoded
        age_category = list(np.logical_and(age < categories[1:], age > categories[:-1]))
        age_category += [False]
        age_category = list(map(float, age_category))

        data_category['labels'] += [[image[0], image[1][:-1] + age_category]]
    
    data_category['categories'] = list(categories)
    
    dataset_dir = os.path.split(dataset)[0]
    file_location = os.path.join(dataset_dir, "dataset_categories.json")
    print(f'saving file at {file_location}')
    out_file = open(file_location, "w")
    json.dump(data_category, out_file)

    print('Completed data categorization')

@click.command()
@click.option('--network', help='Network pickle filename or URL', metavar='PATH', required=False)
@click.option('dataset_json', '--dataset_json', help='dataset.json location to base age categories off of', metavar='PATH', required=True)
@click.option('normalized', '--normalized', is_flag=True, help='Whether or not ages in dataset.json are normalized', default=True)
def main(
    network: str, 
    dataset_json: str, 
    normalized: bool
    ):
    print(network)
    print(dataset_json)
    calc_age_category_from_json(dataset_json)



if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter