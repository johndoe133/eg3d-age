#!/usr/bin/python
from PIL import Image
import os, sys

path = "datasets/FFHQ_128/00000/"
dirs = os.listdir( path )


def resize():
    for ind, item in enumerate(dirs):
        if ind % 10 == 0:
            print(f'{ind/len(dirs)*100}% completed')
        file = f'{path}{item}'
        im = Image.open(file)
        
        imResize = im.resize((512,512), Image.ANTIALIAS)
        imResize.save(file)
resize()