import os
import shutil
import Augmentor
from functions import list_folders

folder = 'raw_data'
for f in list_folders(folder):
    if os.path.isdir(os.path.join(folder, f, 'output')):
        shutil.rmtree(os.path.join(folder, f, 'output'))
    p = Augmentor.Pipeline(os.path.join(folder, f))
    p.random_distortion(probability=1, grid_width=10, grid_height=10, magnitude=8)
    p.sample(500, multi_threaded=False)
