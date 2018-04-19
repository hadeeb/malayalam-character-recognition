"""Module to clean the raw images"""
from functions import *


RAW_FOLDER = 'orig_data'
CLEAN_FOLDER = 'data'

FOLDER_LIST = list_folders(RAW_FOLDER)
print(FOLDER_LIST)

create_folders(CLEAN_FOLDER, FOLDER_LIST)
process_images(RAW_FOLDER, CLEAN_FOLDER, FOLDER_LIST)
