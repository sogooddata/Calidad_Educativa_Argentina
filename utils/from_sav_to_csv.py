import pandas as pd
import pyreadstat

import os, re

PATH_RAW = '../data/raw/'

files = []

#Read all files at PATH_RAW and filter only .sav file
for path in os.listdir(PATH_RAW):
     for file in os.listdir(PATH_RAW+f'{path}/'):
         if '.sav' in file:
            files.append(PATH_RAW+f'{path}/'+f'{file}')

#Read the .sav files and save as csv
for file in files:
    df, meta  = pyreadstat.read_sav(file)
    df.to_csv(file.replace('sav','csv'),index=False)