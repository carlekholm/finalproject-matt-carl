import csv
import numpy
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


plot_list = []
control_dict = {} 
file_name = 'movieplots.csv'

i = 0
data_frame = pd.read_csv(file_name)
for index, row in data_frame.iterrows():
    title = row['Title']
    genre = row['Genre']
    plot = row['Plot']
    genre = genre.split()
    if '/' in genre:
        genre.remove('/')
    if 'of' in genre:
        genre.remove('of')
    control_dict[title] = {'genre': genre, 'plot': plot}
    i+=1

genres = set()
for key in control_dict:
    genre_list = control_dict[key]['genre']
    for elem in genre_list:
        genres.add(elem)
#print(genres)
#print('len =', len(genres)) # We currently have 955 unique genre labels

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


modelname = "distilbert-base-cased"
# tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=True) # Error here!
'''
model = AutoModelForSequenceClassification.from_pretrained(modelname,
                                                            num_labels=955).to(device)
'''