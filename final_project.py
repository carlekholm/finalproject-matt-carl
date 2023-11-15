import csv
import pandas as pd

plot_list = []
control_dict = {} 
file_name = 'movieplots.csv'

i = 0
data_frame = pd.read_csv(file_name)
for index, row in data_frame.iterrows():
    title = row['Title']
    genre = row['Genre']
    plot = row['Plot']
    control_dict[title] = {'genre':genre, 'plot': plot}
    i+=1
    # if i == 12:
    #     break