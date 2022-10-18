import pandas as pd 
import numpy as np 
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# edit name of 'center' columns
def getName(name: str) -> str:
    return name.split("\\")[-1]

# load data to Dataframe
def load_data(path):
    cols = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    df = pd.read_csv(path, names=cols)
    df['Center'] = df['Center'].apply(getName)
    return df 

# balance Steering data 
def balance_data(data, cols_name: str, display=True, nbins=31):
    hist, bin = np.histogram(data[cols_name], bins=nbins)
    samplePerBin = 1000
    center = (bin[:-1] + bin[1:]) * 0.5
    if display:
        plt.bar(center, hist, width=0.06)
        plt.plot((-1,1), (samplePerBin, samplePerBin))
        plt.show()

    # remove center angle to balance dataset 
    remove_list = [] 
    for i in range(nbins):
        bin_list = [] 
        for j in range(len(data[cols_name])):
            if data[cols_name][j] >= bin[i] and data[cols_name][j] <= bin[i+1]:
                bin_list.append(j)
        bin_list = shuffle(bin_list)
        bin_list = bin_list[samplePerBin:]
        remove_list.extend(bin_list)
    
    data.drop(data.index[remove_list], inplace=True)
    print('removed imgs: ', len(remove_list))
    print('remain imgs: ', len(data))

    if display:
        hist, bin = np.histogram(data[cols_name], nbins)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1,1), (samplePerBin, samplePerBin))
        plt.show()
