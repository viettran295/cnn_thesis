from helper import * 
import numpy as np 

df = load_data('data/driving_log.csv')
hist, bin = np.histogram(df['Steering'], bins=31)

for i in range(14,16):
    tmp = []
    for j in range(len(df['Steering'])):
        if df['Steering'][j] >= bin[i] and df['Steering'][j] <= bin[i+1]:
            tmp.append(j)
    print(len(tmp))
    print(tmp)
