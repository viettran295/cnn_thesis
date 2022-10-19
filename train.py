from helper import * 
import numpy as np 

path = 'data'
df = load_data(f'{path}/driving_log.csv')

# balance_data(df, 'Steering')

imgPath, steering = load_data_toArray(path, df)

print(imgPath[0])
print(steering[0])
