from helper import * 
import numpy as np 
from wandb_config import run

print(run.config.epoch)

# path = 'data'
# data = load_data(f"{path}/driving_log.csv")

# # balance_data(data, 'Steering')

# img_arr, steering_arr = load_data_toArray(path, data)
# img_batch, steering_batch = batch_generator(img_arr, steering_arr, 8, train_flag=True)
# print(len(img_batch))
# print(img_batch[0])

# print(len(steering_batch))
# print(steering_batch[0])