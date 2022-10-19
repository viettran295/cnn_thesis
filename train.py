from helper import * 
import numpy as np 


origin = mpimg.imread('data/IMG/center_2022_10_18_15_59_57_016.jpg')
img = img_preprocessing(mpimg.imread('data/IMG/center_2022_10_18_15_59_57_016.jpg'))
plt.imshow(origin)
plt.show()
plt.imshow(img)
plt.show() 