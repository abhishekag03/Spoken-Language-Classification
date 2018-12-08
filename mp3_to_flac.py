#!/usr/bin/env python
# coding: utf-8

# In[1]:''



import os
import subprocess
import pickle
# import numpy

d = dict()
# os.chdir('C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\English_Data\\')
array = pickle.load(open("list_multi_class", 'rb'))
print(os.getcwd())
for file in array:
    # arr = file.split('-')
    string = "ffmpeg -i " + "/mnt/c/Users/Lenovo/Desktop/Utility/Semester\ 5/Machine\ Learning/Project/Data/TopCoder_Data/Training\ data/" + file + " /mnt/c/Users/Lenovo/Desktop/Utility/Semester\ 5/Machine\ Learning/Project/Multi_Class_flac/" + file[:-4] + ".flac"
    print(string)
    os.system(string)    
    # break
