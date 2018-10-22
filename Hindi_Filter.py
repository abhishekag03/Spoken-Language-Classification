#!/usr/bin/env python
# coding: utf-8

# In[3]:


# import pandas as pd
import subprocess
# import numpy
import os


# In[15]:

mp3 = []
print(os.getcwd())
with open("./hindifile.txt") as f:
    for line in f:
        print(line)
        mp3.append("./Data/TopCoder_Data/Training data/"+str(line[:-1]))

        #subprocess.call(["cp","./Data/TopCoder_Data/Training data/" + line[:-1],"./Hindi_Data/"])
        #filenames.append(data[i,0])

for file in mp3:
    flac = file[:-4] + ".flac"
    subprocess.call(["ffmpeg", "-i", file, flac])