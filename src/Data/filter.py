#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:26:07 2023

@author: pavel
"""


from glob import glob
import pandas as pd
import os

df = pd.read_csv('./Training_set.csv')
train_df = list(df['filename'].values)


train = glob('./train/*')

for i in train:
    name = i.split('/')[-1]
    if name in train_df:
        continue
    
    os.remove(i)
    print('not found')
    

