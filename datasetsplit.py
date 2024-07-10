# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:06:17 2021
将数据转成1453*31，采用常规的数据划分方式对其进行划分，0.3
创建网络啦
@author: Lenovo
"""

import numpy as np
import pandas as pd 
# from sklearn.model_selection import train_test_split
import glob, os
from sklearn.preprocessing import MinMaxScaler,StandardScaler
#分割数据集
home_dir =os.getcwd()

path = r'run-p1\traindata'
# path = r'E:\GEO-3\run-p19\traindata'
file = glob.glob(os.path.join(path, "*.csv"))
print(file)
# 多变量特征时，将数据转成cnn输入形式
#数据标题：时间，'轨道倾角','Omega','偏心率','w','半长轴','per','apo','T','星下点纬度','星下点经度'
scaler= StandardScaler()
y = []
for f, i in zip(file, range(len(file))):
    data = pd.read_csv(f, encoding='gbk',dtype=str,header=None)
    
    tsl_x = data.iloc[:, 1:7]#特征,(6根数，编号不取)
    #tsl_x = data.iloc[:,[1,2,3,4,5,10]]
    tsl_x = scaler.fit_transform(tsl_x)#对每一个样本都单独进行了归一化
    tsl_y = data.iloc[:, 7][0]
    y.append(tsl_y)
  
    tsl_x = np.expand_dims(tsl_x, axis=0)
    if i == 0:
        x = tsl_x
    else:
        x = np.concatenate([x, tsl_x], axis=0)

np.save(r'run-p1\trainx_wdf.npy', x)
np.save(r'run-p1\trainy_wdf.npy', y)


path = r'run-p1\testdata'
# path = r'E:\GEO-3\run-p19\traindata'
file = glob.glob(os.path.join(path, "*.csv"))
print(file)
# 多变量特征时，将数据转成cnn输入形式
#数据标题：时间，'轨道倾角','Omega','偏心率','w','半长轴','per','apo','T','星下点纬度','星下点经度'
scaler= StandardScaler()
y = []
for f, i in zip(file, range(len(file))):
    data = pd.read_csv(f, encoding='gbk',dtype=str,header=None)
    
    tsl_x = data.iloc[:, 1:7]#特征,(6根数，编号不取)
    #tsl_x = data.iloc[:,[1,2,3,4,5,10]]
    tsl_x = scaler.fit_transform(tsl_x)#对每一个样本都单独进行了归一化
    tsl_y = data.iloc[:, 7][0]
    y.append(tsl_y)
  
    tsl_x = np.expand_dims(tsl_x, axis=0)
    if i == 0:
        x = tsl_x
    else:
        x = np.concatenate([x, tsl_x], axis=0)

np.save(r'run-p1\testx_wdf.npy', x)
np.save(r'run-p1\testy_wdf.npy', y)

