# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 22:16:24 2021
GEO处理中的第一步
对TLE计算出来的参数mat文件进行读取并处理生成样本
每个目标的前80%训练，后20%测试。
@author: Lenovo
"""

import scipy.io as scio
import pandas as pd
import glob,os
import numpy as np
import math

def _slide_window(rows, sw_width, sw_steps):
    '''
    函数功能：
    按指定窗口宽度和滑动步长实现单列数据截取
    --------------------------------------------------
    参数说明：
    rows：单个文件中的行数；
    sw_width：滑动窗口的窗口宽度；
    sw_steps：滑动窗口的滑动步长；
    '''
    start = 0
    s_num = (rows - sw_width) // sw_steps # 计算滑动次数
    new_rows = sw_width + (sw_steps * s_num) # 完整窗口包含的行数，丢弃少于窗口宽度的采样数据；
    
    while True:
        if (start + sw_width) > new_rows: # 如果窗口结束索引超出最大索引，结束截取；
            return
        yield start, start + sw_width
        start += sw_steps

home_dir =os.getcwd()

path = r'data/debris'#line A
file = glob.glob(os.path.join(path,"*.mat"))

database=[]
x=0
j=0
for f in file:
      data = scio.loadmat(f)
      dataset = data['TLE_one']
      dataset = dataset[[not np.all(dataset[i] == 0) for i in range(dataset.shape[0])], :]
      datal=pd.DataFrame(dataset).T
      datal.columns=['编号','年','DOY','MJD','轨道倾角','升交点赤经','偏心率','近地点角距','半长轴','平近点角'
                     ,'近地点','远地点','周期','u','n','v','x','y','g','星下点纬度','星下点经度'
                     ,'year','month','day','hour','minute','seconds']
      perios=pd.to_datetime(datal[['year', 'month', 'day']],errors='coerce')
      Date=pd.DataFrame(perios)
      Data=pd.concat([datal,Date],axis=1)
      #论文中用轨道6根数，解释起来比较容易（半长轴，偏心率，轨道倾角，升交点赤经，近地点角距，平近点角）
      ts=Data.drop(Data.columns[[0,1,2,3,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]],axis=1)#去掉不要的列，只剩下编号和四个参数
      ts.columns= ['轨道倾角','升交点赤经','偏心率','近地点角距','半长轴','平近点角'
                     # ,'近地点','远地点','周期','星下点纬度','星下点经度'
                     ,'时间']
      cols=list(ts)#列的List
      cols.insert(0,cols.pop(cols.index('时间')))#把时间列拖到第一位
      d=ts.loc[:,cols]#按照clos排列程新的矩阵
      df_1=d.drop_duplicates(subset='时间',keep='last')#以时间为特征，删掉时间重复的行，默认设置，取最后一个值
      #以时间为index对卫星数据的前两个月的数据进行删除，也就是不要爬升阶段的。
      # df_1=df_1.iloc[20:,]
      ###此处注意修改
      df_l=np.column_stack((df_1,np.ones(len(df_1))*0))#添加标签，半长轴为1，碎片为0，半长轴加倾角2
      #此处选择前80%作为训练集，后20%作为测试集
      train = df_l[0:math.ceil(0.8*len(df_l)), :]
      test = df_l[math.ceil(0.8*len(df_l)):len(df_l),:]
      for start,end in _slide_window(len(train),20,1):
          data2=train[start:end]
          x=x+1
          np.savetxt(r'run-p1\traindata\deb_sat{}{}{}.csv'.format(x,start,end),data2,delimiter=',',fmt ='%s')#line B
      for start,end in _slide_window(len(test),20,1):
          data3=test[start:end]
          j=j+1
          np.savetxt(r'run-p1\testdata\deb_sat{}{}{}.csv'.format(j,start,end),data3,delimiter=',',fmt ='%s')#line B     




path = r'data/a-i'#line A
file = glob.glob(os.path.join(path,"*.mat"))

database=[]
x=0
j=0
for f in file:
      data = scio.loadmat(f)
      dataset = data['TLE_one']
      dataset = dataset[[not np.all(dataset[i] == 0) for i in range(dataset.shape[0])], :]
      datal=pd.DataFrame(dataset).T
      datal.columns=['编号','年','DOY','MJD','轨道倾角','升交点赤经','偏心率','近地点角距','半长轴','平近点角'
                     ,'近地点','远地点','周期','u','n','v','x','y','g','星下点纬度','星下点经度'
                     ,'year','month','day','hour','minute','seconds']
      perios=pd.to_datetime(datal[['year', 'month', 'day']],errors='coerce')
      Date=pd.DataFrame(perios)
      Data=pd.concat([datal,Date],axis=1)
      #论文中用轨道6根数，解释起来比较容易（半长轴，偏心率，轨道倾角，升交点赤经，近地点角距，平近点角）
      ts=Data.drop(Data.columns[[0,1,2,3,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]],axis=1)#去掉不要的列，只剩下编号和四个参数
      ts.columns= ['轨道倾角','升交点赤经','偏心率','近地点角距','半长轴','平近点角'
                     # ,'近地点','远地点','周期','星下点纬度','星下点经度'
                     ,'时间']
      cols=list(ts)#列的List
      cols.insert(0,cols.pop(cols.index('时间')))#把时间列拖到第一位
      d=ts.loc[:,cols]#按照clos排列程新的矩阵
      df_1=d.drop_duplicates(subset='时间',keep='last')#以时间为特征，删掉时间重复的行，默认设置，取最后一个值
      #以时间为index对卫星数据的前两个月的数据进行删除，也就是不要爬升阶段的。
      # df_1=df_1.iloc[20:,]
      ###此处注意修改
      df_l=np.column_stack((df_1,np.ones(len(df_1))*2))#添加标签，半长轴为1，碎片为0，半长轴加倾角2
      #此处选择前80%作为训练集，后20%作为测试集
      train = df_l[0:math.ceil(0.8*len(df_l)), :]
      test = df_l[math.ceil(0.8*len(df_l)):len(df_l),:]
      for start,end in _slide_window(len(train),20,1):
          data2=train[start:end]
          x=x+1
          np.savetxt(r'run-p1\traindata\ai_sat{}{}{}.csv'.format(x,start,end),data2,delimiter=',',fmt ='%s')#line B
      for start,end in _slide_window(len(test),20,1):
          data3=test[start:end]
          j=j+1
          np.savetxt(r'run-p1\testdata\ai_sat{}{}{}.csv'.format(j,start,end),data3,delimiter=',',fmt ='%s')#line B     






path = r'data/a-only'#line A
file = glob.glob(os.path.join(path,"*.mat"))

database=[]
x=0
j=0
for f in file:
      data = scio.loadmat(f)
      dataset = data['TLE_one']
      dataset = dataset[[not np.all(dataset[i] == 0) for i in range(dataset.shape[0])], :]
      datal=pd.DataFrame(dataset).T
      datal.columns=['编号','年','DOY','MJD','轨道倾角','升交点赤经','偏心率','近地点角距','半长轴','平近点角'
                     ,'近地点','远地点','周期','u','n','v','x','y','g','星下点纬度','星下点经度'
                     ,'year','month','day','hour','minute','seconds']
      perios=pd.to_datetime(datal[['year', 'month', 'day']],errors='coerce')
      Date=pd.DataFrame(perios)
      Data=pd.concat([datal,Date],axis=1)
      #论文中用轨道6根数，解释起来比较容易（半长轴，偏心率，轨道倾角，升交点赤经，近地点角距，平近点角）
      ts=Data.drop(Data.columns[[0,1,2,3,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]],axis=1)#去掉不要的列，只剩下编号和四个参数
      ts.columns= ['轨道倾角','升交点赤经','偏心率','近地点角距','半长轴','平近点角'
                     # ,'近地点','远地点','周期','星下点纬度','星下点经度'
                     ,'时间']
      cols=list(ts)#列的List
      cols.insert(0,cols.pop(cols.index('时间')))#把时间列拖到第一位
      d=ts.loc[:,cols]#按照clos排列程新的矩阵
      df_1=d.drop_duplicates(subset='时间',keep='last')#以时间为特征，删掉时间重复的行，默认设置，取最后一个值
      #以时间为index对卫星数据的前两个月的数据进行删除，也就是不要爬升阶段的。
      # df_1=df_1.iloc[20:,]
      ###此处注意修改
      df_l=np.column_stack((df_1,np.ones(len(df_1))*1))#添加标签，半长轴为1，碎片为0，半长轴加倾角2
      #此处选择前80%作为训练集，后20%作为测试集
      train = df_l[0:math.ceil(0.8*len(df_l)), :]
      test = df_l[math.ceil(0.8*len(df_l)):len(df_l),:]
      for start,end in _slide_window(len(train),20,1):
          data2=train[start:end]
          x=x+1
          np.savetxt(r'run-p1\traindata\a_sat{}{}{}.csv'.format(x,start,end),data2,delimiter=',',fmt ='%s')#line B
      for start,end in _slide_window(len(test),20,1):
          data3=test[start:end]
          j=j+1
          np.savetxt(r'run-p1\testdata\a_sat{}{}{}.csv'.format(j,start,end),data3,delimiter=',',fmt ='%s')#line B     



















