# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:25:57 2019

@author: hongzk
"""
# 要求：数据切分方式 - 三七分，其中测试集30%，训练集70%，随机种子设置为2018
import pandas as pd
import numpy as np
import os


os.chdir(r'E:\dataWhaleData')

orgData = pd.read_csv('data.csv', encoding="gbk")

"""任务1：对数据进行探索和分析。时间：2天
数据类型的分析
无关特征删除
数据类型转换
缺失值处理
……以及你能想到和借鉴的数据分析处理"""

data = orgData.drop("Unnamed: 0", axis=1)
data.info()
colTypes = pd.DataFrame(data.dtypes).reset_index()
colTypes.columns = ['columns', 'dtype']
colTypes.groupby(['dtype']).count()
"""         columns
dtype           
int64         12
float64       70
object         7 """
var_int = colTypes[colTypes['dtype'] == 'int64']['columns'].tolist()
var_float = colTypes[colTypes['dtype'] == 'float64']['columns'].tolist()
var_obj = colTypes[colTypes['dtype'] == 'object']['columns'].tolist()

for col in var_float:
    col_mean = np.mean(data[col])
    data[col].fillna(col_mean, inplace=True)
for col in var_int:
    data[col].fillna(0, inplace=True)

def drop_corr(df, threshold):
    """
    根据相关系数删除变量
    """
    df_corr = df.corr().abs()
    corr_index = np.where(df_corr >= threshold)
    drop_cols = [df_corr.columns[y] for x, y in zip(*corr_index)
                 if x != y and x < y]
    df_left = df.loc[:, ~df.columns.isin(drop_cols)]
    return df_left
num_df_dropCorr = drop_corr(data[var_int+var_float], threshold=0.9)
