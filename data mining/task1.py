# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:25:57 2019

@author: hongzk
"""
# 要求：数据切分方式 - 三七分，其中测试集30%，训练集70%，随机种子设置为2018
import pandas as pd
import os


os.chdir(r'E:\DataWhale\data mining')

orgData = pd.read_csv('data.csv', encoding="gbk")

"""任务1：对数据进行探索和分析。时间：2天
数据类型的分析
无关特征删除
数据类型转换
缺失值处理
……以及你能想到和借鉴的数据分析处理"""