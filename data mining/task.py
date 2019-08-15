# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:25:57 2019

@author: hongzk
"""
# 要求：数据切分方式 - 三七分，其中测试集30%，训练集70%，随机种子设置为2018
import pandas as pd
import numpy as np
import os
#import scipy.stats as stat
#import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import train_test_split  # 切分训练集测试集
import matplotlib.pyplot as plt

# 中文乱码和坐标轴负号的处理
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']#
plt.rcParams['font.sans-serif']=['SimHei'] #用于显示图片中文
plt.rcParams['axes.unicode_minus'] = False
os.chdir(r'E:\dataWhaleData')

orgData = pd.read_csv('data.csv', encoding="gbk")

"""任务1：对数据进行探索和分析。时间：2天
数据类型的分析
无关特征删除
数据类型转换
缺失值处理
……以及你能想到和借鉴的数据分析处理"""


data = orgData.drop("Unnamed: 0", axis=1)
targetDf = data[["custid", "status"]]
data = data.drop("status", axis=1)
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
var_int.remove("custid")
var_float = colTypes[colTypes['dtype'] == 'float64']['columns'].tolist()
var_obj = colTypes[colTypes['dtype'] == 'object']['columns'].tolist()


for col in var_float:
    col_mean = np.mean(data[col])
    data[col].fillna(col_mean, inplace=True)
    data[col+"_bin"] = pd.qcut(data[col], 5, duplicates='drop')
for col in var_int:
    data[col].fillna(0, inplace=True)
    data[col+"_bin"] = pd.qcut(data[col], 5, duplicates='drop')
for col in var_obj:
    data[col].fillna("null", inplace=True)


"""任务2 - 特征工程（2天）
特征衍生
特征挑选：分别用IV值和随机森林等进行特征选择"""
#缺失值
def drop_null(df, threshold):
    recordNum = df.shape[0]
    tmp = pd.DataFrame(df.isnull().sum(axis=0)).reset_index()
    tmp.columns = ["colName", "nullNum"]
    tmp["nullRate"] = tmp["nullNum"].apply(lambda x: x/recordNum)
    outCols = tmp[tmp["nullRate"] < threshold]["colName"].tolist()
    return df[outCols]



#相关系数筛选变量
def drop_corr(df, threshold):
    #  根据相关系数删除变量
    df_corr = df.corr().abs()
    corr_index = np.where(df_corr >= threshold)
    drop_cols = [df_corr.columns[y] for x, y in zip(*corr_index)
                 if x != y and x < y]
    df_left = df.loc[:, ~df.columns.isin(drop_cols)]
    return df_left


#随机森林挑选变量
def select_by_rf(df, cols, y, threshold):
    x_train = df[cols]
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y)
    importances = rfc.feature_importances_
#    indices = np.argsort(importances)[::-1]
    x_selected = x_train.iloc[:, importances > threshold]
    return x_selected
#x=data[var_int+var_float]
#y=targetDf['status']


#计算woe、iv
bin_cols = [i for i in data.columns if i.endswith("_bin")]
data["status"] = targetDf["status"]
bad_weight, good_weight = 1, 1
qushi = "up"
df_iv = pd.DataFrame()
for i in bin_cols:
    trans_df = pd.crosstab(data[i], data["status"])
    trans_df = trans_df.rename(
        columns={0: 'good_count', 1: 'bad_count', '0': 'good_count', '1': 'bad_count'})
    trans_df["total"] = trans_df["good_count"] + trans_df["bad_count"]
    trans_df['good_count_weight'] = trans_df['good_count'] * good_weight  # 每条bin特征字段对应的好客户数量*好客户权重(默认1)
    trans_df['bad_count_weight'] = trans_df['bad_count'] * bad_weight  # 每条bin特征字段对应的坏客户数量*坏客户权重(默认1)
    trans_df['total_weight'] = trans_df['good_count_weight'] + trans_df['bad_count_weight']  # 权重相加

    good_total = trans_df['good_count_weight'].sum()  # 好客户权重汇总
    bad_total = trans_df['bad_count_weight'].sum()  # 坏客户权重汇总
    all_ = good_total + bad_total  # 权重和
    trans_df['bin_pct'] = trans_df['total_weight'] / all_  # 所占比例

    trans_df["bad_rate"] = trans_df['bad_count_weight'].div(trans_df['total_weight'])
    trans_df['sample_bad_rate'] = bad_total / all_  # 所有坏客户的占比
    good_dist = np.nan_to_num(trans_df['good_count_weight'] / good_total)  # nan值用0替代inf值用有限值替代
    bad_dist = np.nan_to_num(trans_df['bad_count_weight'] / bad_total)

    trans_df['woe'] = np.log(bad_dist / good_dist)
    trans_df['woe'] = round(trans_df['woe'], 4)
    trans_df['iv_i'] = (bad_dist - good_dist) * trans_df['woe']
    col_iv = trans_df['iv_i'].sum()
    tmp = pd.DataFrame(data=[[i, col_iv]], columns=["colName", "iv"])
    df_iv = df_iv.append(tmp)

col_drop_iv = df_iv.sort_values(by="iv", ascending=False).head(30)["colName"].tolist()
col_drop_iv = [i[:-4] for i in col_drop_iv]
df_drop_corr = drop_corr(data[col_drop_iv], 0.7)
df_drop_rf = select_by_rf(df_drop_corr, df_drop_corr.columns, targetDf["status"], 0.04)


"""任务3 - 建模（2天）
用逻辑回归、svm和决策树；随机森林和XGBoost进行模型构建，评分方式任意，如准确率等。"""
X_train, X_test, y_train, y_test = train_test_split(df_drop_rf, targetDf["status"], test_size=0.3, random_state=2018)
from sklearn import metrics
def modEffect(y_test, y_predict):
    acc = metrics.accuracy_score(y_test, y_predict)
    precision = metrics.precision_score(y_test, y_predict)
    recall = metrics.recall_score(y_test, y_predict)
    f1 = metrics.f1_score(y_test, y_predict)
    auc= metrics.roc_auc_score(y_test, y_predict)
    print('准确率:{:.4f},精确率:{:.4f},召回率:{:.4f},f1-score:{:.4f},auc:{:.4f}'.format(acc, precision, recall, f1, auc))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_predict = lr.predict(X_test)
modEffect(y_test, y_predict)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_predict = rf.predict(X_test)
modEffect(y_test, y_predict)

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
modEffect(y_test, y_predict)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
modEffect(y_test, y_predict)

from xgboost import XGBClassifier as XGBC
xgb = XGBC()
xgb.fit(X_train, y_train)
y_predict = xgb.predict(X_test)
xgb.score(y_test, y_predict)
modEffect(y_test, y_predict)


fpr,tpr,threshold = metrics.roc_curve(y_test, y_predict)
# 计算AUC的值
roc_auc = metrics.auc(fpr,tpr)
plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')  # 绘制面积图
plt.plot(fpr, tpr, color='black', lw = 1)  # 添加边际线
plt.plot([0,1],[0,1], color = 'red', linestyle = '--')  # 添加对角线
plt.text(0.5,0.3,'ROC curve (area = %0.3f)' % roc_auc)  # 添加文本信息
plt.xlabel('1-Specificity')  # 添加x轴与y轴标签
plt.ylabel('Sensitivity')
plt.show()




