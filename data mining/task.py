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
from sklearn.linear_model import LogisticRegression

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
var_float = colTypes[colTypes['dtype'] == 'float64']['columns'].tolist()
var_obj = colTypes[colTypes['dtype'] == 'object']['columns'].tolist()

for col in var_float:
    col_mean = np.mean(data[col])
    data[col].fillna(col_mean, inplace=True)
for col in var_int:
    data[col].fillna(0, inplace=True)


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
    x_selected = x_train[:, importances > threshold]
    return x_selected
#x=data[var_int+var_float]
#y=targetDf['status']



#变量分箱
class FeatureBinning(object):
    def __init__(self, bin_type=2, criterion='entropy', max_leaf_nodes=8,
                 min_samples_leaf=100, max_depth=4, bin_count=20, na=-999,
                 minPctThd=0.05, minSamplesThd=5, pThd=0.05):
        self.na = na
        self.bin_type = bin_type
        self.criterion = criterion
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.bin_count = bin_count
        self.minPctThd = minPctThd
        self.minSamplesThd = minSamplesThd
        self.pThd = pThd

    def equalPercentBinning(self, var, y, bin_count=None):
        bin_count = bin_count if bin_count else self.bin_count
        bins = pd.qcut(var, bin_count, retbins=True, duplicates='drop')[1]
        bucket = var.apply(mapBin, bins=bins)
        return bucket, bins

    def decisionTreeBinning(self, var, y):
        min_samples_leaf = min(len(y) / 10, self.min_samples_leaf)
        params = {'criterion': self.criterion,
                  'max_leaf_nodes': self.max_leaf_nodes,
                  'min_samples_leaf': min_samples_leaf,
                  'max_depth': self.max_depth
                  }
        dt = DecisionTreeClassifier(**params)
        var2 = var.values.reshape(-1, 1)
        dt.fit(var2, y)
        bins = np.unique(dt.tree_.threshold[dt.tree_.feature > -2])
        if len(bins) > 0:
            bins = updBin(bins)
            bucket = var.apply(mapBin, bins=bins)
            return bucket, bins
        return '', []

    def linearBinning(self, var, y, bin_count=None):
        bin_count = bin_count if bin_count else self.bin_count
        r = 0
        n = bin_count
        bucket = None
        while np.abs(r) < 1 and n >= 5:
            bins = pd.qcut(var.dropna(), n, retbins=True, duplicates='drop')[1]
            if var.count() != len(var):
                bins = np.concatenate(([self.na], bins))
                bucket = var.apply(mapBin, bins=bins)
            else:
                bucket = var.apply(mapBin, bins=bins)
                d1 = pd.DataFrame({"variable": var, "Y": y, "Bucket": bucket})
                d2 = d1.groupby('Bucket', as_index=False)
            r, p = stat.stats.spearmanr(d2.mean().variable, d2.mean().Y)
            n = n - 1
        return bucket, bins

        # 分类变量手工调整分箱

    def categoryBinning(self, var, y, bins_dict=None):
        if bins_dict is None:
            print('Please set bins_dict')
            bucket = None
        else:
            if var.name in bins_dict:
                bins = bins_dict[var.name]
                var_unique = var.unique().tolist()
                labels_dict = {}
                for i in bins:
                    for j in var_unique:
                        if j in i:
                            labels_dict[j] = ','.join(i)
                bucket = var.map(labels_dict)
        return bucket, bins

    def noBinning(self, var):
        bucket = var
        return bucket, None

    # 连续变量手工调整分箱
    def handBinning(self, var, y, bins_dict=None):
        if bins_dict is None:
            print('Please set bins_dict')
            bucket = None
        else:
            if var.name in bins_dict:
                bins = bins_dict[var.name]
                labels = ["{}~{}".format(bins[j], bins[j + 1])
                          for j in range(0, len(bins) - 1)]
                bucket = pd.cut(var, bins=bins, include_lowest=True,
                                labels=labels).astype(str)
            else:
                print(str(var.name) + ' is not in bins_dict')
                bucket = None
        return bucket, bins

    # 单调最优分箱
    def monotoneOptimalBinning(self, var, y, sign=False):
        varMax = max(var)
        if np.abs(varMax) >= 1000.0:
            scaler = np.power(10, int(np.log10(varMax)))
        else:
            scaler = 1.0
        var = (var / scaler).round(3)
        n_Thd = int(var.count() * self.minPctThd)
        div_ratio = 1.0
        df = pd.DataFrame(dict(var=var, y=y))
        df = df.loc[pd.notnull(df['var']), :]
        df = df.apply(lambda x: x.astype(float), axis=0)
        df_group = df.groupby('var').agg({'y': ['mean', 'size', 'std']},
                                         drop_index=True)
        df_group.columns = ["means", "samples", "std_dev"]
        df_group = df_group.reset_index()
        df_group['del_flag'] = 0
        df_group["std_dev"] = df_group["std_dev"].fillna(0)
        df_group = df_group.sort_values('var', ascending=sign)
        while True:
            i = 0
            df_group = df_group[df_group.del_flag != 1]
            df_group = df_group.reset_index(drop=True)
            while True:
                j = i + 1
                if j >= len(df_group):
                    break
                if df_group.iloc[j].means < df_group.iloc[i].means:
                    i = i + 1
                    continue
                else:
                    while True:
                        n = df_group.iloc[j].samples + df_group.iloc[i].samples
                        m = (df_group.iloc[j].samples * df_group.iloc[j].means
                             + df_group.iloc[i].samples * df_group.iloc[i].means) / n
                        if n == 2:
                            s = np.std([df_group.iloc[j].means,
                                        df_group.iloc[i].means])
                        else:
                            s = np.sqrt((df_group.iloc[j].samples *
                                         (df_group.iloc[j].std_dev ** 2)
                                         + df_group.iloc[i].samples *
                                         (df_group.iloc[i].std_dev ** 2)) / n)
                        df_group.loc[i, "samples"] = n
                        df_group.loc[i, "means"] = m
                        df_group.loc[i, "std_dev"] = s
                        df_group.loc[j, "del_flag"] = 1
                        j = j + 1
                        if j >= len(df_group):
                            break
                        if df_group.loc[j, "means"] < df_group.loc[i, "means"]:
                            i = j
                            break
                if j >= len(df_group):
                    break
            dels = np.sum(df_group["del_flag"])
            if dels == 0:
                break
        while True:
            df_group["means_lead"] = df_group["means"].shift(-1)
            df_group["samples_lead"] = df_group["samples"].shift(-1)
            df_group["std_dev_lead"] = df_group["std_dev"].shift(-1)
            df_group["est_samples"] = (df_group["samples_lead"]
                                       + df_group["samples"])
            df_group["est_means"] = (df_group["means_lead"]
                                     * df_group["samples_lead"]
                                     + df_group["means"] * df_group["samples"]) \
                                    / df_group["est_samples"]
            df_group["est_std_dev2"] = (df_group["samples_lead"]
                                        * df_group["std_dev_lead"] ** 2
                                        + df_group["samples"] * df_group["std_dev"] ** 2) \
                                       / (df_group["est_samples"] - 2)
            df_group["z_value"] = ((df_group["means"] - df_group["means_lead"])
                                   / np.sqrt(df_group["est_std_dev2"] * (1 / df_group["samples"]
                                                                         + 1 / df_group["samples_lead"])))
            df_group["p_value"] = 1 - stat.norm.cdf(df_group["z_value"])
            condition = ((df_group["samples"] < n_Thd) |
                         (df_group["samples_lead"] < n_Thd) |
                         (df_group["means"] * df_group["samples"] < self.minSamplesThd) |
                         (df_group["means_lead"] * df_group["samples_lead"]
                          < self.minSamplesThd))
            df_group[condition].p_value = df_group[condition].p_value + 1
            df_group["p_value"] = df_group.apply(lambda row: row["p_value"] + 1
            if (row["samples"] < n_Thd) |
               (row["samples_lead"] < n_Thd) |
               (row["means"] * row["samples"] < self.minSamplesThd) |
               (row["means_lead"] * row["samples_lead"] < self.minSamplesThd)
            else row["p_value"], axis=1)
            max_p = max(df_group["p_value"])
            row_of_maxp = df_group["p_value"].idxmax()
            row_delete = row_of_maxp + 1
            if max_p > self.pThd:
                df_group = df_group.drop(df_group.index[row_delete])
                df_group = df_group.reset_index(drop=True)
            else:
                break
            df_group["means"] = df_group.apply(lambda row: row["est_means"]
            if row["p_value"] == max_p else row["means"], axis=1)
            df_group["samples"] = df_group.apply(lambda row: row["est_samples"]
            if row["p_value"] == max_p else row["samples"], axis=1)
            df_group["std_dev"] = df_group.apply(lambda row:
                                                 np.sqrt(row["est_std_dev2"]) if row["p_value"] == max_p
                                                 else row["std_dev"], axis=1)
        bins = (np.unique(df_group['var']).astype(float) - 1e-4) * scaler
        bins = updBin(bins)
        var = var * scaler
        bucket = var.apply(mapBin, bins=bins)
        return bucket, bins

    def binning_series(self, var, y, bin_type=None,
                       bin_count=None, bins_dict=None):
        bin_type = (bin_type if bin_type in range(1, 8) else self.bin_type)
        bin_count = bin_count if bin_count else self.bin_count
        var1 = var.fillna(self.na)
        if bin_type == 1:
            bucket, bins = self.equalPercentBinning(var1,
                                                    y, bin_count=bin_count)
        elif bin_type == 2:
            bucket, bins = self.decisionTreeBinning(var1, y)
            if bucket is None:
                bucket = self.noBinning(var1)
            else:
                bucket
        elif bin_type == 3:
            bucket, bins = self.linearBinning(var1, y, bin_count=bin_count)
        elif bin_type == 4:
            bucket, bins = self.categoryBinning(var1, y, bins_dict=bins_dict)
        elif bin_type == 5:
            bucket, bins = self.noBinning(var1)
        elif bin_type == 6:
            bucket, bins = self.handBinning(var1, y, bins_dict=bins_dict)
        elif bin_type == 7:
            bucket_ascend, bins_ascend = self.monotoneOptimalBinning(
                var1, y, sign=True)
            bucket_descend, bins_descend = self.monotoneOptimalBinning(
                var1, y, sign=False)
            iv_ascend = calInformationValue(bucket_ascend, y)
            iv_descend = calInformationValue(bucket_descend, y)
            if iv_ascend < iv_descend:
                bucket, bins = bucket_descend, bins_descend
            else:
                bucket, bins = bucket_ascend, bins_ascend
        else:
            print(u'已没有其他选项')
        return bucket, bins

        # DataFrame 在满足IV阈值条件下变量分箱

    def binning_df(self, df, binning_feature, label, bin_type=None,
                   bin_count=None, bins_dict=None, ivThd=0.02):
        bin_type = bin_type if bin_type in range(1, 8) else self.bin_type
        binsDict = {}
        ivDict = {}
        df = df.copy()
        df_columns = df.columns.tolist()
        df_columns.remove(label)
        in_columns = [i for i in df_columns if i in binning_feature]
        out_columns = [i for i in binning_feature if i not in df_columns]
        if len(out_columns) > 0:
            print(out_columns, 'not in dataframe!')
        start = 0
        for i in in_columns:
            start += 1
            print('start binning var%s: %s' % (start, i))
            bucket, bins = self.binning_series(df[i], df[label],
                                               bin_type=bin_type, bin_count=bin_count,
                                               bins_dict=bins_dict)
            Iv = calInformationValue(bucket, df[label])
            if Iv >= ivThd:
                df[i + '_bin'] = bucket
                ivDict[i] = Iv
                binsDict[i] = bins
        print('')
        return df, ivDict, binsDict

    # 自动分箱, 结合决策树与单调最优
    def auto_binning_df(df, binning_feature, label, criterion='entropy',
                        max_leaf_nodes=8, min_samples_leaf=100, max_depth=4,
                        minPctThd=0.05, minSamplesThd=5, pThd=0.05, ivThd=0.02):
        fb = FeatureBinning(bin_type=2, criterion=criterion,
                            max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                            max_depth=max_depth, minPctThd=minPctThd, minSamplesThd=minSamplesThd,
                            pThd=pThd)
        print("Start Decision Tree binning:")
        dtBinningDf, dtIvDict, dtBinsDict = fb.binning_df(df, binning_feature,
                                                          label, ivThd=ivThd, bin_type=2)
        dt_binning_feature = list(dtIvDict.keys())
        print('The no. of features beyond the threshold of Decision Tree Binning: %s'
              % len(dt_binning_feature))
        print("Start Monotor Optimal Binning:")
        binningDf, ivDict, bins_dict = fb.binning_df(df, dt_binning_feature,
                                                     label, ivThd=ivThd, bin_type=7)
        print('The no. of features beyond the threshold of Monotor Optimal Binning: %s'
              % len(bins_dict))
        dtIvDf = pd.DataFrame(pd.Series(dtIvDict)).reset_index() \
            .rename(columns={'index': 'variable', 0: 'dt_IV'})
        mobIvDf = pd.DataFrame(pd.Series(ivDict)).reset_index() \
            .rename(columns={'index': 'variable', 0: 'mob_IV'})
        ivDf = dtIvDf.merge(mobIvDf, on='variable', how='left')
        cache = {'binningDf': binningDf,
                 'dtBinningDf': dtBinningDf,
                 'ivDf': ivDf,
                 'dtBinsDict': dtBinsDict,
                 'bins_dict': bins_dict}
        return cache


#计算woe、iv
stats_all_df = pd.DataFrame()
bad_weight, good_weight = 1, 1
qushi = "up"
for i in bin_cols:
    trans_df = pd.crosstab(data[i], data["target"])
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
    trans_df['iv'] = trans_df['iv_i'].sum()
    trans_df['iv'] = round(trans_df['iv'], 4)

    woe_df = trans_df.sort_values(by='min_value').drop(['iv_i', 'min_value'],
                                                       axis=1) if qushi == 'up' else trans_df.sort_values(
        by='min_value', ascending=Fasle).drop(['iv_i', 'min_value'], axis=1)
    woe_df['bad_cum'] = (woe_df.bad_count_weight / woe_df.bad_count_weight.sum()).cumsum()  # 累加
    woe_df['good_cum'] = (woe_df.good_count_weight / woe_df.good_count_weight.sum()).cumsum()
    woe_df = woe_df.drop(['good_count_weight', 'bad_count_weight', 'total_weight'], axis=1)
    stats_all_df.append(feature_stats)


lr = LogisticRegression()
lr.fit(x_train, y_train)
y_predict = lr.predict(x)
