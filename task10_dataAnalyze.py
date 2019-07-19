# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:20:06 2019

@author: hongzk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

columns=["id", "date", "qty", "amt"]
data = pd.read_csv(r'C:/Users/hongzk/Desktop/CDNOW.txt',
                 header=None, names=["id", "date", "qty", "amt"], sep="\s+")


#数据质量及结构
data.info()  # 没有缺失值
for col in data.columns:
    # 检查每一列的数据类型组成
    colType = data[col].apply(lambda x: type(x)).value_counts()
    print("\n>>>column %s vars' types count:"%col)
    print(colType)
    # 检查每一列的异常值情况
    try:
        colDescribe = data[col].describe()
        print(">>>column %s vars' describe"%col)
        print(colDescribe)
    except:
        pass
data.groupby(["id", "date"])["date"].count().sort_values(ascending=False)  # 用户一天可以有多条记录



#数据分析
df = data.pivot_table(index=["id", "date"], values=["qty", "amt"], aggfunc=sum).reset_index()
df["price"] = df["amt"] / df["qty"]
df["yearmonth"] = df["date"].astype(str).apply(lambda x: x[:6])
df["date"] = df["date"].astype(str).apply(lambda x: datetime.strptime(x, "%Y%m%d"))

#订单趋势(4月开始骤降)
df.groupby("date")["qty"].sum().plot()
df.groupby("date")["amt"].sum().plot()
df.groupby("date")["price"].mean().plot()

#订单统计(按客户)
orderStat = df.groupby("id")["qty", "amt"].agg({"qty":["count", "sum"], "amt":"sum"})
orderStat.columns=["order_count", "qty_sum", "amt_sum"]
orderStat["qty_order_rate"] = orderStat["qty_sum"].apply(lambda x : x/np.sum(orderStat["qty_sum"]))

#订单统计(按客户+每月)
orderMonthlyStat = df.groupby(["id", "yearmonth"])["qty", "amt"].agg({"qty":["count", "sum"], "amt":"sum"}).reset_index()
orderMonthlyStat.columns=["id", "yearmonth", "order_count", "qty_sum", "amt_sum"]
orderMonthlyStat["monthly_qty_contri"] = orderMonthlyStat.groupby("yearmonth")["qty_sum"].apply(lambda x : x/np.sum(x))
max_monthly_qty_contri = orderMonthlyStat.groupby("yearmonth")["monthly_qty_contri"].max().reset_index()
max_monthly_qty_contri = max_monthly_qty_contri.merge(orderMonthlyStat[["yearmonth", "monthly_qty_contri", "id"]],
                             left_on=["yearmonth", "monthly_qty_contri"],
                             right_on=["yearmonth", "monthly_qty_contri"],
                             how = "left")
idContriSort = max_monthly_qty_contri["id"].value_counts().reset_index().rename(columns={"index":"id", "id":"counts"})

#最大单笔购买数量
df.sort_values(by="qty", ascending=False).head()



#购买时间分析
orderStat = orderStat.reset_index()
df_m2 = df[df["id"].isin(orderStat.loc[orderStat["order_count"]>1, "id"].tolist())]  # 挑出订单数量大于2的id订单记录
user_orderTime_stat = df_m2.groupby("id")["date"].agg(["min", "max"]).rename(columns={"min":"first", "max":"last"})
first_orderTime_count = user_orderTime_stat["first"].value_counts()
first_orderTime_count.plot()
last_orderTime_count = user_orderTime_stat["last"].value_counts()
last_orderTime_count.plot()  # 客户消亡比例增长趋势
user_orderTime_stat["life_period"] = (user_orderTime_stat["last"] - user_orderTime_stat["first"]).apply(lambda x : x.days)
life_period_stat = pd.DataFrame(user_orderTime_stat["life_period"].value_counts()).rename(columns={"life_period":"custs"}).sort_index()
life_period_stat.index.name="life_period"
life_period_stat.plot.bar()

user_max_order = df.groupby("id")["date"].agg("max").reset_index()
user_max_order['interval'] = user_max_order["date"].apply(lambda x: (datetime.now() - x).days)

