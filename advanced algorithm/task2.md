## GBDT算法梳理

### 1. 前向分布算法
对于一个加法模型(由M个基函数按系数相加组成)，在给定训练数据集和损失函数的条件下，学习加法模型成为损失函数最小化的问题。  
通常这是一个复杂的优化问题，前向分布算法求解这一优化问题的思想是：因为学习的是加法模型，如果能够从前向后(按照线性关系的顺序)，每一步之学习
一个奇函数及其系数，逐渐逼近优化目标函数式，那么就可以简化优化的复杂度。

### 2. 负梯度拟合
#### 提升树
采用加法模型与前向分步算法，以决策树作为基函数的提升方法称为提升树。提升树模型可以表示为决策树的加法模型。  
回归问题提升树使用前向分步算法，在第m步中，给定当前模型fm-1(x)，求解使得平方误差函数最小的参数进而得到第m棵树。  
其中平方损失函数L=[y-fm-1(x)-T]^2=[r-T]^2.其中r=y-fm-1(x)。  
*个人理解：即在拟合第m棵树的时候，实际上拟合第目标是使第m棵树的预测值与前面m-1棵树的预测值的误差尽可能相近，用第m棵树来减少前面m-1棵树的误差。*
#### 梯度提升
提升树利用加法模型与前向分步算法实现学习的优化过程。当损失函数是平方损失函数和指数损失函数时，每一步优化是很简单的，但对一般的损失函数而言，往往每
一步优化并不那么容易。针对这一问题，Freidman提出了梯度提升算法。这是利用最速下降法的近似方法，其关键是利用损失函数的负梯度在当前模型的值作为回归问题
提升树算法中的残差的近似值，拟合一个回归树。

### 3. 损失函数
对于分类算法，其损失函数一般有对数损失函数和指数损失函数两种:  
如果是指数损失函数，则损失函数表达式为其负梯度计算和叶子节点的最佳残差拟合  
如果是对数损失函数，分为二元分类和多元分类两种。  
对于回归算法，常用损失函数有如下4种:  
1）均方差  
2）绝对损失  
3）Huber损失，它是均方差和绝对损失的折衷产物，对于远离中心的异常点，采用绝对损失，而中心附近的点采用均方差。这个界限一般用分位数点度量。  
4）分位数损失。它对应的是分位数回归的损失函数。  

### 4. 回归
输入是训练集样本， 最大迭代次数T, 损失函数L。  
输出是强学习器f(x)  
1）初始化弱学习器  
2）对迭代轮数t=1,2,...T有：  
a)对样本i=1,2，...m，计算负梯度  
b)利用, 拟合一颗CART回归树,得到第t颗回归树，其对应的叶子节点区域为。其中J为回归树t的叶子节点的个数。  
c) 对叶子区域j =1,2,..J,计算最佳拟合值  
d) 更新强学习器  
3）得到强学习器f(x)的表达式  

### 5. 二分类，多分类
这里看看GBDT分类算法，GBDT的分类算法从思想上和GBDT的回归算法没有区别，但是由于样本输出不是连续的值，而是离散的类别，导致我们无法直接从输出类别去拟合输出类别的误差。
为了解决这个问题，主要有两个方法，一个是用指数损失函数，此时GBDT退化为Adaboost算法。另一种方法用类似逻辑回归的对数似然函数的方法。也就是说，我们用的是类别的预测概率值和真实概率值的差来拟合损失。此处仅讨论用对数似然函数的GBDT分类。对于对数似然损失函数，我们有又有二元分类和的多元分类的区别。

### 6. 正则化
BDT的正则化主要有三种方式。
第一种是和Adaboost类似的正则化项，即步长(learning rate)。对于同样的训练集学习效果，较小的意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定算法的拟合效果。  
第二种正则化的方式是通过子采样比例（subsample）。取值为(0,1]。注意这里的子采样和随机森林不一样，随机森林使用的是放回抽样，而这里是不放回抽样。如果取值为1，则全部样本都使用，等于没有使用子采样。如果取值小于1，则只有一部分样本会去做GBDT的决策树拟合。选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低。推荐在[0.5, 0.8]之间。  
使用了子采样的GBDT有时也称作随机梯度提升树(Stochastic Gradient Boosting Tree, SGBT)。由于使用了子采样，程序可以通过采样分发到不同的任务去做boosting的迭代过程，最后形成新树，从而减少弱学习器难以并行迭代的弱点。  
第三种是对于弱学习器即CART回归树进行正则化剪枝。  

### 7. 优缺点
GBDT主要的优点有：  
1) 可以灵活处理各种类型的数据，包括连续值和离散值。  
2) 在相对少的调参时间情况下，预测的准备率也可以比较高。这个是相对SVM来说的。  
3）使用一些健壮的损失函数，对异常值的鲁棒性非常强。比如 Huber损失函数和Quantile损失函数。  
  
GBDT的主要缺点有：  
1)由于弱学习器之间存在依赖关系，难以并行训练数据。不过可以通过自采样的SGBT来达到部分并行。  

### 8. sklearn参数
```
from sklearn.ensemble import GradientBoostingRegressor

GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, 
                subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3,
                min_impurity_decrease=0.0, min_impurity_split=None, init=None, 
                random_state=None, max_features=None, alpha=0.9, verbose=0, 
                max_leaf_nodes=None, warm_start=False, presort='auto', 
                validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)
```

                
### 9. 应用场景
特征选择、模型预测

参考文章、书籍：
1.统计学习方法。
2.https://blog.csdn.net/u014465639/article/details/73911669