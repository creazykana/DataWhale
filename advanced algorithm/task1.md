## Task 1 随机森林算法梳理


#### 1.集成学习的概念
集成学习就是组合这里的多个弱监督模型(有偏好的模型，在某些方面表现的比较好)以期得到一个更好更全面的强监督模型，集成学习潜在的思想是即便某一个弱分类器得到了错误的预测，其他的弱分类器也可以将错误纠正回来。


#### 2.个体学习器的概念
个体学习器是一个相对的概念，集成之前的学习器称为个体学习器


#### 3.boosting bagging的概念、异同点
*boosting bagging都是将已有的效果偏差的模型通过一定方式进行组合形成一个更强的分类器。*
**boosting**：把样本做K次有放回的抽样；每次抽样训练一个模型共K个；用K个模型投票或均值作为最后结果；
**bagging**：初始化时对每一个训练赋予相同的权重1/n，然后用该算法对训练集训练t轮，每次训练后，对训练失败的训练列赋予较大的权重，也就是让学习算法在后续的学习中集中对比较难的训练列进行训练（就是把训练分类错了的样本，再次拿出来训练，看它以后还敢出错不），从而得到一个预测函数序列h_1,h_m,其中h_i也有一定的权重，预测效果好的预测函数权重大，反之小。
##### 两者的区别：
二者的主要区别是取样本方式不同。(1)bagging采用均匀取样，而boosting根据错误率来采样，因此boosting的分类精度要由于bagging。baging的训练集选择是随机的，各轮训练集之前互相独立，而boosting的各轮训练集的选择与前面各轮的学习结果相关；(2)bagging的各个预测函数没有权重，而boost有权重；
baging分类器的权重是一致的，boosting的分类器权重不一样(准确率高的分类器权重越大)
bagging的各个函数可以并行生成，而boosting的各个预测函数只能顺序生成。对于象神经网络这样极为消耗时间的算法，bagging可通过并行节省大量的时间开销。baging和boosting都可以有效地提高分类的准确性。在大多数数据集中，boosting的准确性要比bagging高。有一些数据集总，boosting会退化-overfit。boosting思想的一种改进型adaboost方法在邮件过滤，文本分类中有很好的性能。


#### 4.理解不同的结合策略(平均法，投票法，学习法)
**平均法**应该是主要针对回归模型，将模型结果取平均值最为最后的预测结果；
**投票法**应该是主要针对分类模型，分类结果中出现次数最多的类型作为最终分类结果；
**学习法(stacking)**，不是对弱学习器的结果做简单的逻辑处理，而是再加上一层学习器。也就是说，我们将训练集弱学习器的学习结果作为输入，将训练集的输出作为输出，重新训练一个学习器来得到最终结果。


#### 5.随机森林的思想
用bagging的思想选出K个训练子集，对每个子集用决策树作为分类器进行训练，但是决策树入模的变量也是从全部变量中随机取一部分出来的。最终训练出K个决策树组成森林。个人理解随机主要体现在训练集选取随机和训练变量随机。


#### 6.随机森林的推广
- extra trees
是RF的一个变种, 原理几乎和RF一模一样，仅有区别有：1.extra trees一般不采用随机采样，即每个决策树采用原始训练集；2.在选定了划分特征后，extra trees会随机的选择一个特征值来划分决策树；
- Totally Random Trees Embedding
(以下简称 TRTE)是一种非监督学习的数据转化方法。它将低维的数据集映射到高维，从而让映射到高维的数据更好的运用于分类回归模型。


#### 7.随机森林的优缺点
- 优点：
baging的优点，训练时可以并行化，速度快精度高
由于采用了随机抽样，训练出来的模型的方差小，泛化能力强
实现简单，对部分特征缺失比较不敏感
- 缺点：
在某些噪音比较大的样本集上，RF模型容易陷入过拟合
取值划分比较多的特征容易对RF的决策产生更大的影响，从而影响拟合的模型的效果


8.随机森林在sklearn中的参数解释
```
class sklearn.ensemble.RandomForestClassifier(n_estimators=’warn’, criterion=’gini’, max_depth=None, min_samples_split=2,
min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0,
min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False,
class_weight=None)
```
>n_estimators：integer, optional (default=10，0.22版本默认值为100)；随机森林中用到树的数量
>criterion : string,optional(default="mse")；测量分割的好坏，mae(均方误差)/mae(平均绝对误差)
>max_depth : integer or None,optional (default=None)；树的最大深度
>min_samples_split : int,float,optional(default=2)；分割一个内部节点所需最小样本数
>min_samples_leaf : int,float,optional(default=1)；叶节点上所需最小样本数
>min_weight_fraction_leaf : float,optional(default=0.)叶节点所需的(所有输入样本的)权值之和的最小加权分数，当其为0(默认情况)时所有样本具有相同的权重
>max_features : int,float,string("auto","sqrt","log2") or None,optional(default="auto")；每次在寻找最佳分割时需要考虑的特性数量
>max_leaf_nodes : int or None, optional (default=None)；最大叶节点数
>min_impurity_decrease : float, optional (default=0.)；最小的杂质减少量，若一个节点能通过分割把杂质的量降到这个数值以下，那么分割就会进行
>min_impurity_split : float, (default=1e-7)；树早期停止生长的阈值，若一个节点的杂质超过阈值则会进行分裂，否则就是叶节点
>bootstrap : boolean, optional (default=True)；构建树时是否使用bootstrap samples
>oob_score : bool, optional (default=False)；是否使用袋外样本来估计不可示数据的R^2
>n_jobs : int or None, optional (default=None)；fit和predict并行运算的作业数
>random_state : int, RandomState instance or None, optional (default=None)；随机种子
>verbose : int, optional (default=0)；在拟合和预测时控制详细程度
>warm_start : bool, optional (default=False)；为True时重复使用上一个fit后的model并向整体添加更多的估算器



9.随机森林的应用场景
1. 缺失值填充
2. 变量选择
3. 模型预测

















