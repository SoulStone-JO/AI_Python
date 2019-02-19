# KNN算法
---
## KNN算法概述
> KNN是通过测量不同特征值之间的距离进行分类。它的思路是：如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别，其中K通常是不大于20的整数。KNN算法中，所选择的邻居都是已经正确分类的对象。该方法在定类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。 在KNN中，通过计算对象间距离来作为各个对象之间的非相似性指标，避免了对象之间的匹配问题，在这里距离一般使用欧氏距离或曼哈顿距离。

###  欧式距离

```math
d(x,y)= 
\sqrt[]{\sum_{k=1}^{n}(x_k-y_k)^2}
```

###  曼哈顿距离


```math
d(x,y)= 
\sqrt[]{\sum_{k=1}^{n}
\left|x_k-y_k\right|}
```


其中
```math
x(x_1,x_2,x_3,x_4,x_5,...,x_k)

y(y_1,y_2,y_3,y_4,y_5,...,y_k)
```
都是k维空间中的点。 
<br>
<br>
    
### KNN算法步骤
<br>

1. 计算测试数据与各个训练数据之间的距离
2. 按照距离的递增关系进行排序
3. 选取距离最小的K个点
4. 确定前K个点所在类别的出现频率
5. 返回前K个点中出现频率最高的类别作为
测试数据的预测分类

<br>

## 利用PYTHON实现KNN算法
### 代码数据准备

```
"""
样本集：
X = [[1, 1], [1, 1.5], [2, 2], [4, 3], [4, 4]]
Y = ['A', 'A', 'A', 'B', 'B']

测试样本:
设k=3时；t=[3,2] 属于A,B哪个类型

算法要求：
矩阵减法，按行(列)求和，开方
"""
```
### PYTHON代码


```
import numpy as np
import operator


def knn(x, y, k, t):
    # 计算test和x样本中所有元素的距离
    distance = np.sum((t-x)**2, axis=1)**0.5
    # 距离排序
    paiindex = np.argsort(distance)
    # 前k个最短距离的类别个数分别是多少，并置入一个集合
    classcount = {}
    for i in range(k):
        label = y[paiindex[i]]
        classcount[label] = classcount.get(label, 0) + 1
    # 排序并且返回类别标号
    return sorted(classcount.items(), key=operator.itemgetter(1), reverse=True)[0][0]


if __name__ == "__main__":
    X = np.array([[1, 1], [1, 1.5], [2, 2], [4, 3], [4, 4]])
    Y = np.array(['A', 'A', 'A', 'B', 'B'])
    T = [2, 3]
    predict = knn(X, Y, 3, T)
    print(predict)
    
# The output will be as follows: 

A
```


