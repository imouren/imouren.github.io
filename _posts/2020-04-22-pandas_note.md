---
layout: post
title:  "pandas常用操作"
date:   2020-02-22 10:15:31
categories:  pandas
tags:  pandas
---

* content
{:toc}


### 删除行

通过行名称删除：

```
df = df.drop(['1', '2'])           # 不指定axis默认为0
df.drop(['1', '3'], inplace=True)
```

通过行号删除：

```
df.drop(df.index[0], inplace=True)       # 删除第1行
df.drop(df.index[0:3], inplace=True)     # 删除前3行
df.drop(df.index[[0, 2]], inplace=True)  # 删除第1第3行
```

随机删除

```
import pandas as pd
import numpy as np
np.random.seed(10)

remove_n = 1
df = pd.DataFrame({"a":[1,2,3,4], "b":[5,6,7,8]})
drop_indices = np.random.choice(df.index, remove_n, replace=False)
df_subset = df.drop(drop_indices)
```


### 删除列

直接删除

```
del df['A']  # 删除A列，会就地修改

```

通过列名称删除：

```
df = df.drop(['B', 'C'], axis=1)               # drop不会就地修改，创建副本返回
df.drop(['B', 'C'], axis=1, inplace=True)      # inplace=True会就地修改
```

使用列数删除，传入参数是int，列表，者切片

```
df.drop(df.columns[0], axis=1, inplace=True)       # 删除第1列
df.drop(df.columns[0:3], axis=1, inplace=True)     # 删除前3列
df.drop(df.columns[[0, 2]], axis=1, inplace=True)  # 删除第1第3列
```


### 删除为空的

df = df.dropna()  # 删除所有为空的行

df = df.dropna(axis=1)  # 删除所有为空的列

df = df4.dropna(subset=["age", "sex"])  # 可以通过subset参数来删除在age和sex中含有空数据的全部行


### merge多个dataframe

df_final = reduce(lambda left, right: pd.merge(left, right, on='hour'), dfs)


### 根据某列排序

vv_data_df.sort_values("play_count", ascending=False, inplace=True)


### 强制转换

df.vid = pd.to_numeric(df.vid, errors='coerce').fillna(0).astype(np.int32)


### 遍历dataframe

iterrows(): 按行遍历，将DataFrame的每一行迭代为(index, Series)对，可以通过row[name]对元素进行访问。
itertuples(): 按行遍历，将DataFrame的每一行迭代为元祖，可以通过row[name]对元素进行访问，比iterrows()效率高。
iteritems():按列遍历，将DataFrame的每一列迭代为(列名, Series)对，可以通过row[index]对元素进行访问

```
for index, row in df.iterrows():
    print(index) # 输出每行的索引值


for row in df.itertuples():
    print(row["name"])


for index, row in df.iteritems():
    print(index) # 输出列名
```

### 重命名列

```
a.rename(columns={'A':'a', 'B':'b', 'C':'c'}, inplace = True)

df.columns=columns

在spark中
spark_df = spark_df.toDF(*columns)

```


### 读取多个文件并连接成一个dataframe

```
df_list = []
for r in glob.glob("data/*.csv"):
    adf = pd.read_csv(r, sep="\t", header=None, names=columns)
    df_list.append(adf)
df = pd.concat(df_list, ignore_index=True)
df.head()

```


### apply 输出两列

```
def f(x):
    return x*2, x*10


df["x1"], df["x2"] = zip(*df["a"].apply(f))
```

### apply处理多列生成一列

```
def my_test(a, b):
    return a + b

df['Value'] = df.apply(lambda row: my_test(row['a'], row['c']), axis=1)
```


## 计算cosine相似度

```
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

A =  np.array([[0, 1, 0, 0, 1], [0, 0, 1, 1, 1],[1, 1, 0, 1, 0]])
A_sparse = sparse.csr_matrix(A)

similarities = cosine_similarity(A_sparse)
print('pairwise dense output:\n {}\n'.format(similarities))

#also can output sparse matrices
similarities_sparse = cosine_similarity(A_sparse,dense_output=False)
print('pairwise sparse output:\n {}\n'.format(similarities_sparse))
```


## numpy 数字 apply处理

```
import numpy as np
x = np.array([1, 2, 3, 4, 5])
f = lambda x: x ** 2
squares = f(x)
```


## 一行变多行

参考 https://www.cnblogs.com/traditional/p/11967360.html


```

print(df)
"""
           姓名        生日                           声优
0    琪亚娜·卡斯兰娜   12月7日                      陶典,钉宫理惠
1   布洛妮娅·扎伊切克  8月18日            TetraCalyx,Hanser,阿澄佳奈,花泽香菜
2  德丽莎·阿波卡利斯   3月28日                     花玲,田村由香里
"""

df = df.set_index(["姓名", "生日"])["声优"].str.split(",", expand=True)\
    .stack().reset_index(drop=True, level=-1).reset_index().rename(columns={0: "声优"})
print(df)
"""
           姓名       生日           声优
0    琪亚娜·卡斯兰娜  12月7日          陶典
1    琪亚娜·卡斯兰娜  12月7日        钉宫理惠
2   布洛妮娅·扎伊切克  8月18日      TetraCalyx
3   布洛妮娅·扎伊切克  8月18日        Hanser
4   布洛妮娅·扎伊切克  8月18日        阿澄佳奈
5   布洛妮娅·扎伊切克  8月18日        花泽香菜
6  德丽莎·阿波卡利斯   3月28日         花玲
7  德丽莎·阿波卡利斯   3月28日       田村由香里
"""

```

## 两列统计

```
df = df.groupby(['Col1', 'Col2']).size().reset_index(name='Freq')
```
