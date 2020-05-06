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


