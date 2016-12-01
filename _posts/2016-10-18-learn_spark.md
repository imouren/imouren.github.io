---
layout: post
title:  "spark快速大数据分析"
date:   2016-10-19 09:05:31
categories: spark
tags: spark
---

* content
{:toc}


## 入门

去 http://spark.apache.org/downloads.html 下载相应版本的spark

```python
tar -xf spark-2.0.2-bin-hadoop2.4.tgz

cp spark-2.0.2-bin-hadoop2.4 /usr/local/spark

# 启动 spark shell
./bin/pyspark       # python
./bin/spark-shell   # scala
```

调整日志级别

```python

cp conf/log4j.properties.template conf/log4j.properties

vim log4j.properties

log4j.rootCategory=INFO, console
改为
log4j.rootCategory=WARN, console

```

使用IPYTHON

```python

PYSPARK_DRIVER_PYTHON=ipython  ./bin/pyspark

PYSPARK_DRIVER_PYTHON=ipython PYSPARK_DRIVER_PYTHON_OPTS="notebook --pylab  inline" ./bin/pyspark

```

简单的行数统计

```python

In [1]: lines = sc.textFile("/tmp/themes")

In [2]: lines.count()
Out[2]: 127145                                                                  

In [3]: lines.first()
Out[3]: u'video_id:1110079#theme_tag:\u5a31\u4e50'

```

构建独立应用

```python
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("mr_spark_playtm")
sc = SparkContext(conf=conf)

# appName参数是在集群UI上显示的你的应用的名称。

# master是一个Spark、Mesos或YARN集群的URL,如果你在本地运行那么这个参数应该是特殊的”local”字符串。

# 在实际使用中，当你在集群中运行你的程序，你一般不会把master参数写死在代码中，而是通过用spark-submit运行程序来获得这个参数
```

## RDD编程

Spark中的RDD是一个不可变的分布式对象集合。每个RDD都被分为多个分区，这些分区运行在集群中的不同节点上。

### 创建RDD的方式

```python
# 读取外部数据
lines = sc.textFile("/path/to/README.md")

# 在驱动程序中对集合进行并行化，多在学习spark时用，开发原型用的少，因为需要把整个数据集放到一台机器的内存中
lines = sc.parallelize(["hello", "mouren"])
```

###　RDD操作：转化操作和行动操作。

* 转化操作是返回一个新的RDD的操作，如map()和filter()
* 行动操作则是向驱动程序返回结果或者是把结果写入到外部操作系统中，会触发实际的计算。如count()和first()
* 转化操作是惰性的，只有在行动操作中用到这些RDD的时候，才会被计算。
* 行动操作有个collect()函数，可以获取整个RDD中的数据，只有你的整个数据集能在单台机器内存放得下的时候才可以使用它，不能用在大规模数据集上。

### 向spark传递函数

```python
# 可以使用lambda函数、传递顶层函数或者局部函数
word = rdd.filter(lambda s: "error" in s)

def containsError(s):
    return "error" in s
word = rdd.filter(containsError)

# 注意！
# python会把函数所在的对象也序列化传递出去。

# 传递一个带字段引用的函数（别这么做！）
class SearchFunctions(object):
    def __init__(self, query):
        self.query = query
    def isMatch(self, s):
        return self.query in s
    def getMatchesFunctionReference(self, rdd):
        # 问题： 在"self.isMatch"中引用了整个self
        return rdd.filter(self.isMatch)
    def getMatchesMemberReference(self, rdd):
        # 问题： 在"self.query"中引用了整个self
        return rdd.filter(lambda x: self.query in x)

# 传递不带字段引用的 Python 函数
class WordFunctions(object):
    ...
    def getMatchesNoReference(self, rdd):
        # 安全：只把需要的字段提取到局部变量中
        query = self.query
        return rdd.filter(lambda x: query in x)
```

###　常见的转化操作和行动操作

#### 转换操作

map() 和 filter()

```python

In [13]: nums = sc.parallelize([1,2,3,4,5])

In [16]: squared = nums.map(lambda x: x**2).collect()

In [17]: squared
Out[17]: [1, 4, 9, 16, 25]

In [18]: evens = nums.filter(lambda x: x%2 == 0).collect()

In [19]: evens
Out[19]: [2, 4]

```

sample(withReplacement, fraction, [seed]) 对RDD取样

```python
In [41]: nums = sc.parallelize(range(20))

In [42]: nums.sample(False, 0.5).collect()
Out[42]: [0, 2, 4, 6, 7, 8, 9, 10, 13, 14, 19]

In [43]: nums.sample(True, 0.5).collect()
Out[43]: [0, 2, 7, 9, 9, 10, 10, 11, 16, 17, 18, 18]

```

flatMap() 的函数被分别应用到了输入 RDD 的每个元素上。不
过返回的不是一个元素，而是一个返回值序列的迭代器。

```python
In [20]: lines = sc.parallelize(["hello mr!", "hi mouren"])

In [21]: words = lines.flatMap(lambda line: line.split()).collect()

In [22]: words
Out[22]: ['hello', 'mr!', 'hi', 'mouren']

# 与map的区别
In [23]: words_map = lines.map(lambda line: line.split()).collect()

In [24]: words_map
Out[24]: [['hello', 'mr!'], ['hi', 'mouren']]
```

distinct() 可以完成去重操作，不过开销很大，需要将所有的数据通过网络进行混洗（shuffle）

union(other) 返回包含两个RDD中所有元素的RDD，输入RDD有重复数据，结果也包含重复数据

intersection(other) 返回两个RDD中都有的元素，会移除重复元素。性能差，需要shuffle

subtract(other) 返回一个由只存在第一个RDD中而不存在第二个RDD中的所有元素组成的RDD。性能差，需要shuffle

cartesian(other) 计算两个RDD的笛卡尔积
，求大规模RDD的笛卡尔积开销巨大。

```python

In [25]: rdd1 = sc.parallelize(["coffee", "coffee", "panda", "monkey", "tea"])

In [26]: rdd2 = sc.parallelize(["coffee", "monkey", "kitty"])

In [28]: rdd1.distinct().collect()
Out[28]: ['tea', 'panda', 'monkey', 'coffee']

In [30]: rdd1.union(rdd2).collect()
Out[30]: ['coffee', 'coffee', 'panda', 'monkey', 'tea', 'coffee', 'monkey', 'kitty']

In [31]: rdd1.intersection(rdd2).collect()
Out[31]: ['coffee', 'monkey']

In [33]: rdd1.subtract(rdd2).collect()
Out[33]: ['tea', 'panda']

In [34]: rdd1.cartesian(rdd2).collect()
Out[34]: 
[('coffee', 'coffee'),
 ('coffee', 'monkey'),
 ('coffee', 'kitty'),
 ('coffee', 'coffee'),
 ('coffee', 'monkey'),
 ('coffee', 'kitty'),
 ('panda', 'coffee'),
 ('panda', 'monkey'),
 ('panda', 'kitty'),
 ('monkey', 'coffee'),
 ('tea', 'coffee'),
 ('monkey', 'monkey'),
 ('tea', 'monkey'),
 ('monkey', 'kitty'),
 ('tea', 'kitty')]

```

#### 行为操作

reduce() 和 fold()

fold() 类似于reduce(), 再加上一个初始值作为每个分区第一次调用的结果。
初始值进行多次计算，不会改变结果（+对应0，*对应1，拼接操作对应[]） 

fold() 和 reduce() 都要求函数的返回值类型需要和我们所操作的 RDD 中的元素类型相
同

```python
In [72]: from operator import add

In [73]: sc.parallelize(range(20)).reduce(add)
Out[73]: 190

In [74]: sc.parallelize(range(20)).fold(0, add)
Out[74]: 190

# 这里增加了5，是因为有4个分区，全部分区相加的时候（combine），再加一个初始值
In [75]: sc.parallelize(range(20)).fold(1, add)
Out[75]: 195

In [76]: sc.parallelize(range(20)).glom().collect()
Out[76]: [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]

# 直接查看分区数
In [131]: sc.parallelize(range(20)).getNumPartitions()
Out[131]: 4

```

aggregate() 函数的返回值类型不必与所操作的 RDD 类型相同，需要提供返回值类型的初始化值，中间是处理函数，每个节点都需要在本地进行combine，还需要提供一个函数进行本地的合并操作。

def aggregate[U](zeroValue: U)(seqOp: (U, T) : U, combOp: (U, U) : U)(implicit arg0: ClassTag[U]): U

aggregate用户聚合RDD中的元素，先使用seqOp将RDD中每个分区中的T类型元素聚合成U类型，再使用combOp将之前每个分区聚合后的U类型聚合成U类型，特别注意seqOp和combOp都会使用zeroValue的值，zeroValue的类型为U

```python
# 求单词的平均长度
In [78]: x = sc.parallelize(["hello mouren", "hi mouren", "hello word"]).flatMap(lambda x: x.split())

In [83]: res = x.aggregate((0, 0), lambda x, y: (x[0]+len(y), x[1]+1), lambda x, y: (x[0]+y[0], x[1]+y[1]))
Out[83]: (28, 6)

In [85]: res[0]/float(res[1])
Out[85]: 4.666666666666667

```

### 将数据返回到驱动器中

collect() 将整个RDD内容返回

take(n) 返回RDD中的n个元素，并且尝试只访问尽量少的分区，返回顺序与你预期可能不一致

top(n) 按顺序返回前几个元素

takeSample(withReplacement, num, seed)  采样返回数据

foreach() 行动操作来对 RDD 中的每个元素进行操作，而不需要把 RDD 发回本地

countByValue() 返回一个从各值到值对应的计数的映射表

takeOrdered(num, key=None) 从 RDD 中按照提供的顺序返回最前面的 num 个元素

```python

In [87]: nums = sc.parallelize([1,2,3,3])

In [88]: nums.collect()
Out[88]: [1, 2, 3, 3]

In [89]: nums.count()
Out[89]: 4

In [92]: nums.countByValue()
Out[92]: defaultdict(int, {1: 1, 2: 1, 3: 2})

In [93]: nums.take(3)
Out[93]: [1, 2, 3]

In [94]: nums.top(3)
Out[94]: [3, 3, 2]

In [95]: nums.takeOrdered(3, key=lambda x: -x)
Out[95]: [3, 3, 2]

In [96]: nums.takeOrdered(3, key=lambda x: x)
Out[96]: [1, 2, 3]

In [97]: nums.takeSample(False, 3)
Out[97]: [3, 3, 2]

In [98]: nums.takeSample(False, 3)
Out[98]: [3, 1, 3]

In [99]: nums.takeSample(True, 3)
Out[99]: [2, 1, 1]

In [100]: nums.takeSample(True, 3)
Out[100]: [1, 3, 3]

In [103]: cnt = sc.accumulator(0)

In [107]: nums.foreach(lambda x: cnt.add(x))

In [109]: cnt.value
Out[109]: 9
```

### 在不同RDD类型间转换

有些函数只能用于特定类型的 RDD，比如 mean() 和 variance() 只能用在数值 RDD 上，
而 join() 只能用在键值对 RDD 上。

在 Python 中，所有的函数都实现在基本的
RDD 类中，但如果操作对应的 RDD 数据类型不正确，就会导致运行时错误。

### 持久化（缓存）

为了避免多次计算同一个 RDD，可以让 Spark 对数据进行持久化。

在 Python 中，我们会始终序列化要持久化存储的数据，所以持久化级别默认值就是
以序列化后的对象存储在 JVM 堆空间中。当我们把数据写到磁盘或者堆外存储上时，也
总是使用序列化后的数据。

org.apache.spark.storage.StorageLevel和pyspark.StorageLevel中的持久化级
别；如有必要，可以通过在存储级别的末尾加上“_2”来把持久化数据存为两份

`MEMORY_ONLY  MEMORY_ONLY_SER  MEMORY_AND_DISK MEMORY_AND_DISK_SER DISK_ONLY`

RDD 还有一个方法叫作 unpersist()，调用该方法可以手动把持久化的 RDD 从缓
存中移除

```python

In [111]: nums.persist()
Out[111]: ParallelCollectionRDD[94] at parallelize at PythonRDD.scala:475

In [112]: nums.is_cached
Out[112]: True

In [113]: nums.unpersist()
Out[113]: ParallelCollectionRDD[94] at parallelize at PythonRDD.scala:475

In [114]: nums.is_cached
Out[114]: False

```

## 键值对操作

Spark 为包含键值对类型的 RDD 提供了一些专有的操作。这些 RDD 被称为 pairRDD

### 创建pairRDD

当需要把一个普通的 RDD 转为 pair RDD 时，可以调用 map() 函数来实现，传递的函数需要返回键值对。

```python
# 使用第一个单词作为键创建出一个 pair RDD
In [124]: lines.collect()
Out[124]: ['hello mr!', 'hi mouren']

In [127]: lines.map(lambda x: (x.split(" ")[0], x)).collect()
Out[127]: [('hello', 'hello mr!'), ('hi', 'hi mouren')]

```

### Pair RDD的转化操作

Pair RDD 可以使用所有标准 RDD 上的可用的转化操作。

一些转换操作举例：

```python
rdd = sc.parallelize([(1,2), (3, 4), (3, 6)])

# 合并相同键的值
In [4]: rdd.reduceByKey(lambda x, y: x+y).collect()
Out[4]: [(1, 2), (3, 10)]

# 对具有相同键的值进行分组
In [8]: rdd.groupByKey().collect()
Out[8]: 
[(1, <pyspark.resultiterable.ResultIterable at 0x7f41b432ff50>),
 (3, <pyspark.resultiterable.ResultIterable at 0x7f41b432fd50>)]

In [9]: [(k, list(v)) for k, v in _]
Out[9]: [(1, [2]), (3, [4, 6])]

# 对 pair RDD 中的每个值应用一个函数而不改变键
In [10]: rdd.mapValues(lambda x: x+1).collect()
Out[10]: [(1, 3), (3, 5), (3, 7)]

# 对 pair RDD 中的每个值应用一个返回迭代器的函数，然后对返回的每个元素都生成一个对应原键的键值对记录。
In [12]: rdd.flatMapValues(lambda x: range(x)).collect()
Out[12]: 
[(1, 0),
 (1, 1),
 (3, 0),
 (3, 1),
 (3, 2),
 (3, 3),
 (3, 0),
 (3, 1),
 (3, 2),
 (3, 3),
 (3, 4),
 (3, 5)]


# keys values
In [15]: rdd.keys().collect()
Out[15]: [1, 3, 3]

In [16]: rdd.values().collect()
Out[16]: [2, 4, 6]

# sortedByKey((ascending=True, numPartitions=None, keyfunc=)
rdd.sortByKey(False).collect()
```

针对两个RDD的操作

```python

rdd = sc.parallelize([(1,2), (3, 4), (3, 6)])
other = sc.parallelize([(3, 9)])

# 删掉 RDD 中键与 other RDD 中的键相同的元素
In [23]: rdd.subtractByKey(other).collect()
Out[23]: [(1, 2)]

# 对两个 RDD 进行内连接
In [24]: rdd.join(other).collect()
Out[24]: [(3, (4, 9)), (3, (6, 9))])

# 左外连接 和 右外连接

In [26]: rdd.rightOuterJoin(other).collect()
Out[26]: [(3, (4, 9)), (3, (6, 9))]

In [27]: rdd.leftOuterJoin(other).collect()
Out[27]: [(1, (2, None)), (3, (4, 9)), (3, (6, 9))]

# 将两个 RDD 中拥有相同键的数据分组到一起

In [58]: rdd.cogroup(other).collect()
Out[58]: 
[(1,
  (<pyspark.resultiterable.ResultIterable at 0x7f41b42d1110>,
   <pyspark.resultiterable.ResultIterable at 0x7f41b42abc10>)),
 (3,
  (<pyspark.resultiterable.ResultIterable at 0x7f41b42abc90>,
   <pyspark.resultiterable.ResultIterable at 0x7f41b42abcd0>))]

{(1,([2],[])), (3,([4, 6],[9]))}

```
