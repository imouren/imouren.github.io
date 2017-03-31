---
layout: post
title:  "spark快速大数据分析——上"
date:   2016-10-18 09:05:31
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

### RDD操作：转化操作和行动操作。

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

### 常见的转化操作和行动操作

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

#### 聚合操作

当数据集以键值对形式组织的时候，聚合具有相同键的元素进行一些统计是很常见的操作。

```python
# 使用 reduceByKey() 和 mapValues() 计算每个键对应的平均值
In [4]: rdd = sc.parallelize([("panda", 0), ("pink", 3), ("pirate", 3), ("panda", 1), ("pink", 4)])

In [5]: rdd.mapValues(lambda x: (x, 1)).reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])).collect()
Out[5]: [('pink', (7, 2)), ('panda', (1, 2)), ('pirate', (3, 1))]

```

调用 reduceByKey() 和 foldByKey() 会在为每个键计算全局的总结之前先自动在每台机器上进行本地合并, 用户不需要指定合并器。

更泛化的combineByKey() 接口可以让你自定义合并的行为。

* combineByKey() 会遍历分区中的所有元素，因此每个元素的键要么还没有遇到过，要么就
  和之前的某个元素的键相同。

* 如果这是一个新的元素， combineByKey() 会使用一个叫作 createCombiner() 的函数来创建那个键对应的累加器的初始值。需要注意的是，这一过程会在每个分区中第一次出现各个键时发生，而不是在整个 RDD 中第一次出现一个键时发生。

* 如果这是一个在处理当前分区之前已经遇到的键，它会使用 mergeValue() 方法将该键的累
  加器对应的当前值与这个新的值进行合并。

* 由于每个分区都是独立处理的，因此对于同一个键可以有多个累加器。如果有两个或者更多的分区都有对应同一个键的累加器，就需要使用用户提供的 mergeCombiners() 方法将各个分区的结果进行合并。

```python
In [7]: rdd.combineByKey(
   ...:     lambda x: (x, 1),
   ...:     lambda x, y: (x[0] + y, x[1] + 1),
   ...:     lambda x, y: (x[0]+y[0], x[1]+y[1])
   ...: ).collect()
Out[7]: [('pink', (7, 2)), ('panda', (1, 2)), ('pirate', (3, 1))]
```

每个 RDD 都有固定数目的分区，分区数决定了在 RDD 上执行操作时的并行度。

在执行聚合或分组操作时，可以要求 Spark 使用给定的分区数。

```python

In [8]: data = [("a", 3), ("b", 4), ("a", 1)]

# 系统默认计算分区数
In [9]: sc.parallelize(data).reduceByKey(lambda x, y: x+y).collect()
Out[9]: [('a', 4), ('b', 4)]

# 自己指定分区数
In [10]: sc.parallelize(data).reduceByKey(lambda x, y: x+y, 10).collect()
Out[10]: [('b', 4), ('a', 4)]

```

在除分组操作和聚合操作之外的操作中也能改变 RDD 的分区。对于这样的情况， Spark 提供了 repartition() 函数。

它会把数据通过网络进行混洗，并创建出新的分区集合。切记，对数据进行重新分区是代价相对比较大的操作。 

Spark 中也有一个优化版的repartition()，叫作 coalesce()。

你可以使用 Python 中的 rdd.getNumPartitions 查看 RDD 的分区数，并确保调用 coalesce() 时将 RDD合并到比现在的分区数更少的分区中。

```python
In [14]: rdd.collect()
Out[14]: [('panda', 0), ('pink', 3), ('pirate', 3), ('panda', 1), ('pink', 4)]

In [15]: rdd.getNumPartitions()
Out[15]: 4

In [18]: rddx = rdd.coalesce(2)

In [19]: rddx.getNumPartitions()
Out[19]: 2

```

#### 数据分组

groupByKey() 就会使用 RDD 中的键来对数据进行分组。对于一个由类型 K 的键和类型 V 的值组成的 RDD，所得到的结果 RDD 类型会是[K, Iterable[V]]

如果你发现自己写出了先使用 groupByKey() 然后再对值使用 reduce() 或者fold() 的代码，
你很有可能可以通过使用一种根据键进行聚合的函数来更高效地实现同样的效果

对两个键的类型均为 K 而值的类型分别为 V 和 W 的 RDD 进行cogroup() 时，
得到的结果 RDD 类型为 [(K, (Iterable[V], Iterable[W]))]


### Pair RDD的行动操作

```python
In [32]: rdd = sc.parallelize([(1, 2), (3, 4), (3, 6)])

# 对每个键对应的元素分别计数
In [33]: rdd.countByKey()
Out[33]: defaultdict(int, {1: 1, 3: 2})

In [34]: rdd.collectAsMap()
Out[34]: {1: 2, 3: 6}

# 返回给定键对应的所有值
In [35]: rdd.lookup(1)
Out[35]: [2]

In [36]: rdd.lookup(2)
Out[36]: []

In [37]: rdd.lookup(3)
Out[37]: [4, 6]

```

### 数据分区

分区并不是对所有应用都有好处的——比如，如果给定RDD 只需要被扫描一次，我们完全没有必要对其预先进行分区处理。

只有当数据集多次在诸如连接这种基于键的操作中使用时，分区才会有帮助。

```python

In [43]: rdd.getNumPartitions()
Out[43]: 4

In [45]: rdd100=rdd.partitionBy(100)

In [46]: rdd100.getNumPartitions()
Out[46]: 100

```

Spark 的许多操作都引入了将数据根据键跨节点进行混洗的过程。所有这些操作都会从数据分区中获益。

对于二元操作，输出数据的分区方式取决于父 RDD 的分区方式。 

python 自定义分区

```python
import urlparse

def hash_domain(url):
    return hash(urlparse.urlparse(url).netloc)

rdd.partitionBy(20, hash_domain) # 创建20个分区

```

## Spark编程进阶

两种类型的共享变量：

* 累加器 (accumulator)

* 广播变量 (broadcast variable)

### 累加器

累加器，提供了将工作节点中的值聚合到驱动器程序中的简单语法，常见用途是在调试时对作业执行过程中的事件进行计数。

#### 用法

1. 通过在驱动器中调用 SparkContext.accumulator(initialValue) 方法，创建出存有初始值的累加器。

2. Spark 闭包里的执行器代码可以使用累加器的 += 方法增加累加器的值。

3. 驱动器程序可以调用累加器的 value 属性来访问累加器的值。

工作节点上的任务不能访问累加器的值。从这些任务的角度来看，累加器是一个只写变量。

```python
file = sc.textFile(inputFile)
# 创建Accumulator[Int]并初始化为0
blankLines = sc.accumulator(0)

def extractCallSigns(line):
    global blankLines # 访问全局变量
    if (line == ""):
    blankLines += 1
    return line.split(" ")

callSigns = file.flatMap(extractCallSigns)
callSigns.saveAsTextFile(outputDir + "/callsigns")

print "Blank lines: %d" % blankLines.value
```

#### 累加器的容错

Spark 会自动重新执行失败的或较慢的任务来应对有错误的或者比较慢的机器，因此最终结果就是同一个函数可能对同一个数据运行了多次。

对于要在行动操作中使用的累加器， Spark只会把每个任务对各累加器的修改应用一次。

因此，如果想要一个无论在失败还是重复计算时都绝对可靠的累加器，我们必须把它放在 foreach() 这样的行动操作中

在转化操作中，无法保证，累加器通常只用于调试目的。


### 广播变量

广播变量，可以让程序高效地向所有工作节点发送一个较大的只读值，以供一个或多个 Spark 操作使用。

Spark 会自动把闭包中所有引用到的变量发送到工作节点上。虽然这很方便，但也很低效。

原因有二：首先，默认的任务发射机制是专门为小任务进行优化的；其次，事实上你可能会在多个并行操作中使用同一个变量，但是 Spark 会为每个操作分别发送。

广播变量使用：

(1) 通过对一个类型 T 的对象调用 SparkContext.broadcast 创建出一个 Broadcast[T] 对象。任何可序列化的类型都可以这么实现。
(2) 通过 value 属性访问该对象的值。
(3) 变量只会被发到各个节点一次，应作为只读值处理（修改这个值不会影响到别的节点），使用的是
一种高效的类似 BitTorrent 的通信机制。

广播变量的优化：

当广播一个比较大的值时，选择既快又好的序列化格式是很重要的，

Spark的 Scala 和 Java API 中默认使用的序列化库为 Java 序列化库，

因此它对于除基本类型的数组以外的任何对象都比较低效。

你可以使用 spark.serializer 属性选择另一个序列化库来优化序列化过程（使用 Kryo 这种更快的序列化库），也可以为你的数据类型实现自己的序列化方式（对 Java 对象使用 java.io.Externalizable 接口实现序列化，或使用 reduce() 方法为 Python 的 pickle 库定义自定义的序列化）。

### 基于分区进行操作

基于分区对数据进行操作可以让我们避免为每个数据元素进行重复的配置工作。

诸如打开数据库连接或创建随机数生成器等操作，都是我们应当尽量避免为每个元素都配置一次的工作。 

Spark 提供基于分区 的 map 和 foreach，让你的部分代码只对 RDD 的每个分区运行一次，这样可以帮助降低这些操作的代价。

有一个在线的业余电台呼号数据库，可以用这个数据库查询日志中记录过的联系人呼号列表。
通过使用基于分区的操作，可以在每个分区内共享一个数据库连接池，来避免建立太多连接，同时还可以重用 JSON 解析器。

```python

def processCallSigns(signs):
    """使用连接池查询呼号"""
    # 创建一个连接池
    http = urllib3.PoolManager()
    # 与每条呼号记录相关联的URL
    urls = map(lambda x: "http://73s.com/qsos/%s.json" % x, signs)
    # 创建请求（非阻塞）
    requests = map(lambda x: (x, http.request('GET', x)), urls)
    # 获取结果
    result = map(lambda x: (x[0], json.loads(x[1].data)), requests)
    # 删除空的结果并返回
    return filter(lambda x: x[1] is not None, result)

def fetchCallSigns(input):
    """获取呼号"""
    return input.mapPartitions(lambda callSigns : processCallSigns(callSigns))

contactsContactList = fetchCallSigns(validSigns)

```

使用 mapPartitions 函数获得输入 RDD 的每个分区中的元素迭代器，而需要返回的是执行结果的序列的迭代器。

类似的还有 mapPartitionsWithIndex()  foreachPartitions()

在 Python 中不使用 mapPartitions() 求平均值

```python

def combineCtrs(c1, c2):
    return (c1[0] + c2[0], c1[1] + c2[1])

def basicAvg(nums):
    """计算平均值"""
    return nums.map(lambda num: (num, 1)).reduce(combineCtrs)
```

在 Python 中使用 mapPartitions() 求平均值

```python
def partitionCtr(nums):
    """计算分区的sumCounter"""
    sumCount = [0, 0]
    for num in nums:
        sumCount[0] += num
        sumCount[1] += 1
    return [sumCount]

def fastAvg(nums):
    """计算平均值"""
    sumCount = nums.mapPartitions(partitionCtr).reduce(combineCtrs)
    return sumCount[0] / float(sumCount[1])

```


### 数值RDD的操作

Spark 的数值操作是通过流式算法实现的，允许以每次一个元素的方式构建出模型。

这些统计数据都会在调用 stats() 时通过一次遍历数据计算出来，并以 StatsCounter 对象返回。

StatsCounter中可用的汇总统计数据:

```python
count() RDD 中的元素个数
mean() 元素的平均值
sum() 总和
max() 最大值
min() 最小值
variance() 元素的方差
sampleVariance() 从采样中计算出的方差
stdev() 标准差
sampleStdev() 采样的标准差
```

例如：

```python
In [27]: nums = sc.parallelize(range(10))

In [28]: nums.mean(), nums.sum()
Out[28]: (4.5, 45)

In [30]: nums.stats()
Out[30]: (count: 10, mean: 4.5, stdev: 2.87228132327, max: 9.0, min: 0.0)

In [31]: st = nums.stats()

In [33]: st.mean()
Out[33]: 4.5

```


## 在集群上运行Spark

### Spark运行时架构

在分布式环境下， Spark 集群采用的是主 / 从结构。
在一个 Spark 集群中，有一个节点负责中央协调，调度各个分布式工作节点。
这个中央协调节点被称为驱动器（Driver）节点，与之对应的工作节点被称为执行器（executor）节点。
驱动器节点可以和大量的执行器节点进行通信，它们也都作为独立的 Java 进程运行。
驱动器节点和所有的执行器节点一起被称为一个 Spark 应用 （application）

分布式 Spark 应用中的组件
![lspark_model.png](/files/lspark_model.png)

Spark 应用通过一个叫作集群管理器（Cluster Manager）的外部服务在集群中的机器上启动。 
Spark 自带的集群管理器被称为独立集群管理器。 
Spark 也能运行在 Hadoop YARN 和Apache Mesos 这两大开源集群管理器上。

**驱动器节点**

spark 驱动器是执行你程序中的main()方法的进程。它执行用户编写的用来创建SparkContent、创建RDD，以及进行RDD的转换操作和行动操作代码。启动spark shell时，就已经启动了spark驱动程序。

驱动器一旦终止，spark应用就结束了。

主要两个职责：

* 把用户程序转为任务

  驱动器负责把用户程序转为多个物理执行单元，这些单元也称为任务（task）

  spark程序隐式的创建出一个由操作组成的逻辑上的 有向无环图（DAG），当驱动程序运行时，它会把这个逻辑图转为物理执行计划。

  spark会对逻辑执行计划进行优化，将多个操作合并到一个步骤中。这样逻辑计划转为一系列步骤（stage），每个步骤由多个任务组成。任务是spark中最小的工作单元。

* 为执行器节点调度任务

  执行器进程启动后，会想驱动器进程注册自己。驱动器程序始终对应用中所有执行器节点有完整记录。每个执行器节点代表一个能够处理任务和存储RDD数据的进行。

  驱动器会根据当前的执行器节点集合，尝试把所有任务基于数据所在的位置分配给合适的执行器。

  任务执行时，执行器进程会把缓存数据存储起来，驱动器同样会跟踪这些缓存数据位置，并合理调度。

  驱动器程序会将spark应用运行时信息通过网页呈现出来，默认在端口4040上。



**执行器节点**

spark执行器节点是一种工作进程，负责spark作业中运行任务，任务间相互独立。

主要负责：

* 负责运行spark应用的任务，并将结果返回给驱动器程序
* 通过自身的块管理器（block manager）为用户程序要求缓存的RDD提供内容存储。



本地模式下，spark驱动器程序和各个执行器程序在同一个java进程中。这是一个特例，执行器通常都运行在专用的进程中。



**集群管理器**

spark依赖集群管理器来启动执行器节点，而在某些特殊情况下也依赖集群管理器启动驱动器节点。

集群管理器是spark中可插拔组件。也可以运行其他外部集群管理器，如yarn 和 mesos



**启动一个程序**

spark提供统一的脚本`spark-submit`将你的应用提交到需要的集群管理器上。

**小结**

* 用户通过spark-submit脚本提交应用
* spark-submit脚本启动驱动程序，调用用户定义的main()方法
* 驱动器程序与集群管理器通信，申请资源以启动执行器节点
* 集群管理器为驱动程序启动执行器节点
* 驱动器进程执行用户应用中的操作。根据程序定义将对RDD的转换操作和行动操作，驱动器节点把工作以任务的形式发送到执行器程序中。
* 任务在执行器中进行计算并保存结果
* 如果驱动器程序main()方法退出，或者调用了SparkContent.stop()，驱动器程序会终止执行进行，并通过集群管理器释放资源。



### 使用spark-submit部署应用

一般格式为：

```python
bin/spark-submit [options] <app jar| python file> [app options]
```

![spark_submit.png](/files/spark_submit.png)



--master 标记指定要连接的集群 URL

![spark_submit_master.png](/files/spark_submit_master.png)



### 打包代码与依赖

pyspark本身就运行在python环境中了，如果需要第三方依赖，可以安装依赖到集群所有机器上。

也可以直接用 --py-Files参数提交独立库，他们也会被添加到python解释器的路径中。



### spark应用内与应用间调度

两个不同的用户，提交了希望使用整个集群所有资源的应用，该如何处理？

spark有一系列调度策略来保证资源不会被过渡使用，还允许工作负载设置优先级。

在调度多用户集群时，spark主要依赖集群管理器来在spark应用间共享资源。



### 集群管理器

spark可以运行在各种集群管理器上，通过集群管理器访问集群中的机器。

如果想在一堆机器上运行spark，自带的独立模式是部署该集群最简单的方法。

如果需要与别的分布式应用共享集群，可以运行在 hadoop YARN 与 Apache Mesos上
