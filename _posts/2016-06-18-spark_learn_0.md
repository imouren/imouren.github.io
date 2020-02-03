---
layout: post
title:  "spark学习之wordcount"
date:   2016-06-18 10:35:31
categories: spark
tags: spark
---

* content
{:toc}


## 直接贴代码

```python

# -*- coding: utf-8 -*-
import sys
from operator import add
import json

from pyspark import SparkContext, SparkConf

def json_output(kv):
    k, v = kv
    return "%s\t%s" % (json.dumps(k), json.dumps(v))


if __name__ == "__main__":
    conf = SparkConf().setAppName("mr_wordcount")
    sc = SparkContext(conf=conf)
    lines = sc.textFile("/test/input/1.txt")
    counts = lines.flatMap(lambda x: x.split()) \
                  .map(lambda x: (x, 1)) \
                  .reduceByKey(add)
    counts = counts.sortBy(lambda x:x[1], False)
    counts = counts.map(json_output)
    counts.saveAsTextFile("/test/output/spark_wordcount")

    sc.stop()

```

## 问题和解决

如何进行排序，按照单词数量大小？

```
counts = counts.sortBy(lambda x:x[1], False)
```

如何输出到HDFS，并使用一个格式？

```python

# 这里 key 和 value 都是josn格式  与mrjob中保持一致
counts = counts.map(json_output)
counts.saveAsTextFile("/test/output/spark_wordcount")

```

如何读取本地文件？

```python

lines = sc.textFile("file:///test/input/1.txt")

```

打包python环境（未验证）

```
打包本地python环境，一定要进入目录里面


cd path_to_python

zip -r path_to_pythonzip/python_user.zip ./*

hadoop fs -put python_user.zip /somepath/python_user.zip

拷贝一份spark配置文件
cd /usr/local/spark/conf
cp spark-defaults.conf spark-user.conf

添加一行配置，一定有#python
spark.yarn.dist.archives hdfs:///somepath/python_user.zip#python

提交的时候指定配置文件就可以

spark-submit --master yarn \
--properties-file /usr/local/spark/conf/spark-user.conf \
cang_theme_program_spark.py
```


结束任务

```
yarn application -kill application_1450259063324_0001
```