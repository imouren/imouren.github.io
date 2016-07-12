---
layout: post
title:  "spark学习之icf"
date:   2016-06-19 10:35:31
categories: spark
tags: spark
---

* content
{:toc}


## 直接贴代码

```python

# -*- coding: utf-8 -*-
import sys
import os
from operator import add, itemgetter
import json
import math
from functools import partial
from datetime import datetime, timedelta

from pyspark import SparkContext, SparkConf

"""
日志路径: hdfs:///dw/logs/format/app_fbuffer/2016/03/03/
分隔符：\t
需要的数据索引： fudid:11 media_id:14 channel_id:15

./bin/spark-submit \
  --master yarn \
  --num-executors 30 \
  --executor-memory 8G \
  --executor-cores 4 \
  --driver-memory 1G \
  --conf spark.default.parallelism=500 \
  --conf spark.storage.memoryFraction=0.5 \
  --conf spark.shuffle.memoryFraction=0.5 \
  icfx.py
"""

os.environ["PYSPARK_PYTHON"]="/usr/local/Python-2.7.11/bin/python"

def json_output(kv):
    k, v = kv
    return "%s\t%s" % (json.dumps(k), json.dumps(v))

def _get_mid_count(line):
    y = line.split("\t")
    if y[14].isdigit() and int(y[14]) < 1000000:
        yield int(y[14]), 1

def _get_fudid_mid(line):
    y = line.split("\t")
    if y[14].isdigit() and int(y[14]) < 1000000:
        yield y[11], int(y[14])

def _get_mid_relate(mids):
    n = len(mids)
    for i in xrange(n):
        for j in xrange(i+1, n):
            if mids[i] > mids[j]:
                yield (mids[i], mids[j]), 1
            elif mids[i] < mids[j]:
                yield (mids[j], mids[i]), 1


def _get_mid_sim(pops, pairs):
    mid_tuple, num = pairs
    mid_i, mid_j = mid_tuple
    if mid_i in pops and mid_j in pops:
        sim = num / math.sqrt(pops[mid_i] * pops[mid_j])
        sim = round(sim, 8)
        yield mid_i, (mid_j, sim) 
        yield mid_j, (mid_i, sim)


def _get_mids_set(pairs):
    fudid, mids = pairs
    res = []
    for mid in set(mids):
        res.append(mid)
    if 2 < len(res) < 50:
        yield res


if __name__ == "__main__":
    conf = SparkConf().setAppName("mr_spark_icf")
    sc = SparkContext(conf=conf)
    base_path = "/dw/logs/format/app_fbuffer/{}/{:02}/{:02}/part*"
    paths = []
    yesterday = datetime.now() - timedelta(1)
    for i in range(1, 8):
        the_day = datetime.now() - timedelta(i)
        apath = base_path.format(the_day.year, the_day.month, the_day.day)
        paths.append(apath)
    path = ",".join(paths)
    #path = "/dw/logs/format/app_fbuffer/2016/07/03/part-r-00000"
    lines = sc.textFile(path).cache()

    # log
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)
    LOGGER.info("init log ...")

    # get mid pop dict
    mid_pop = lines.flatMap(_get_mid_count) \
                .countByKey()
    pops = sc.broadcast(mid_pop)

    LOGGER.info("mid_pop log ...")

    # fudid mid
    # fudid set(mids)
    pairs = lines.flatMap(_get_fudid_mid) \
                .groupByKey() \
                .flatMap(_get_mids_set)
    
    # (mid_i, mid_j) n
    pairs_counts = pairs.flatMap(_get_mid_relate) \
                                .reduceByKey(add)
    # mid_i (mid_j, sim)
    # mid [(mid,sim), (mid, sim)]
    mid_sims = pairs_counts.flatMap(partial(_get_mid_sim, pops.value)) \
                            .groupByKey() \
                            .mapValues(lambda x: sorted(list(x), key=itemgetter(1), reverse=True)[:100])

    # writ to hdfs
    mid_sims = mid_sims.map(json_output)
    output_path = "/test/spark/icf/res/{}/{:02}/{:02}".format(yesterday.year, yesterday.month, yesterday.day)
    mid_sims.saveAsTextFile(output_path)

    sc.stop()




```

## 问题和解决

如何提交到yarn中执行？

```
spark-submit --master yarn icf.py
```

异常解决

Exception: Python in worker has different version 2.6 than that in driver 2.7, PySpark cannot run with different minor versions

```python
# 统一只用python 2.7
os.environ["PYSPARK_PYTHON"]="/usr/local/Python-2.7.11/bin/python"

```

结束掉yarn上的任务

```python

copy past the application Id from the spark scheduler, for instance application_1428487296152_25597

connect to the server that have launch the job

yarn application -kill application_1428487296152_25597

```

如何使用`yield`？

```python

#可以用 flatMap 代替 map 来实现
>>> words = sc.textFile("README.md")
>>> def mapper(line):
...     for word in line.split():
...         yield (word, 1)
...
>>> words.flatMap(mapper).take(4)
[(u'#', 1), (u'Apache', 1), (u'Spark', 1), (u'Lightning-Fast', 1)]
>>> counts = words.flatMap(mapper).reduceByKey(lambda x, y: x + y)
>>> counts.take(5)
[(u'all', 1), (u'help', 1), (u'webpage', 1), (u'when', 1), (u'Hadoop', 12)]

```

如何读取多个输入文件？

```python

path1 = "/dw/logs/format/app_fbuffer/2016/07/03/part*"
path2 = "/dw/logs/format/app_fbuffer/2016/07/03/part*"
paths = [path1, path2]
lines = sc.textFile(",".join(paths))

```

如何使用日志？

```python

# log
log4jLogger = sc._jvm.org.apache.log4j
LOGGER = log4jLogger.LogManager.getLogger(__name__)
LOGGER.info("init log ...")

```
