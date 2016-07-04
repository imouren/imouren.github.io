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

groupByKey().mapValues
"""

os.environ["PYSPARK_PYTHON"]="/usr/local/Python-2.7.11/bin/python"

def json_output(kv):
    k, v = kv
    return "%s\t%s" % (json.dumps(k), json.dumps(v))

def _get_fudid_mid(line):
    y = line.split("\t")
    return (y[11], y[14])

def _get_mid_relate(fudid_mids):
    fudid, mids = fudid_mids
    mids = list(mids)
    n = len(mids)
    res = []
    for i in xrange(n):
        for j in xrange(i+1, n):
            if mids[i] > mids[j]:
                res.append(((mids[i], mids[j]), 1))
            elif mids[i] < mids[j]:
                res.append(((mids[j], mids[i]), 1))
    return res


def _get_mid_sim(pops, pairs):
    mid_tuple, num = pairs
    mid_i, mid_j = mid_tuple
    if mid_i in pops and mid_j in pops:
        sim = num / math.sqrt(pops[mid_i] * pops[mid_j])
        return [(mid_i, (mid_j, sim)), (mid_j, (mid_i, sim))]


if __name__ == "__main__":
    conf = SparkConf().setAppName("mr_spark_icf")
    sc = SparkContext(conf=conf)
    yesterday = datetime.now() - timedelta(1)
    base_path = "/dw/logs/format/app_fbuffer/{}/{:02}/{:02}/part*"
    path = base_path.format(yesterday.year, yesterday.month, yesterday.day)
    # path = "/dw/logs/format/app_fbuffer/2016/07/03/part-r-00000"
    lines = sc.textFile(path)

    # get mid pop dict
    mid_pop = lines.map(lambda x: x.split("\t")[14]) \
                .map(lambda x: (x, 1)) \
                .countByKey()
    sc.broadcast(mid_pop)

    # fudid mid
    # fudid set(mids)
    pairs = lines.map(_get_fudid_mid) \
                .groupByKey() \
                .mapValues(set)
    # (mid_i, mid_j) n
    pairs_counts = pairs.flatMap(_get_mid_relate) \
                        .filter(lambda x: len(x)>0) \
                        .reduceByKey(add)
    # mid_i (mid_j, sim)
    # mid [(mid,sim), (mid, sim)]
    mid_sims = pairs_counts.flatMap(partial(_get_mid_sim, mid_pop)) \
                            .filter(lambda x: isinstance(x, tuple)) \
                            .groupByKey() \
                            .mapValues(list) \
                            .mapValues(lambda x: sorted(x, key=itemgetter(1), reverse=True)[:100])

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

