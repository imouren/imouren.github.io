---
layout: post
title:  "shingle算法"
date:   2016-04-11 14:05:31
categories: 算法
tags: 算法
---

* content
{:toc}



Shingle算法是搜索引擎去掉相同或相似页面的其中一种基本算法。

Shingle算法：对长度L的文档，每个N个汉字取一个Shingle，一共渠道L-N+1个Shingle

比如对`今天的天气真好`取N为1，结果为`[今天, 天的, 的天, 天气, 气真, 真好]`



取cut可以用

```python

[word[i:i + n] for i in xrange(len(word) - n + 1)]
```


或者参考 cookbook 中的重叠窗口循环序列 进行改写

```python

import itertools

def windows(iterable, length=2, overlap=1):
    it = iter(iterable)
    results = list(itertools.islice(it, length))
    while len(results) == length:
        yield results
        results = results[length-overlap:]
        results.extend(itertools.islice(it, length-overlap))
    #if results:
    #    yield results


if __name__ == "__main__":
    x = "foobarbazer"
    print map("".join, windows(x))

```

相似度计算使用 jaccard系数：

$$jaccard = \frac{A \bigcup B}{A \bigcap B}$$
