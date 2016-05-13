---
layout: post
title:  "取样"
date:   2016-04-14 10:35:31
categories: 算法 编程珠玑
tags: 算法 编程珠玑
---

* content
{:toc}


## 从序列中随机选出n个不重复项

可以使用将列表顺序打乱，然后取前n个

这里其实就是 random.shuffle 的实现方式

从[0, n)遍历当前元素i, 从[i+1, n]随机取元素跟 i 互换

```python

import random

def shuffle(x):
    for i in reversed(xrange(1, len(x))):
        j = int(random.random() * (i+1))
        x[i], x[j] = x[j], x[i]


```




更优化的方法是打乱前n个数据，然后取前n个 其实就是 random.sample的实现

实际源码中进行了其他优化

```python

def sample(x, n):
    res = []
    for i in range(n):
        if n < len(x):
            j = random.randint(i+1, len(x)-1)
            x[i], x[j] = x[j], x[i]
        res.append(x[i])
    return res

```

## 指定概率选取元素

均来自cookbook

概率总和为1.0

```python

def random_pick(some_list, probabilities):
    x = random.uniform(0,1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:break
    return item

```

基于机会

```python

def random_picks(sequence, relative_odds):
    table = [z for x, y in zip(sequence, relative_odds) for z in [x]*y]
    while True:
        yield random.choice(table)

import itertools

x = random_picks('ciao', [1, 2, 3, 1])

print list(itertools.islice(x, 8))

```
