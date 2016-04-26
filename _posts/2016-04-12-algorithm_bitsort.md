---
layout: post
title:  "使用位图排序"
date:   2016-04-12 15:05:31
categories: 算法 编程珠玑
tags: 算法 编程珠玑
---

* content
{:toc}


### 给定不重复正整数均不大于N的输入进行排序

因为上限值确定，可以利用空间换时间，使用位图的方式进行排序

```python

# 构建数组，使输入数字所在索引值为改数字值
def bit_sort(input_list, N):
    x = [None] * N
    for i in input_list:
        x[i] = i
    res = [i for i in x if i]
    return res

N = 20
input_list = [3, 8, 1, 18, 2, 5]

print bit_sort(input_list, N)

# [1, 2, 3, 5, 8, 18]

```

python位移的操作

```python

def set_bit(num, index):
    num |= (1<<index)
    return num

def clear_bit(num, index):
    num &= ~(1<<index)
    return num

def check_bit(num, index):
    return bool(num & (1<<index))

```

直接使用位来操作可以得到同样结果

```python

def bit_sort2(input_list, N):
    x = 0
    for i in input_list:
        x = set_bit(x, i)
    res = [i for i in xrange(N) if check_bit(x, i)]
    return res

print bit_sort2(input_list, N)

```
