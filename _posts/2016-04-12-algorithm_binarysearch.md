---
layout: post
title:  "二分搜索"
date:   2016-04-12 16:35:31
categories: 算法 编程珠玑
tags: 算法 编程珠玑
---

* content
{:toc}


简单编写下：

```python

def binary_search(L, num):
    right = len(L) - 1
    left = 0
    while right >= left:
        middle = (left + right) / 2
        if num == L[middle]:
            return num
        if num > L[middle]:
            left = middle + 1
        else:
            right = middle - 1
    return -1

L = range(10)

for i in L:
    print binary_search(L, i)

print binary_search(L, 10)

```

python 标准库提供了bisect模块，可以很容易检测x是否在排序后的L中

```python
# cookbook
import bisect

L = range(100)
x = -1
x_insert_point = bisect.bisect_right(L, x)
x_is_present = L and L[x_insert_point-1] == x

```
