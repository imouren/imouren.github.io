---
layout: post
title:  "排序"
date:   2016-04-13 10:35:31
categories: 算法 编程珠玑
tags: 算法 编程珠玑
---

* content
{:toc}


## 插入排序

大多数纸牌游戏玩家都采用插入排序来整理牌。

顺序取元素i，[1, n) 比较元素i和之前的元素大小，

若之前元素大于i则进行交互，直到不大于。用j来跟踪筛选元素

```python

def insert_sort(L):
    n = len(L)
    for i in range(1, n):
        for j in range(i, 0, -1):
            if L[j] < L[j-1]:
                L[j], L[j-1] = L[j-1], L[j]
            # 之前的元素是有序的，不必比较到最后了
            else:
                break

```

算法的时间复杂度为 $O(n^2)$

如果插入排序中的元素基本都是有序的，则排序速度会快很多，因为每个元素移动的距离很短

## 冒泡排序


比较相邻的元素。如果第一个比第二个大，就交换他们两个。

对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对。在这一点，最后的元素应该会是最大的数。

针对所有的元素重复以上的步骤，除了最后一个。

持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较

```python

def bubble_sort(L):
    n = len(L)
    for i in range(n-1, 0, -1):
        for j in range(i):
            if L[j] > L[j+1]:
                L[j], L[j+1] = L[j+1], L[j]

```

算法的时间复杂度为 $O(n^2)$

## 选择排序

类似冒泡排序，只是不两两交换，而是当前值和剩余序列的最小值进行交换

```python

def choice_sort(L):
    n = len(L)
    for i in range(n):
        minidx = i
        for j in range(i+1, n):
            if L[j] < L[minidx]:
                minidx = j
        if minidx != i:
            L[minidx], L[i] = L[i], L[minidx]

```

算法的时间复杂度为 $O(n^2)$


## 快速排序

快速排序使用了 分治法 的原理，将数组分为两个小部分，然后递归排序

随机选择轴心计算，大于改数在一个数组，小于的在另外一个数组，等于的在一个数

然后递归排序，大于和小于的数组

```python
# cookbook
def quick_sort(L):
    if not L:
        return L
    pivot = random.choice(L)
    def lt(x): return x<pivot
    def gt(x): return x>pivot
    return quick_sort(filter(lt, L)) + [pivot]*L.count(pivot) + quick_sort(filter(gt, L))

```
算法的时间复杂度为 $O(nlogn)$

## 选取序列中最小的第n个元素

思路类似快速排序，采用分治法。

随机选个元素，大于改元素在over数组，小于的在under数组，等于的没有给数组，直接用个计数

若n 小于under数组长度，则第n个元素在under数组

若n 小于 under数组长度+计数 则第n个元素就是我们选取的轴心数

否则，去under数组找，同时将n相应减去 under长度和计数

```python
# cookbook 代码
def select(data, n):
    # 最小元素为0
    # 创建新列表
    data = list(data)
    if n < 0:
        n += len(data)
    if not 0 <= n < len(data):
        raise ValueError, "can't get rank %d out of %d" % (n, len(data))
    # 主循环，类似快速排序，但不需要递归
    times = 0
    while True:
        pivot = random.choice(data)
        pcount = 0 # 和中轴元素相等元素数量
        under, over = [], []
        uappend, oappend = under.append, over.append  # 移除到循环外，优化速率
        for item in data:
            times += 1
            if item < pivot:
                uappend(item)
            elif item > pivot:
                oappend(item)
            else:
                pcount += 1
        numunder = len(under)
        if n < numunder:
            data = under
        # 找到最小的第n个元素
        elif n < (numunder + pcount):
            print times
            return pivot
        else:
            data = over
            n -= (numunder + pcount)

```

算法的时间复杂度为 $O(n)$


## 获取序列中最小的几个元素

排序的时间复杂度为 $O(nlogn)$

如果n很小，则获取的时间复杂度可以为 $O(n)$

```python

import heapq

# 生成器
def isorted(data):
    data = list(data)
    heapq.heapify(data)
    while data:
        yield heapq.heappop(data)

from itertools import islice

res = list(islice(isorted(L), n))

# 直接取最小n个
def smallest(data, n):
    return heapq.nsmallest(n, data)

```
