---
layout: post
title:  "连续向量最大和（算法设计技术）"
date:   2016-04-13 10:35:31
categories: 算法 编程珠玑
tags: 算法 编程珠玑
---

* content
{:toc}


## 问题

输入是具有n个浮点数的向量x，输出是输入向量的任何连续子向量中的最大和。


## 直观解法

对所有满足 0≤i≤j<n 的(i,j)整数进行迭代。对每个整数对，求和，并检验该总和是否大于迄今为止的最大总和。

```python

def max_sum(L):
    maxsofar = 0
    n = len(L)
    for i in range(n):
        for j in range(i, n):
            s = 0
            for k in range(i, j+1):
                s += L[k]
            maxsofar = max(s, maxsofar)
    return maxsofar

```

算法的时间复杂度为 $O(n^3)$

## 进行简单优化

我们可以求出起始为 i 元素的最大连续和(0≤i<n)，即 [i], [i, i+1], ... [i, i+1, ..n] 中最大和

这样循环两次就可以了

```python

def max_sum2(L):
    maxsofar = 0
    n = len(L)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += L[j]
            maxsofar = max(s, maxsofar)
    return maxsofar

```

算法的时间复杂度为 $O(n^2)$


## 分治算法

分治法原理：

:    要解决规模为n的问题，可递归解决两个规模近似为n/2的子问题，然后对它们的答案进行合并以得到整个问题的答案。


本例中，将n个向量的问题，分为两个近似相等的子向量a和b

然后递归找出a b 中元素总的最大和子向量，分别记为$m_a$和$m_b$

最大和子向量，要么在$m_a$中，要么在$m_b$中，或者跨越 a b 向量的 $m_c$ 中

```python

def max_sum_quick(L, left, right):
    print left, right
    # 无元素
    if left > right:
        return 0
    # 只有一个元素
    if left == right:
        return max(L[left], 0)
    # 以上为退出条件
    # 以下为求mc
    middle = (left + right) / 2
    # 左边距离中间的最大连续和
    left_max = 0
    left_sum = 0
    for i in range(middle, left-1, -1):
        left_sum += L[i]
        left_max = max(left_max, left_sum)
    # 右边距离中间的最大连续和
    right_max = 0
    right_sum = 0
    for i in range(middle+1, right+1):
        right_sum += L[i]
        right_max = max(right_max, right_sum)

    return max(left_max+right_max,
            max_sum_quick(L, left, middle),
            max_sum_quick(L, middle+1, right))

```

算法的时间复杂度为 $O(nlogn)$


## 扫描算法


前i个元素中，最大连续总和要么在前i-1个元素中（maxsofar），要么结束位置为i （maxendinghere）


```python

def max_sum_line(L):
    maxsofar = 0
    maxendinghere = 0
    for num in L:
        maxendinghere = max(maxendinghere+num, 0)
        maxsofar = max(maxsofar, maxendinghere)
    return maxsofar

```

算法的时间复杂度为 $O(n)$


## 相关变种问题

给定一个长度为N的整数数组a，求不重叠的两个子数组的和的最大值


1. left数组 保存 0 到 i 的连续最大和 [0, n)

2. right数组 保存 i 到 n 的连续最大和 (0, n]

3. 求出对应的 left[i]+right[i] 最大值即可

```python

def get_two_max_sum(L):
    if len(L) < 2:
        return -1

    left = []
    right = []

    maxsofar, maxendinghere = 0, 0
    for num in L[:-1]:
        maxendinghere = max(maxendinghere+num, 0)
        maxsofar = max(maxsofar, maxendinghere)
        left.append(maxsofar)

    maxsofar, maxendinghere = 0, 0
    for num in L[:0:-1]:
        maxendinghere = max(maxendinghere+num, 0)
        maxsofar = max(maxsofar, maxendinghere)
        right.append(maxsofar)

    total = [x+y for x, y in zip(left, right[::-1])]
    print left, right[::-1]
    return max(total)

```
