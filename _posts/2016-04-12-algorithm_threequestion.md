---
layout: post
title:  "找不存在数、旋转向量、变位词"
date:   2016-04-12 15:35:31
categories: 算法 编程珠玑
tags: 算法 编程珠玑
---

* content
{:toc}


## 给定一个最多含有40亿个随机排列的32位整数的顺序文件，找一个不存在文件中的32位整数

为什么文件至少缺少这样的数？

```python

32位整数有 (2**32) 个；
40亿的不重复整数有 (40*10**8) 个；

(2**32) > (40*10**8) 所以必然存在

```

查找方法，使用二分查找。

关键点在于找到能确定某一个半范围缺少整数的探测方法。

如果未缺少，那么低位为0和1的数字应该各为2**16个，如果不足则缺少，以此类推。

```python

import random

def set_bit(num, index):
    num |= (1<<index)
    return num

def clear_bit(num, index):
    num &= ~(1<<index)
    return num

def check_bit(num, index):
    return bool(num & (1<<index))


def get_lost_one(alist, N):
    res = []
    for i in xrange(N):
        r0 = []
        r1 = []
        for num in alist:
            if check_bit(num, i):
                r1.append(num)
            else:
                r0.append(num)
        if len(r0) < 2**(N-i-1):
            alist = r0
            res.append("0")
        else:
            alist = r1
            res.append("1")
    return eval('0b'+''.join(res[::-1]))

N = 16

alist = range(2**N)

x = random.choice(alist)

alist.pop(x)

res = get_lost_one(alist, N)

print res == x  # True

```


## 将一个n元一维向量向左旋转i个位置

当n=8,i=3时，向量 abcdefgh 旋转为 defgabc

可以使用额外空间：

```python

def rotate(L, i):
    n = len(L)
    L[:n] = L[i:n] + L[:i]

x = ["a", "b", "c", "d", "e", "f", "g"]

rotate(x, 3)
print x

```

使用翻手法：

可以将问题看做数组ab转变为ba

写一个函数，可以将数组中特定元素求逆

ab --> 对a求逆 $a^rb$ --> 对b求逆 $a^rb^r$ --> 对整体求逆 $(a^rb^r)^r$ --> 恰好是ba


```python

def rotate2(L, i):
    n = len(L)
    def _reverse(alist, i, j):
        while i < j:
            alist[i], alist[j] = alist[j], alist[i]
            i += 1
            j -= 1
    _reverse(L, 0, i-1)
    _reverse(L, i, n-1)
    _reverse(L, 0, n-1)

x = ["a", "b", "c", "d", "e", "f", "g"]
rotate2(x, 3)
print x

```


## 给定一个英语字典，找出其中所有变位词集合。

例如 "pots" "stop" "tops" 互为变位词，每个单词都可以通过改变其他单词中字母的顺序得到

将变位词生产相同的指纹，这里使用把字母排序作为指纹。

```python

from collections import defaultdict

def get_anagram(words):
    word_dict = defaultdict(list)
    for word in words:
        sign = "".join(sorted(list(word)))
        word_dict[sign].append(word)
    return word_dict

words = ["pans", "pots", "opt", "snap", "stop", "tops"]
word_dict = get_anagram(words)
for v in word_dict.itervalues():
    if len(v) > 1:
        print v

```
