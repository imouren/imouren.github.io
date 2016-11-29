---
layout: post
title:  "动态更改python的类方法和实例方法"
date:   2016-10-20 09:05:31
categories: spark
tags: spark
---

* content
{:toc}

## 更改类方法

直接指向另外一个函数即可

```python
class A(object):

    def test(self):
        print "testa"


def test2(self):
    print "test2"

A.test = test2

a = A()
a.test()

```

## 修改实例方法

直接指向会报错，需要额外处理下

```python
class B(object):

    def test(self):
        print "testb"

A.test = test2

b = B()
# b.test = test2  # TypeError: test2() takes exactly 1 argument (0 given)

# 引发错误的原因就是被调用的对象并没有作为第一个参数传给我们写的函数。
# 当然我们可以自己把参数传进去，但是在这个替换类方法的场景下并不奏效。
# 解决这个问题的正确方案是用 type 这个模块里的 MethodType 函数
import types
b.test = types.MethodType(test2, b)
b.test()
```
