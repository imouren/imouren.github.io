---
layout: post
title:  "读取文本并自动转义"
date:   2016-06-08 10:35:31
categories: python
tags: python
---

* content
{:toc}


## 需求

从文件读取字符串，遇到 "\t" 这样的字符 会变为 "\\t" 这样转义过的字符

需要保留原来的"\t"，即 table 分隔符


## 思路

使用 json.loads 实现

将字符串拼为json的字符串，然后还原为python的字符串

```python

In [16]: json.loads('"%s"' % "a\\tb")
Out[16]: u'a\tb'

```

## 问题

遇到 '\001' 类似的字符会出异常

```python

In [10]: x
Out[10]: 'a\\tb\\001\\u0001'

In [11]: import json

In [13]: json.loads('"%s"' % x) 
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-13-d30ef9fcdeb8> in <module>()
----> 1 json.loads('"%s"' % x)

/usr/local/lib/python2.7/json/__init__.pyc in loads(s, encoding, cls, object_hook, parse_float, parse_int, parse_constant, object_pairs_hook, **kw)
    336             parse_int is None and parse_float is None and
    337             parse_constant is None and object_pairs_hook is None and not kw):
--> 338         return _default_decoder.decode(s)
    339     if cls is None:
    340         cls = JSONDecoder

/usr/local/lib/python2.7/json/decoder.pyc in decode(self, s, _w)
    364 
    365         """
--> 366         obj, end = self.raw_decode(s, idx=_w(s, 0).end())
    367         end = _w(s, end).end()
    368         if end != len(s):

/usr/local/lib/python2.7/json/decoder.pyc in raw_decode(self, s, idx)
    380         """
    381         try:
--> 382             obj, end = self.scan_once(s, idx)
    383         except StopIteration:
    384             raise ValueError("No JSON object could be decoded")

ValueError: Invalid \escape: line 1 column 6 (char 5)

```

## 解决

使用 [python-cjson](https://pypi.python.org/pypi/python-cjson) 来解决这个问题

```python

In [14]: cjson.decode('"%s"' % x)     
Out[14]: u'a\tb\x01\x01'

```
