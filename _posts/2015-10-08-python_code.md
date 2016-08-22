---
layout: post
title:  "python代码片段"
date:   2015-10-08 10:05:31
categories:  python
tags:  python
---

* content
{:toc}



## 数字格式化为百分百

```python

from numbers import Number

def as_percent(v, precision='0.2'):
    """Convert number to percentage string."""
    if isinstance(v, Number):
        return "{{:{}%}}".format(precision).format(v)
    else:
        raise TypeError("Numeric type required")

as_percent(0.5)  # '50.00%'

```