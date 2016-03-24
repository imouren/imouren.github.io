---
layout: post
title:  "linux问题解决"
date:   2016-03-20 09:05:31
categories: linux
tags: linux
---

* content
{:toc}

## 如何查看IP是否冲突

```python
yum install arp-scan  # 安装工具

arp-scan -I eth1 -l  # 使用工具
```

查看结果

```python
192.168.28.167  30:59:b7:b6:42:7f       (Unknown)
192.168.28.175  00:0c:29:c7:e3:5e       VMware, Inc. (DUP: 2)
192.168.28.176  a4:1f:72:52:f7:1a       (Unknown)
192.168.28.174  00:0c:29:fa:c5:7d       VMware, Inc. (DUP: 2)
192.168.28.169  d4:be:d9:63:dd:1f       (Unknown)
```












