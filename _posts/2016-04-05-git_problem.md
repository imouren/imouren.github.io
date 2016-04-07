---
layout: post
title:  "GIT 错误"
date:   2016-04-05 09:05:31
categories: git
tags: git
---

* content
{:toc}

## 提交到远程报错

fatal: the remote end hung up unexpectedly

windows:

```python
在 .git/config 文件中加入
[http]
postBuffer = 524288000
```


linux:

```python
git config http.postBuffer 524288000
```













