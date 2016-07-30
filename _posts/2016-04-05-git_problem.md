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

## github 免密码

修改使用ssh方式

点击头像--> setttings --> SSH and GPG keys

```python

git remote rm origin

git@github.com:yourname/yourname.github.io.git

git push origin master
```

## clone所有远程分支到本地

```python
克隆项目
$ git clone git://example.com/myproject
$ cd myproject

查看分支
$ git branch
* master

查看所有分支
$ git branch -a
* master
  remotes/origin/HEAD
  remotes/origin/master
  remotes/origin/v1.0-stable
  remotes/origin/experimental

克隆远程分支
$ git checkout origin/experimental
克隆并切换到远程分支
$ git checkout -b experimental origin/experimental
```

自动脚本

```python

#!/bin/bash
for branch in `git branch -a | grep remotes | grep -v HEAD | grep -v master `; do
   git branch --track ${branch#remotes/origin/} $branch
   git checkout ${branch#remotes/origin/}
done

```

或者


```python

#!/bin/bash
for branch in `git branch -a | grep remotes | grep -v HEAD | grep -v master `; do
   git checkout -b ${branch#remotes/origin/}
done

```







