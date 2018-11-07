---
layout: post
title:  "GIT相关问题"
date:   2016-04-05 09:05:31
categories: git
tags: git
---

* content
{:toc}

## 创建远程分支

git checkout -b lzc  # 从当前分支创建新分支

git push origin lzc  # 提交到远程

`git branch --set-upstream-to=origin/lzc`  # 设置跟踪

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

## 删除文件

windows下删除pyc

for /r ./ %i in (*.pyc) do del %i

将删除同步给git
git add -A

git commit -a -m "del pyc"



## 取消跟踪


取消跟踪文件

`git rm --cached FILENAME`

取消跟踪目录

`git rm --cached FILENAME -r`

## 生成密钥（windows）

查看密钥
`ls -al ~/.ssh`

生成密钥
`ssh-keygen -t rsa -b 4096 -C "your_email@example.com"`

操作均使用 git bash 以及后续 git 指令操作


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


git 端口号修改

```python

# .git/config

原来的
url = git@code.funshion.com:japan/thor_crystal.git

修改过的

url = ssh://git@code.funshion.com:88888/japan/thor_crystal.git

```


## 合并其他版本的部分文件

```python
git checkout source_branch <paths>...
```

## github 误提交大文件解决

```python
git filter-branch -f --index-filter "git rm -rf --cached --ignore-unmatch FOLDERNAME" -- --all
```

## 查看最新的版本号

```python
git rev-parse HEAD
```

## 回滚代码

```python
git reset --hard   22f8aae # 22f8aae 为某次提交的提交号

git push origin maser --force  # 强推到远程
```

## 提交到多个项目

参考 https://www.cnblogs.com/mafly/p/4639572.html?utm_source=tuicool&utm_medium=referral

```python
git remote set-url --add origin git@code.xx.com:malaysia/your.git

git push -f origin master # 首次强制提交
```
