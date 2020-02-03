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


## 代码冲突

错误一：

Please, commit your changes or stash them before you can merge

果希望用代码库中的文件完全覆盖本地工作版本. 方法如下:

```python
git reset --hard
git pull
```

错误二：

Please move or remove them before you can merge

如果确定使用云端的代码,最方便的解决方法是删除本地修改,可以使用以下命令：

```python
git clean  -d  -fx ""
d  -----删除未被添加到git的路径中的文件
f  -----强制运行
x  -----删除忽略文件已经对git来说不识别的文件
```




## 设置使用代理

将你的proxy server地址代替以下的127.0.0.1

http类型代理

git config --global http.proxy http://127.0.0.1:1080 （这条即可）
git config --global https.proxy https://127.0.0.1:1080

socks5类型代理

git config --global http.proxy ‘socks5://127.0.0.1:1080’ （这条即可）
git config --global https.proxy ‘socks5://127.0.0.1:1080’


查当前的代理配置

git config --global --get http.proxy
git config --global --get https.proxy

取消设置代理

git config --global --unset http.proxy
git config --global --unset https.proxy

设置使用需账户密码验证的代理
http类型代理
git config --global http.proxy http://username:passwd@127.0.0.1:1080
socks5类型代理
git config --global http.proxy socks5://username:passwd@127.0.0.1:1080
