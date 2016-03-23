---
layout: post
title:  "centos 依赖"
date:   2015-09-02 15:05:31
categories:  linux
tags:  linux
---

* content
{:toc}


## 常用的安装包

```python
yum install openssl openssl-devel
yum install sqlite-devel libffi-devel libxslt-devel
yum install libjpeg-devel freetype-devel

yum -y install cairo-devel libxml2-devel pango-devel pango libpng-devel freetype freetype-devel libart_lgpl-devel

yum install ntp vim-enhanced gcc gcc-c++ gcc-g77 flex bison autoconf automake bzip2-devel ncurses-devel zlib-devel libjpeg-devel libpng-devel libtiff-devel freetype-devel libXpm-devel gettext-devel  pam-devel

yum install -y rrdtool perl-rrdtool rrdtool-devel

yum install glibc-static

yum install gcc cmake make python-devel cairo-devel libxml2 libxml2-devel pango-devel pango libpng-devel freetype freetype-devel libart_lgpl-devel -y

```

## python 安装 readline错误

```python
yum install patch ncurses-devel readline-devel
```

## python 没有 bz2 模块

```python
yum install bzip2-devel zlib zlib-devel
# 重新编译python
```

也可以这样

```python
cp /usr/lib64/python2.6/lib-dynload/bz2.so ../../../env/lib/python2.7/
```


