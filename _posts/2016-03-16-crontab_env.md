---
layout: post
title:  "crontab 与 环境变量"
date:   2016-03-16 09:05:31
categories: linux crontab
tags: linux crontab
---

* content
{:toc}


### 问题

cx_Oracle 的运行需要环境变量，本身已经配置到 /etc/profile 中

并且手动运行脚本OK，而在crontab中有报错，显示找不到so文件

### 分析

某个用户登陆都会自动运行`/etc/profile` 和 `~/.bash_profile` 中的环境变量

但是，crontab 并不知道用户的环境，所以你要保证在脚本中提供所有必要的路径和环境变量

### 解决

* 使用全路径。本身自己也习惯使用全路径，所以没能提前规避环境变量的问题。

* 在执行脚本中重新导入下环境变量

```bash

source /etc/profile

source ~/bash_profile

```

* 在crontab 中添加环境变量

```bash
0 * * * * . /etc/profile; /path/your.sh
```

