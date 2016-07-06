---
layout: post
title:  "linux问题解决"
date:   2016-03-20 09:05:31
categories: linux
tags: linux
---

* content
{:toc}

## 打印字符画

```python

echo "Hi Reboot" | figlet

```

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


## 查看CPU核数

查看物理CPU的个数

```python
cat /proc/cpuinfo |grep "physical id"|sort |uniq|wc -l
```
查看逻辑CPU的个数

```python
cat /proc/cpuinfo |grep "processor"|wc -l
```

查看CPU是几核

```python
cat /proc/cpuinfo |grep "cores"|uniq
```

查看CPU的主频

```python
cat /proc/cpuinfo |grep MHz|uniq
```


## 去掉windows换行

```python
sed -i "s/\r//" test.sh

sed -i -e 's/\r$//' scriptname.sh
```

## 查看网络IO

```python
sar -n DEV 1
```

```python
sar 命令行的常用格式：
sar [options] [-A] [-o file] t [n]
在命令行中，n 和t 两个参数组合起来定义采样间隔和次数，t为采样间隔，是必须有的参数，n为采样次数，是可选的，默认值是1，-o file表示将命令结果以二进制格式存放在文件中，
file 在此处不是关键字，是文件名。options 为命令行选项，sar命令的选项很多，下面只列出常用选项：
-A：所有报告的总和。
-u：CPU利用率
-v：进程、I节点、文件和锁表状态。
-d：硬盘使用报告。
-r：没有使用的内存页面和硬盘块。
-g：串口I/O的情况。
-b：缓冲区使用情况。
-a：文件读写情况。
-c：系统调用情况。
-R：进程的活动情况。
-y：终端设备活动情况。
-w：系统交换活动。
-n: 记录网络使用情况
默认监控: sar 5 5     //  CPU和IOWAIT统计状态
(1) sar -b 5 5        // IO传送速率
(2) sar -B 5 5        // 页交换速率
(3) sar -c 5 5        // 进程创建的速率
(4) sar -d 5 5        // 块设备的活跃信息
(5) sar -n DEV 5 5    // 网路设备的状态信息
(6) sar -n SOCK 5 5   // SOCK的使用情况
(7) sar -n ALL 5 5    // 所有的网络状态信息
(8) sar -P ALL 5 5    // 每颗CPU的使用状态信息和IOWAIT统计状态
(9) sar -q 5 5        // 队列的长度（等待运行的进程数）和负载的状态
(10) sar -r 5 5       // 内存和swap空间使用情况
(11) sar -R 5 5       // 内存的统计信息（内存页的分配和释放、系统每秒作为BUFFER使用内存页、每秒被cache到的内存页）
(12) sar -u 5 5       // CPU的使用情况和IOWAIT信息（同默认监控）
(13) sar -v 5 5       // inode, file and other kernel tablesd的状态信息
(14) sar -w 5 5       // 每秒上下文交换的数目
(15) sar -W 5 5       // SWAP交换的统计信息(监控状态同iostat 的si so)
(16) sar -x 2906 5 5  // 显示指定进程(2906)的统计信息，信息包括：进程造成的错误、用户级和系统级用户CPU的占用情况、运行在哪颗CPU上
(17) sar -y 5 5       // TTY设备的活动状态
(18) 将输出到文件(-o)和读取记录信息(-f)
```





