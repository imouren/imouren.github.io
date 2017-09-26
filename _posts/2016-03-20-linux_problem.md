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

## 查看centos版本

```python
cat  /etc/redhat-release

# 查看内核版本
uname  -r
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

## wget获取链接 & 后面参数无法获取

可以将URL用双引号括起来

wget "http://api-vmis.xxx.xx/list/?fudid=3d6118bc04c605c6e8c506c6e7ab70655f75d58452317c7d3ba70ec62129fb2d&v=1.0.0.0&app=1&pg=1&cl=TODO2&theme=0&uc=201&lt=first"

## gclib

查看系统glibc支持的版本

```python
strings /lib64/libc.so.6 |grep GLIBC_

rpm -qa |grep glibc
```

去下载需要的版本

```python
http://ftp.gnu.org/gnu/glibc/
```

## 查看请求某个URL用时多久

```python
curl -so/dev/null baidu.com -w "%{time_total}\n"
```

### 网络性能

```python

1、iftop界面相关说明

界面上面显示的是类似刻度尺的刻度范围，为显示流量图形的长条作标尺用的。

中间的<= =>这两个左右箭头，表示的是流量的方向。

TX：发送流量
RX：接收流量
TOTAL：总流量
Cumm：运行iftop到目前时间的总流量
peak：流量峰值
rates：分别表示过去 2s 10s 40s 的平均流量

2、iftop相关参数

常用的参数

-i设定监测的网卡，如：# iftop -i eth1

-B 以bytes为单位显示流量(默认是bits)，如：# iftop -B

-n使host信息默认直接都显示IP，如：# iftop -n

-N使端口信息默认直接都显示端口号，如: # iftop -N

-F显示特定网段的进出流量，如# iftop -F 10.10.1.0/24或# iftop -F 10.10.1.0/255.255.255.0

-h（display this message），帮助，显示参数信息

-p使用这个参数后，中间的列表显示的本地主机信息，出现了本机以外的IP信息;

-b使流量图形条默认就显示;

-f这个暂时还不太会用，过滤计算包用的;

-P使host信息及端口信息默认就都显示;

-m设置界面最上边的刻度的最大值，刻度分五个大段显示，例：# iftop -m 100M

进入iftop画面后的一些操作命令(注意大小写)

按h切换是否显示帮助;

按n切换显示本机的IP或主机名;

按s切换是否显示本机的host信息;

按d切换是否显示远端目标主机的host信息;

按t切换显示格式为2行/1行/只显示发送流量/只显示接收流量;

按N切换显示端口号或端口服务名称;

按S切换是否显示本机的端口信息;

按D切换是否显示远端目标主机的端口信息;

按p切换是否显示端口信息;

按P切换暂停/继续显示;

按b切换是否显示平均流量图形条;

按B切换计算2秒或10秒或40秒内的平均流量;

按T切换是否显示每个连接的总流量;

按l打开屏幕过滤功能，输入要过滤的字符，比如ip,按回车后，屏幕就只显示这个IP相关的流量信息;

按L切换显示画面上边的刻度;刻度不同，流量图形条会有变化;

按j或按k可以向上或向下滚动屏幕显示的连接记录;

按1或2或3可以根据右侧显示的三列流量数据进行排序;

按<根据左边的本机名或IP排序;

按>根据远端目标主机的主机名或IP排序;

按o切换是否固定只显示当前的连接;

按f可以编辑过滤代码，这是翻译过来的说法，我还没用过这个！

按!可以使用shell命令，这个没用过！没搞明白啥命令在这好用呢！

按q退出监控。

```


### IO状态

http://www.cnblogs.com/ggjucheng/archive/2013/01/13/2858810.html

```python

root@l-web6.44.recommended.prod.ctc:~># iostat -x
Linux 2.6.32-642.el6.x86_64 (l-web6.44.recommended.prod.ctc)    09/14/2017      _x86_64_        (8 CPU)

avg-cpu:  %user   %nice %system %iowait  %steal   %idle
           5.16    0.01    0.52    0.04    0.00   94.28

Device:         rrqm/s   wrqm/s     r/s     w/s   rsec/s   wsec/s avgrq-sz avgqu-sz   await r_await w_await  svctm  %util
sdb               0.00     0.00    0.00    0.00     0.00     0.00    11.59     0.00    0.28    0.28    0.00   0.28   0.00
sda               0.00     6.25    0.03    1.04     1.21    58.28    55.49     0.01   12.12    1.07   12.48   4.29   0.46
sdc               0.00     6.29    0.00    0.06     0.56    50.84   803.01     0.02  277.88    1.89  288.07   3.20   0.02

我们先列举一下各个性能指标的简单说明。

rrqm/s
每秒进行merge的读操作数目。
wrqm/s
每秒进行merge的写操作数目。
r/s
每秒完成的读I/O设备次数。
w/s
每秒完成的写I/O设备次数。
rsec/s
每秒读扇区数。
wsec/s
每秒写扇区数。
rkB/s
每秒读K字节数。
wkB/s
每秒写K字节数。
avgrq-sz
平均每次设备I/O操作的数据大小(扇区)。
avgqu-sz
平均I/O队列长度。
await
平均每次设备I/O操作的等待时间(毫秒)。
svctm
平均每次设备I/O操作的服务时间(毫秒)。
%util
一秒中有百分之多少的时间用于I/O操作，或者说一秒中有多少时间I/O队列是非空的。

```