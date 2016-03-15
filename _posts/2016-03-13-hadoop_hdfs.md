---
layout: post
title:  "hadoop HDFS"
date:   2016-03-13 09:05:31
categories: hadoop
tags: hadoop
---

* content
{:toc}

namenode 和 datanode

## 客户端上传文件到hdfs的过程


1. client要将文件上传到 hdfs://myhdfs:9000/aa/bb/cc/dd.txt
2. client 请求 namenode，问是否可以存
3. namenode 查询存储的元数据，发现hdfs上面没有这个目录，可以存放
4. namenode 通知 client 可以存放，假如文件很大，需要分成三个块，namenode 通知 client 存放文件的三个 datanode1、datanode2、datanode3
5. client 请求 datanode1 存放文件块，文件块默认需要三个副本
6. 异步处理，datanode1 通过pipline负责写多个副本，如果写失败，会通知到 datanode1，datanode1 通知 namenode，namenode 会选择任意一个副本所在的datanode写其余副本到其他datanode
7. client继续写入 datanode2、datanode3

任意一个数据块（默认128M）都在namenode上有一条记录，如果大量小文件的话，会浪费namenode的元数据空间，但不会浪费datanode空间

datanode 备份block块，优先找不同机架上 datanode，保证高可用，然后在本机架随机找个datanode

## namenode 元数据存储

大量客户端访问namenode，如果元数据存在磁盘上，响应速度明显不够用；如果存放到内存上，则宕机数据丢失；如果换成内存部分数据，定时flush到磁盘，则可能丢失内存中未保存到磁盘的数据。

存储机制：

* namenode 内存提供查询，读写分离
* 新记录以日志的形式记录到 edits，这个文件很小，只提供追加，不提供修改
* fsimage 磁盘文件  是内存中数据的镜像



元数据记录过程：

1. 客户端上传文件时，NN先向edits log 中记录元数据日志
2. 客户端开始上传文件，完成后返回成功信息给NN，NN就在内存中写入这次上传操作产生的元数据
3. 每当edits log写满的时候，要将这段时间新的元数据刷到fsimage中



edits log 合并到 fsimage的时候，需要secondary namenode

1. 当edits满的时候，NN 通知 secondary namenode 进行 checkpoint 操作
2. secondary namenode 告诉 NN，停止往edits log中写数据，因为要进合并
3. NN 产生新的 edits 日志文件叫 edits.new
4. secondary namenode 将NN的fsimage 和 edits 下载下来
5. 进行合并，成为新的文件fsimage.chkpoint
6. SN 将新的镜像上传给NN
7. NN 用新的fsimage 替换旧的 fsimage，重命名 edits.new 为 edits

namenode 的主要职责：

* 维护元数据信息
* 维护HDFS的目录树
* 响应客户端的请求

元数据的格式：

{% highlight python %}
文件名，副本数，几个block块，每个block块所在datanode

/test/a.log, 3, {blk_1,blk_2}, [{blk_1:[h0,h1,h3]}, {blk_2:[h0,h2,h4]}]
{% endhighlight %}

触发checkpoint的条件：

* fs.checkpoint.period指定两次checkpoint的最大时间间隔，默认3600秒
* fs.checkpoint.size 规定edits文件的最大值，一旦超过这个值，强制checkpoint，默认64M

namdenode存放位置，在`$HADOOP_HOME/data/dfs/name/current`

{% highlight python %}
[hadoop@hadoop01 current]$ pwd
/home/hadoop/app/hadoop-2.4.1/data/dfs/name/current
[hadoop@hadoop01 current]$ ll
total 3148
-rw-rw-r--. 1 hadoop hadoop      42 Dec  6 01:46 edits_0000000000000000001-0000000000000000002
-rw-rw-r--. 1 hadoop hadoop   13199 Dec  6 02:46 edits_0000000000000000003-0000000000000000115
-rw-rw-r--. 1 hadoop hadoop      42 Dec  6 03:46 edits_0000000000000000116-0000000000000000117
-rw-rw-r--. 1 hadoop hadoop      42 Dec  6 04:46 edits_0000000000000000118-0000000000000000119
-rw-rw-r--. 1 hadoop hadoop 1048576 Dec  6 04:46 edits_0000000000000000120-0000000000000000120
-rw-rw-r--. 1 hadoop hadoop      42 Dec  6 05:33 edits_0000000000000000121-0000000000000000122
-rw-rw-r--. 1 hadoop hadoop      42 Dec  6 06:33 edits_0000000000000000123-0000000000000000124
-rw-rw-r--. 1 hadoop hadoop 1048576 Dec  6 06:33 edits_0000000000000000125-0000000000000000125
-rw-rw-r--. 1 hadoop hadoop      42 Mar 13 05:34 edits_0000000000000000126-0000000000000000127
-rw-rw-r--. 1 hadoop hadoop     104 Mar 13 06:35 edits_0000000000000000128-0000000000000000130
-rw-rw-r--. 1 hadoop hadoop      42 Mar 13 07:35 edits_0000000000000000131-0000000000000000132
-rw-rw-r--. 1 hadoop hadoop      42 Mar 13 08:35 edits_0000000000000000133-0000000000000000134
-rw-rw-r--. 1 hadoop hadoop 1048576 Mar 13 08:35 edits_inprogress_0000000000000000135
-rw-rw-r--. 1 hadoop hadoop    1850 Mar 13 07:35 fsimage_0000000000000000132
-rw-rw-r--. 1 hadoop hadoop      62 Mar 13 07:35 fsimage_0000000000000000132.md5
-rw-rw-r--. 1 hadoop hadoop    1850 Mar 13 08:35 fsimage_0000000000000000134
-rw-rw-r--. 1 hadoop hadoop      62 Mar 13 08:35 fsimage_0000000000000000134.md5
-rw-rw-r--. 1 hadoop hadoop       4 Mar 13 08:35 seen_txid
-rw-rw-r--. 1 hadoop hadoop     204 Mar 13 05:33 VERSION
{% endhighlight %}

## datanaode 的工作原理

datanode 提供真实的文件数据存储服务

文件块（block）是最基础的存储单位。HDFS默认block为128M，一个256M的文件会有两个block。可以通过 dfs.block.size 配置

如果一个文件小于一个数据库，并不占用整个数据库的存储空间

多复本。hdfs-site.xml的dfs.replication属性

我们上传一个大于128M的文件到hdfs中：

{% highlight python %}
[hadoop@hadoop01 sbin]$ hadoop fs -ls -h /jdk-7u65-linux-i586.tar.gz
-rw-r--r--   1 hadoop supergroup    136.9 M 2015-12-06 01:51 /jdk-7u65-linux-i586.tar.gz
{% endhighlight %}

查看下这个文件被分割的块数：
{% highlight python %}
[hadoop@hadoop01 finalized]$ ll -h
total 139M
-rw-rw-r--. 1 hadoop hadoop 128M Dec  6 01:51 blk_1073741825
-rw-rw-r--. 1 hadoop hadoop 1.1M Dec  6 01:51 blk_1073741825_1001.meta
-rw-rw-r--. 1 hadoop hadoop 9.0M Dec  6 01:51 blk_1073741826
-rw-rw-r--. 1 hadoop hadoop  72K Dec  6 01:51 blk_1073741826_1002.meta
{% endhighlight %}

分割的文件，完全可以手工合并还原：
{% highlight python %}
# 拷贝两个文件到其他目录
[hadoop@hadoop01 ~]$ cat blk_1073741826 >> blk_1073741825 # 合并
[hadoop@hadoop01 ~]$ tar zxvf blk_1073741825 # 正常解压
{% endhighlight %}




