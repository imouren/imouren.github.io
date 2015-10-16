---
layout: post
title:  "安装cx_orcle"
date:   2015-10-16 15:05:31
categories: python
tags: python
---

* content
{:toc}



### 下载oracle客户端

去[官方网站]下载客户端

{% highlight python %}
wget http://download.oracle.com/otn/linux/instantclient/121020/instantclient-basic-linux.x64-12.1.0.2.0.zip
wget http://download.oracle.com/otn/linux/instantclient/121020/instantclient-sdk-linux.x64-12.1.0.2.0.zip
{% endhighlight %}

### 安装oracle

{% highlight python %}
yum -y install libaio bc flex # 可能的依赖
unzip instantclient-basic-linux.x64-12.1.0.2.0.zip -d /opt/
unzip instantclient-sdk-linux.x64-12.1.0.2.0.zip -d /opt/
cd /opt/instantclient_12_1
ln -s libclntsh.so.12.1 libclntsh.so # libclntsh.so软连
{% endhighlight %}


### 环境变量添加
{% highlight python %}
vim /etc/profile
#添加两行
export ORACLE_HOME=/opt/instantclient_12_1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ORACLE_HOME
#生效
source /etc/profile
{% endhighlight %}

### 安装cx_oracle
{% highlight python %}
pip install cx_oracle
{% endhighlight %}

[官方网站]: http://www.oracle.com/technetwork/topics/linuxx86-64soft-092277.html