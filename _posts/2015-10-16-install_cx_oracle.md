---
layout: post
title:  "安装cx_oracle"
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
wget http://download.oracle.com/otn/linux/instantclient/121020/instantclient-sqlplus-linux.x64-12.1.0.2.0.zip
{% endhighlight %}

### 安装oracle

{% highlight python %}
yum -y install libaio bc flex # 可能的依赖
unzip instantclient-basic-linux.x64-12.1.0.2.0.zip -d /opt/
unzip instantclient-sdk-linux.x64-12.1.0.2.0.zip -d /opt/
unzip instantclient-sqlplus-linux.x64-12.1.0.2.0.zip -d /opt/ # sqlplus
cd /opt/instantclient_12_1
ln -s libclntsh.so.12.1 libclntsh.so # libclntsh.so
{% endhighlight %}


### 环境变量添加
{% highlight python %}
vim /etc/profile
#添加如下
export ORACLE_HOME=/opt/instantclient_12_1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ORACLE_HOME
export PATH=$ORACLE_HOME:$PATH
#生效
source /etc/profile
{% endhighlight %}

### 安装cx_oracle
{% highlight python %}
pip install cx_oracle
{% endhighlight %}

### 连接错误

"ORA-21561: OID generation failed"

解决方法：
配置客户端的hosts
{% highlight python %}
127.0.0.1 vagrant-centos65.vagrantup.com # 通过 hostname 命令获得
{% endhighlight %}

[官方网站]: http://www.oracle.com/technetwork/topics/linuxx86-64soft-092277.html