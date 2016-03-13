---
layout: post
title:  "安装hadoop"
date:   2016-03-10 19:05:31
categories: hadoop
tags: hadoop
---

* content
{:toc}

## 安装

### 环境设置

设置虚拟机
:   选择桥接方式，方便与其他主机通信

给hadoop用户开放sudo 权限：
:   `su`切换到root权限，修改 `/etc/sudoers`增加`hadoop ALL=(ALL) ALL` 同root配置一样

修改为无界面模式，节约内存
{% highlight python %}
# 3 为无界面，多用户模式
vim /etc/inittab
id:3:initdefault:
{% endhighlight %}

修改主机名
{% highlight python %}
hostname hadoop01 # 即时生效，退出重新登录即可看到效果
vim /etc/sysconfig/network
NETWORKING=yes
HOSTNAME=hadoop01
{% endhighlight %}

修改hosts
{% highlight python %}
vim /etc/hosts
192.168.1.200 hadoop01

# 测试下
ping hadoop01
{% endhighlight %}

关闭防火墙
{% highlight python %}
#查看防火墙状态
service iptables status
#关闭防火墙
service iptables stop
#查看防火墙开机启动状态
chkconfig iptables --list
#关闭防火墙开机启动
chkconfig iptables off
{% endhighlight %}

数据传递
{% highlight python %}
yum install lrzsz
rz -b # choose your pc file
sz filename # get file to your pc
{% endhighlight %}

### 安装JDK

* 上传jdk-7u_65-i585.tar.gz到虚拟机
* 解压jdk `tar -zxvf jdk-7u55-linux-i586.tar.gz -C /home/hadoop/app`
* 将java添加到环境变量中

{% highlight python %}
vim /etc/profil
#在文件最后添加
export JAVA_HOME=/home/hadoop/app/jdk-7u_65-i585
export PATH=$PATH:$JAVA_HOME/bin
#刷新配置
source /etc/profile
{% endhighlight %}
* 运行`java` 命令验证

### 安装hadoop2.4.1

* 先上传hadoop的安装包到服务器上去/home/hadoop/

注意：hadoop2.x的配置文件$HADOOP_HOME/etc/hadoop

* 伪分布式需要修改5个配置文件

  * 配置hadoop

    {% highlight python %}
    vim hadoop-env.sh
    export JAVA_HOME=/home/hadoop/app/jdk1.7.0_65
    {% endhighlight %}

  * core-site.xml

    {% highlight python %}
    <!-- 指定HADOOP所使用的文件系统schema（URI），HDFS的老大（NameNode）的地址 -->
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://hadoop01:9000</value>
    </property>
    <!-- 指定hadoop运行时产生文件的存储目录 -->
    <property>
        <name>hadoop.tmp.dir</name>
        <value>/home/hadoop/app/hadoop-2.4.1/data</value>
    </property>
    {% endhighlight %}

  * hdfs-site.xml   hdfs-default.xml

    {% highlight python %}
    <!-- 指定HDFS副本的数量 -->
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
    {% endhighlight %}

  * mapred-site.xml

    {% highlight python %}
        mv mapred-site.xml.template mapred-site.xml
        vim mapred-site.xml
        <!-- 指定mr运行在yarn上 -->
        <property>
            <name>mapreduce.framework.name</name>
            <value>yarn</value>
        </property>
    {% endhighlight %}

  * yarn-site.xml

    {% highlight python %}
    <!-- 指定YARN的老大（ResourceManager）的地址 -->
    <property>
            <name>yarn.resourcemanager.hostname</name>
            <value>hadoop01</value>
    </property>
        <!-- reducer获取数据的方式 -->
    <property>
            <name>yarn.nodemanager.aux-services</name>
            <value>mapreduce_shuffle</value>
     </property>
     {% endhighlight %}

  * slaves datanodes

    {% highlight python %}
     hadoop01
    {% endhighlight %}


将hadoop添加到环境变量
{% highlight python %}
vim /etc/proflie
    export HADOOP_HOME=/home/haddoop/app/hadoop-2.4.1
    export PATH=$PATH:$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

source /etc/profile
{% endhighlight %}

### 格式化namenode（是对namenode进行初始化）
{% highlight python %}
    hdfs namenode -format (hadoop namenode -format)
{% endhighlight %}

### 启动hadoop
{% highlight python %}
    #先启动HDFS
    start-dfs.sh

    #再启动YARN
    start-yarn.sh
{% endhighlight %}

### hadoop shell操作

使用`hadoop fs` 可以操作shell，

例如： 上传本地文件到hdfs上

{% highlight python %}
    hadoop fs -put localfile hdfs://hadoop01:9000/hdfs_file # hdfs路径可以简写为/hdfs_file
{% endhighlight %}

其他命令参考
{% highlight python %}
[hadoop@hadoop01 ~]$ hadoop fs -usage
Usage: hadoop fs [generic options]
        [-appendToFile <localsrc> ... <dst>]
        [-cat [-ignoreCrc] <src> ...]
        [-checksum <src> ...]
        [-chgrp [-R] GROUP PATH...]
        [-chmod [-R] <MODE[,MODE]... | OCTALMODE> PATH...]
        [-chown [-R] [OWNER][:[GROUP]] PATH...]
        [-copyFromLocal [-f] [-p] <localsrc> ... <dst>]
        [-copyToLocal [-p] [-ignoreCrc] [-crc] <src> ... <localdst>]
        [-count [-q] <path> ...]
        [-cp [-f] [-p] <src> ... <dst>]
        [-createSnapshot <snapshotDir> [<snapshotName>]]
        [-deleteSnapshot <snapshotDir> <snapshotName>]
        [-df [-h] [<path> ...]]
        [-du [-s] [-h] <path> ...]
        [-expunge]
        [-get [-p] [-ignoreCrc] [-crc] <src> ... <localdst>]
        [-getfacl [-R] <path>]
        [-getmerge [-nl] <src> <localdst>]
        [-help [cmd ...]]
        [-ls [-d] [-h] [-R] [<path> ...]]
        [-mkdir [-p] <path> ...]
        [-moveFromLocal <localsrc> ... <dst>]
        [-moveToLocal <src> <localdst>]
        [-mv <src> ... <dst>]
        [-put [-f] [-p] <localsrc> ... <dst>]
        [-renameSnapshot <snapshotDir> <oldName> <newName>]
        [-rm [-f] [-r|-R] [-skipTrash] <src> ...]
        [-rmdir [--ignore-fail-on-non-empty] <dir> ...]
        [-setfacl [-R] [{-b|-k} {-m|-x <acl_spec>} <path>]|[--set <acl_spec> <path>]]
        [-setrep [-R] [-w] <rep> <path> ...]
        [-stat [format] <path> ...]
        [-tail [-f] <file>]
        [-test -[defsz] <path>]
        [-text [-ignoreCrc] <src> ...]
        [-touchz <path> ...]
        [-usage [cmd ...]]

Generic options supported are
-conf <configuration file>     specify an application configuration file
-D <property=value>            use value for given property
-fs <local|namenode:port>      specify a namenode
-jt <local|jobtracker:port>    specify a job tracker
-files <comma separated list of files>    specify comma separated files to be copied to the map reduce cluster
-libjars <comma separated list of jars>    specify comma separated jar files to include in the classpath.
-archives <comma separated list of archives>    specify comma separated archives to be unarchived on the compute machines.
{% endhighlight %}


### 验证是否启动成功
{% highlight python %}
    使用jps命令验证
    27408 NameNode
    28218 Jps
    27643 SecondaryNameNode
    28066 NodeManager
    27803 ResourceManager
    27512 DataNode

    http://192.168.1.200:50070 （HDFS管理界面）
    http://192.168.1.200:8088 （MR管理界面）
{% endhighlight %}

### 简单测试MR
{% highlight python %}
vim test.txt # 随便写点啥
hadoop fs -mkdir -p /test/input
hadoop fs -put test.txt /test/input
cd /home/hadoop/app/hadoop-2.4.1/share/hadoop/mapreduce # 例子目录
hadoop jar hadoop-mapreduce-examples-2.4.1.jar wordcount /test/input /test/output  # output 自动生成
hadoop fs -ls /test/output # 查看生成的目录
hadoop fs -cat /test/output/part-r-00000 # 查看结果
{% endhighlight %}

### HDFS的实现思想：

* hdfs是通过分布式集群来存储文件，为客户端提供一个便捷的访问方式，一个虚拟的目录结构
* 文件存储到hdfs集群中的时候被切分城若干block，block大小默认128M
* 文件的block存放到若干台datanade节点上
* hdfs 文件系统中的文件与真实block直接的映射关系，由namenode来管理
* 每一个block在集群总存储多个副本，好处是可以提高可靠行，还可以提高访问的吞吐量。副本数量配置在`hdfs-site.xml` 的`dfs.replication`中


## 配置ssh无密码登陆

### 启动服务的时候，需要输入密码，可以通过配置SSH免登陆来解决
{% highlight python %}
ssh-keygen -t rsa (默认回车继续到底)
生成 ~/.ssh/ 目录下 公钥(id_rsa.pub)和私钥(id_rsa)
将公钥拷贝到目标机器上

# 目标机器操作 假设拷贝到了 ~/id_rsa.pub
# 加入授权列表
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys  # 权限必须600
cat ~/id_rsa.pub >> ~/.ssh/authorized_keys

# 自己主机
ssh targethost # 这样就直接进入了
# 注意： 默认用户名是当前登录名，目标主机操作的用户也为同名用户，端口默认

# 否则 需要说明端口和用户 如
ssh -p 2222 imouren@targethost

# 根据上述原理 把自己的公钥加入到自己的认证列表
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# 验证
stop-dfs.sh # 停止dfs服务
stop-yarn.sh # 停止yarn服务
start-dfs.sh
start-yarn.sh  # 两个都没有要求输入密码
{% endhighlight %}

### SSH 免登陆过程： A 登陆 B

* A 生成公钥和私钥
* A 把 公钥 复制给B
* B 把 A的公钥 放入自己授权列表 `~/.ssh/authorized_keys`
* A 请求登陆 B， 携带自己用户名 和主机
* B 查看授权列表，是否有A公钥
* B 用A公钥加密一个字符串，发送给A
* A 用自己的私钥解密，并将解密结果发送给B
* B 验证 是否和自己的发送的字符串一致，
* 结果一致 B 发送给A 指令，通过验证




