---
layout: post
title:  "memcached笔记"
date:   2015-12-18 09:05:31
categories: memcached
tags: memcached
---

* content
{:toc}


Free & open source, high-performance, distributed memory object caching system
自由、开源、高性能、分布式内存对象缓存系统

## 1，memcached 安装

### 安装编译工具

{% highlight python %}
yum install gcc make cmake autoconf libtool -y
{% endhighlight %}

### 编译memcached

memcached 依赖于 libevent 库

分别到 libevent.org 和 memcached.org 下载最新的 stable 版本(稳定版)

先编译 libevent ,再编译 memcached,编译 memcached 时要指定 libevent 的路径

{% highlight python %}
tar zxvf libevent-x-stable.tar.gz
cd l ibevent-x-stable
./configure --prefix=/usr/local/libevent
make && make install


tar zxvf memcached-x.tag.gz
cd memcached-x
./configure--prefix=/usr/local/memcached \
--with-libevent=/usr/local/libevent
# make && make install
{% endhighlight %}

在虚拟机下编译，可能遇到问题--虚拟机的时间不对，如下解决
{% highlight python %}
date -s 'yyyy-mm-dd hh:mm:ss'
clock -w # 把时间写入 cmos
{% endhighlight %}

### 启动mecached

一般启动方式
{% highlight python %}
 memcached -d -p 11211 -u memcached -m 64 -c 1024 -P /var/run/memcached/memcached.pid
{% endhighlight %}

参数说明：
{% highlight python %}
-p <num>      TCP port number to listen on (default: 11211) # 监听的TCP端口(默认: 11211)
-U <num>      UDP port number to listen on (default: 11211, 0 is off) # 监听的UDP端口(默认: 11211, 0表示不监听)
-s <file>     UNIX socket path to listen on (disables network support) # 用于监听的UNIX套接字路径（禁用网络支持）
-a <mask>     access mask for UNIX socket, in octal (default: 0700) # UNIX套接字访问掩码，八进制数字（默认：0700）
-l <ip_addr>  interface to listen on (default: INADDR_ANY, all addresses) # 监听的IP地址。（默认：INADDR_ANY，所有地址）
-d            run as a daemon # 作为守护进程来运行
-r            maximize core file limit # 最大核心文件限制
-u <username> assume identity of <username> (only when run as root) # 设定进程所属用户。（只有root用户可以使用这个参数）
-m <num>      max memory to use for items in megabytes (default: 64 MB) # 分配给memcached实例的内存，以MB为单位。（默认：64MB）
-M            return error on memory exhausted (rather than removing items) # 内存用光时报错，不删除数据
-c <num>      max simultaneous connections (default: 1024) # 最大并发连接数（默认：1024）
-k            lock down all paged memory.  Note that there is a
              limit on how much memory you may lock.  Trying to
              allocate more than that would fail, so be sure you
              set the limit correctly for the user you started
              the daemon with (not for -u <username> user;
              under sh this is done with 'ulimit -S -l NUM_KB'). # 锁定所有内存页
-v            verbose (print errors/warnings while in event loop) # 提示信息（在事件循环中打印错误/警告信息）
-vv           very verbose (also print client commands/reponses) # 详细信息（还打印客户端命令/响应）
-vvv          extremely verbose (also print internal state transitions) # 超详细信息（还打印内部状态的变化）
-h            print this help and exit # 显示帮助信息并退出
-i            print memcached and libevent license # 显示memcached和libevent的许可
-P <file>     save PID in <file>, only used with -d option # 保存进程ID到指定文件，只有在使用 -d 选项的时候才有意义
-f <factor>   chunk size growth factor (default: 1.25) # 不同slab class里面的chunk大小的增长倍率。（默认：1.25）
-n <bytes>    minimum space allocated for key+value+flags (default: 48) # chunk的最小空间（默认：48）chunk数据结构本身需要消耗48个字节，所以一个chunk实际消耗的内存是n+48
-L            Try to use large memory pages (if available). Increasing
              the memory page size could reduce the number of TLB misses
              and improve the performance. In order to get large pages
              from the OS, memcached will allocate the total item-cache
              in one large chunk. # 尝试使用大内存页
-D <char>     Use <char> as the delimiter between key prefixes and IDs.
              This is used for per-prefix stats reporting. The default is
              ":" (colon). If this option is specified, stats collection
              is turned on automatically; if not, then it may be turned on
              by sending the "stats detail on" command to the server. # 使用 <char> 作为前缀和ID的分隔符
-t <num>      number of threads to use (default: 4) # 使用的线程数（默认：4）
-R            Maximum number of requests per event, limits the number of
              requests process for a given connection to prevent
              starvation (default: 20) # 每个连接可处理的最大请求数（默认：20）
-C            Disable use of CAS # 禁用CAS（禁止版本计数，减少开销）
-b            Set the backlog queue limit (default: 1024) # 设置后台日志队列的长度（默认：1024）
-B            Binding protocol - one of ascii, binary, or auto (default) # 绑定协议 - 可能值：ascii,binary,auto（默认）
-I            Override the size of each slab page. Adjusts max item size
              (default: 1mb, min: 1k, max: 128m) # 重写每个数据页尺寸，调整数据项最大尺寸
{% endhighlight %}


## 2，memcached命令行

使用`telnet localhost 11211`连接服务

### 存储命令

格式：
{% highlight python %}
<command> <key> <flags> <exptime> <bytes> [<version>] 回车
<datablock> 填写值，回车
<status> 返回状态

# 说明
command     set 无论如何都进行存储
            add 只有数据不存在的时候存储
            replace 只有数据存在的时候进行替换
            append 往后追加
            prepend 往前追加
            cas 按版本号更改
key         字符串，小于250个字符，不包含空格和控制符
flags       客户端用来标识数据格式的数值，如json,xml等
exptime     存活时间 0 为永远，小于30天60*60*24*30为秒数，大于30天为unixtime
bytes       字节数，不包含\r\n
datablock   要存储的文本
{% endhighlight %}

示例：
{% highlight python %}
# add replace set 的区别
set name 1 0 6
python
STORED
add name 1 0 4
java
NOT_STORED
replace name 1 0 4
java
STORED
get name
VALUE name 1 4
java
END
set name 1 0 3
php
STORED
get name
VALUE name 1 3
php
END

# append 和 prepend
append name 1 0 2
xx
STORED
get name
VALUE name 1 5
phpxx
END
prepend name 1 0 2
aa
STORED
get name
VALUE name 1 7
aaphpxx
END

{% endhighlight %}

### 读取命令
{% highlight python %}
<command> <key1> [<key2> ...] 回车
VALUE <key1> <flags> <bytes> [<version>]
<datablock> 数据
VALUE <key2> <flags> <bytes> [<version>]
<datablock> 数据
END

command     get  普通查询
            gets  查询带版本号
{% endhighlight %}


示例:
{% highlight python %}
# get 普通查询
# gets 查询带版本的值
set age 1 0 2
10
STORED
get name age
VALUE name 1 7
aaphpxx
VALUE age 1 2
10
END
gets age name
VALUE age 1 2 16
10
VALUE name 1 7 14
aaphpxx
END

{% endhighlight %}

每次更改数据都会触发版本号自增
{% highlight python %}
set ve 1 0 2
tt
STORED
gets ve
VALUE ve 1 2 17
tt
END
replace ve 1 0 2
hh
STORED
gets ve
VALUE ve 1 2 18
hh
END
{% endhighlight %}

回过头，在说下`cas`命令，check and set 的缩写
只有版本号一致的情况下才能修改，否则返回EXISTS
设计意图：解决多个客户端同时修改一条记录的问题，防止使用已经修改了数据

演示：
{% highlight python %}
set test 1 0 2
aa
STORED
gets aa
END
gets test
VALUE test 1 2 19
aa
END
# 版本不一样，不能修改
cas test 1 0 2 18
bb
EXISTS
gets test
VALUE test 1 2 19
aa
END
# 版本号一样，可以修改
cas test 1 0 2 19
bb
STORED
gets test
VALUE test 1 2 20
bb
END
{% endhighlight %}


### 计数命令
{% highlight python %}
<command> <key> <int> 回车
<int>

command     incr  增加
            desc  减少
{% endhighlight %}

示例
{% highlight python %}
set num 1 0 1
1
STORED
incr num 8
9
get num
VALUE num 1 1
9
END
decr num 6
3
get num
VALUE num 1 1
3
END
# 不存在的不能
delete num
DELETED
incr num 1
NOT_FOUND
# 非数字的不能
incr name 2
CLIENT_ERROR cannot increment or decrement non-numeric value
{% endhighlight %}

### 删除命令
{% highlight python %}
delete <key>
DELETED
{% endhighlight %}
示例
{% highlight python %}
delete age
DELETED
{% endhighlight %}

### 其他命令

version 查看版本
quit 退出
echo command args |nc ip host # 快捷方式直接发送命令
flush_all [delay] # 立即或者指定时间后失效所有的元素，但并没有清空内存空间
flush_all 效果是导致所有更新时间早于 flush_all 所设定时间的项目，在被执行取回命令时命令被忽略


## 3，memcached 状态

### 状态命令 stats
{% highlight python %}
stats
STAT pid 1672  # 进程ID
STAT uptime 1576749 # 运行时间 单位：秒
STAT time 1450426057 # 当前unix时间
STAT version 1.4.4 # 版本号
STAT pointer_size 64 # 当前操作系统指针大小
# CPU占用
STAT rusage_user 59.326980 # 进程累计用户时间
STAT rusage_system 36.930385 # 进程累计系统时间
# 连接数
STAT curr_connections 10 # 当前打开着的连接数
STAT total_connections 38 # 从服务器开启后，曾经打开过的连接数
STAT connection_structures 12 # 服务器分配的连接构造函数
# 命中
STAT cmd_get 41 # get 命令总的请求次数
STAT get_hits 29 # get命中次数
STAT get_misses 12 # get未命中次数
STAT cmd_set 25 # set 命令总的请求次数
STAT cmd_flush 0 # flush 命令总的请求次数
STAT delete_misses 5
STAT delete_hits 4
STAT incr_misses 1
STAT incr_hits 4
STAT decr_misses 0
STAT decr_hits 2
STAT cas_misses 0
STAT cas_hits 1
STAT cas_badval 1
STAT auth_cmds 0
STAT auth_errors 0
STAT limit_maxbytes 67108864
STAT accepting_conns 1
STAT listen_disabled_num 0
STAT threads 4
STAT conn_yields 0
# 字节流量
STAT bytes 211 # 当存储内容所占字节数
STAT bytes_read 1922 # 网络读取字节数
STAT bytes_written 13718 # 写入字节数
STAT limit_maxbytes 67108864 # 存储时被允许使用的总字节数
# LRU 频率
STAT curr_items 3 # 当前存储内容数量
STAT total_items 20 # 启动以来存储过的总内容数
STAT evictions 0  # LRU释放对象数量
END
{% endhighlight %}

### 查看设置 stats settings
{% highlight python %}
stats settings
STAT maxbytes 67108864 # 最大字节数限制
STAT maxconns 1024 # 最大连接数
STAT tcpport 11211
STAT udpport 11211
STAT inter NULL
STAT verbosity 0
STAT oldest 0
STAT evictions on  # 是否启用LRU
STAT domain_socket NULL
STAT umask 700
STAT growth_factor 1.25  # 增长因子
STAT chunk_size 48 # key+value+flags 大小
STAT num_threads 4
STAT stat_key_prefix :
STAT detail_enabled no
STAT reqs_per_event 20
STAT cas_enabled yes
STAT tcp_backlog 1024
STAT binding_protocol auto-negotiate
STAT auth_enabled_sasl no
STAT item_size_max 1048576
END
{% endhighlight %}

### 数据项统计 stats items
{% highlight python %}
stats items
STAT items:1:number 3  # 该slab 中对象数，含过期对象
STAT items:1:age 1573203
STAT items:1:evicted 0
STAT items:1:evicted_nonzero 0
STAT items:1:evicted_time 0
STAT items:1:outofmemory 0
STAT items:1:tailrepairs 0
END
{% endhighlight %}

### 对象数量统计
{% highlight python %}
stats sizes
STAT 96 3 # size count
END
{% endhighlight %}

### 区块统计 stats slabs
{% highlight python %}
stats slabs
STAT 1:chunk_size 96 # chunk 大小
STAT 1:chunks_per_page 10922 # 每个page的chunk数量
STAT 1:total_pages 1 # 总page
STAT 1:total_chunks 10922 # 总的chunk
STAT 1:used_chunks 3 # 使用的chunk
STAT 1:free_chunks 2 # 剩余的chunk
STAT 1:free_chunks_end 10917 # 分完page 浪费chunk数量
STAT 1:mem_requested 225
STAT 1:get_hits 29
STAT 1:cmd_set 25
STAT 1:delete_hits 4
STAT 1:incr_hits 4
STAT 1:decr_hits 2
STAT 1:cas_hits 1
STAT 1:cas_badval 1
STAT active_slabs 1 # slab 数量
STAT total_malloced 1048512  # 总的内存
END
{% endhighlight %}

### 查看memcached 的key

* 使用 `stats items` 列出所有的slab id
* 使用 `stats cachedump id num` 来看存放的key num=0 为全部查看
* 使用 `get key` 获取值

## 4，memcached 的内存管理机制

### 基本概念：

* slab，是一个逻辑概念。它是在启动memcached实例的时候预处理好的，每个slab对应一个chunk size，也就是说不同slab有不同的chunk size。具体分配多少个slab由参数 -f （增长因子）和 -n （chunk最小尺寸）决定的。
* page，可以理解为内存页。大小固定为1m。slab会在存储请求时向系统申请page，并将page按chunk size进行切割。
* chunk，是保存用户数据的最小单位。用户数据item（包括key，value）最终会保存到chunk内。chunk规格是固定的，如果用户数据放进来后还有剩余则这剩余部分不能做其他用途

### 工作流程：

* memcahed实例启动，根据 -f 和 -n 进行预分配slab。以 -n 为最小值开始，以 -f 为比值生成等比数列，直到1m为止（每个slab的chunk size都要按8的倍数进行补全，比如：如果按比值算是556的话，会再加4到560成为8的整倍数）
* 每个slab分配一个page
* 当用户发来存储请求时（key,value），memcached会计算key+value的大小，看看属于哪个slab。
* 确定slab后看里面的是否有空闲chunk放key+value，
* 如果不够就再向系统申请一个page，申请后将该page按本slab的chunk size 进行切割，然后分配一个来存放用户数据
* 如果此时已经达到 -m 参数设置的内存使用上限，无法申请page，如果设置了 -M 则返回错误提示
* 否则按LRU算法删除改slab的数据。并不会找其他空闲slab

### 删除机制

* 惰性删除：

    * 当某个值过期后,并没有从内存删除, 因此,stats 统计时, curr_item 有其信息
    * 当某个新值去占用他的位置时,当成空 chunk 来占用.
    * 当 get 值时,判断是否过期,如果过期,返回空,并且清空, curr_item 就减少了

* LRU 删除机制：

    * memcached 用的 lru 删除机制
    * 当某个单元被请求时,维护一个计数器,通过计数器来判断最近谁最少被使用就把谁 t 出

### 注意事项

* chunk是在page里面划分的，而page固定为1m，所以chunk最大不能超过1m。
* chunk实际占用内存要加48B，因为chunk数据结构本身需要占用48B。
* chunk 大小为固定，如果item 不能填充满，则造成浪费，需要根据实际，定制chunk
* key 的限制为250B，value最大为1M
* 已分配出去的page不能回收
* 永久存储的数据，可能被T出去

### 举例子：

* 例子一：
如果有 100B 的内容要存,但 122 大小的仓库中的 chunk 满了，
并不会寻找更大的,如 144 的仓库来存储，
而是把 122 仓库的旧数据踢掉

* 例子二：
某个key设置的永久性的，但是所在的slab，所有的chunk分配完了，又无法获得新的page，
新来非永久的key1，如果key在LRU删除机制中，就被删除了

* 例子三：
测试用的，如果数据大小不集中，特别浪费内存。
比如，10000个key，分别分配从低到高的大小的value，会迅速用掉slab，内存其实没用用完的，但无法分配了。
哪怕都删除掉key，基本上每个slab都分配了一个page，真正需要多个page的slab却无法再申请了

* 例子四：
使用memcached python客户端存放好过1M数据失败

### 如何优化

* 根据实际需要调整 -n -f 参数
* 提前分析自己存储的数据大小
* 避免存放超过1M数据
* 尽量缓存小数据，省带宽，省内存
* 为数据大小区间小的数据，分配专用memcacehd实例，可以调小-f参数，减少内存浪费


## 5，memcached 分布式

### 取模

可以分散，但是增加节点和删除节点的时候，受影响的节点多

### 一致性HASH算法

添加节点和删除节点，只有该节点收到影响

增加虚拟节点，使得压力分配均匀

python的实现参考[hash_ring](https://github.com/Doist/hash_ring)

## 6，memcached 应用与问题

### 常见应用

作为缓存，减少DB的压力：

* 客户端请求数据，先从memcached获取
* 如果没有，则从DB获取，同时更新至memcached
* 如果数据有变化，更新DB的同时，更新至memcached

### 中继MYSQL主从延迟数据

mysql 主从复制的时候，有延迟，导致数据不统一。

考虑如下场景:

* 用户 U 购买电子书 B, `insert into Master (U,B)`
* 用户 U 观看电子书 B, `select 购买记录[user=’A’,book=’B’] from Slave`
* 由于主从延迟,第2步中无记录,用户无权观看该书

这时,可以利用 memached 在 master 与 slave 之间做过渡:

* 用户 U 购买电子书 B, `memcached->add(‘U:B’,true)`
* 主数据库 `insert into Master (U,B)`
* 用户 U 观看电子书 B, `select 购买记录[user=’U’,book=’B’] from Slave`.如果没查询到,则 `memcached->get(‘U:B’)`,查到则说明已购买但 Slave 延迟.
* 由于主从延迟,第2步中无记录,用户无权观看该书.

### 缓存雪崩

缓存雪崩一般是由某个缓存节点失效,导致其他节点的缓存命中率下降, 缓存中缺失的数据
去数据库查询.短时间内,造成数据库服务器崩溃

问题：缓存失效时间过于集中

解决：分散开来，随机到不同的时间段

### 缓存的无底洞现象 multiget-hole

该问题由 facebook 的工作人员提出的, facebook 在 2010 年左右,memcached 节点就已经达
3000 个.缓存数千 G 内容.

他们发现了一个问题---memcached 连接频率,效率下降了,于是加 memcached 节点,

添加了后,发现因为连接频率导致的问题,仍然存在,并没有好转,称之为”无底洞现象”

**问题：**

以用户为例: user-133-age, user-133-name,user-133-height .....N 个 key,

当服务器增多,133 号用户的信息,也被散落在更多的节点,

所以,同样是访问个人主页,得到相同的个人信息, 节点越多,要连接的节点也越多.

对于 memcached 的连接数,并没有随着节点的增多,而降低. 于是问题出现.

**解决：**

把某一组 key,按其共同前缀,来分布.

比如 user-133-age, user-133-name,user-133-height 这 3 个 key,

在用分布式算法求其节点时,应该以 ‘user-133’来计算,而不是以 user-133-age/name/height 来计算.

这样,3 个关于个人信息的 key,都落在同 1 个节点上,访问个人主页时,只需要连接 1 个节点.

### 永久数据被踢现象

由于LRU删除机制导致永久数据被删除，

解决为 分开存放永久数据与非永久数据


### Memcache mutex设计模式

Mutex主要用于有大量并发访问并存在cache过期的场合，如

* 首页top 10, 由数据库加载到memcache缓存n分钟
* 微博中名人的content cache, 一旦不存在会大量请求不能命中并加载数据库
* 需要执行多个IO操作生成的数据存在cache中, 比如查询db多次

问题在大并发的场合，当cache失效时，大量并发同时取不到cache，会同一瞬间去访问db并回设cache，可能会给系统带来潜在的超负荷风险。

**解决方法方法一:**

在load db之前先add一个mutex key, mutex key add成功之后再去做加载db,

如果add失败则sleep之后重试读取原cache数据。为了防止死锁，mutex key也需要设置过期时间。

**方法二:**

在value内部设置1个超时值(timeout1), timeout1比实际的memcache timeout(timeout2)小。

当从cache读取到timeout1发现它已经过期时候，马上延长timeout1并重新设置到cache。

然后再从数据库加载数据并设置到cache中


## memcached 监控和工具

* 主要自己使用命令行看
* [memcached-admin](https://github.com/ianare/django-memcache-admin)
* memcahced perl管理工具 memcached-tool
* 管理工具 [memcached-tool](http://libmemcached.org/libMemcached.html)
