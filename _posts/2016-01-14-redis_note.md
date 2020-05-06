---
layout: post
title:  "redis笔记"
date:   2016-01-14 09:05:31
categories: redis
tags: redis
---

* content
{:toc}



# 一、简介与安装

## 简介

redis 是开源的BSD许可的，在内存存储结构化数据，可以被用来作为数据库，缓存，消息中介。

redis 与 memcached 对比：

* redis 可以用作存储，memcached 只能用作缓存
* redis 存储的数据可以是结构化的，memcached只有string
* redis 可以做任务队列，订阅等
* redis 可以做主从复制

## 安装

下载最新的[stable源码](http://download.redis.io/redis-stable.tar.gz)


{% highlight python %}
yum install tcl # 依赖tcl
make  # 不用configure
make test  # 可选
make install  # 可以增加 PREFIX=/usr/local/redis  指定安装目录
{% endhighlight %}

安装后得到如下文件：

* redis-benchmark  性能测试工具
* redis-check-aof  日志文件检测工具
* redis-check-dump 快照文件检测工具
* redis-cli 客户端
* redis-server 服务端

## 启动

`redis-server`  直接启动 默认端口6379

`redis-server --port 6380`  指定端口启动

`redis-server redis.conf`  使用配置文件启动

修改配置文件 `daemonize yes` 以后台进程形式启动

redis 源码目录有 utils 文件夹，有`redis-init-script`脚本

可以拷贝到`/etc/init.d/`目录，并修改相关配置，方便管理redis

然后增加服务并开机自启动：
{% highlight python %}
# chmod 755 /etc/init.d/redis
# chkconfig --add redis
# chkconfig --level 345 redis on
# chkconfig --list redis
{% endhighlight %}

## 连接

redis-cli 是 Redis Command Line Interface 的缩写

两种方式发送命令

* 连接redis 通过-h 和 -p 参数
{% highlight python %}
$ redis-cli -h 127.0.0.1 -p 6379
$ redis-cli PING
PONG
{% endhighlight %}

* 通过交互方式
{% highlight python %}
$ redis-cli
redis 127.0.0.1:6379> PING
PONG
redis 127.0.0.1:6379> ECHO hi
"hi"
{% endhighlight %}

命令返回值：
{% highlight python %}
1，状态回复
显示状态信息
redis 127.0.0.1:6379> PING
PONG
2，错误回复
错误回复以(error)开头
redis 127.0.0.1:6379> SOMETHING
(error) ERR unknown command 'SOMETHING'
3，整数回复
以(integer)开头
redis 127.0.0.1:6379> INCR foo
(integer) 1
4，字符串回复
双引号包裹
redis 127.0.0.1:6379> GET foo
"1"
特殊
redis 127.0.0.1:6379> GET noexists
(nil)
5，多行字符串回复
redis 127.0.0.1:6379> KEYS *
1) "foo"
2) "bar"
{% endhighlight %}

## 关闭

`redis-cli SHUTDOWN`  使用客户端命令关闭更安全

结束进程
{% highlight python %}
ps -ef|grep redis
kill -9 pid
或者
pkill -9 redis
{% endhighlight %}


# 二、redis基础命令和数据结构

## 对key的操作命令

### 获得符合规则的键名列表

KYES pattern

pattern 支持glob风格通配符格式:

* `?` 匹配一个字符
* `*` 匹配任意个，包含0个 字符
* `[]` 匹配括号间的任一字符
* `\x`  匹配字符x，x为特殊字符，比如\? 匹配?

{% highlight python %}
redis 127.0.0.1:6379> set x 1
OK
redis 127.0.0.1:6379> keys *
1) "x"
redis 127.0.0.1:6379> keys x
1) "x"
{% endhighlight %}

### 判断一个键是否存在

EXISTS key
{% highlight python %}
redis 127.0.0.1:6379> exists x
(integer) 1
redis 127.0.0.1:6379> exists y
(integer) 0
{% endhighlight %}

### 删除键

DEL key [key ...]

删除一个或者多个键，返回值为删除键的个数

{% highlight python %}
redis 127.0.0.1:6379> del x
(integer) 1
redis 127.0.0.1:6379> del x
(integer) 0
{% endhighlight %}

DEL 命令的参数不支持通配符，可以使用linux通配符来操作

注意：`keys x*` 会严重影响性能 不要线上操作

{% highlight python %}
redis-cli KEYS "x*" | xargs redis-cli  DEL
# 如果需要站位，xargs 增加-i 参数
# 比如 设置一类key过期时间
redis-cli keys "db:poseidon:media:*"|xargs -i redis-cli expire {} 86400
或者
redis-cli  DEL  `redis-cli KEYS "x*"`

redis-cli --scan --pattern users:* | xargs redis-cli del
{% endhighlight %}

### 获得键值的数据类型

TYPE key

TYPE命令用来获得键值的数据类型，返回值可能是string （字符串类型）、hash （散列类
型）、list （列表类型）、set （集合类型）、zset （有序集合类型）

{% highlight python %}
redis 127.0.0.1:6379> set a 1
OK
redis 127.0.0.1:6379> type a
string
# LPUSH 命令的作用是向指定列表类型键中增加一个元素，如果键不存在则创建
redis 127.0.0.1:6379> lpush aa 1
(integer) 1
redis 127.0.0.1:6379> type aa
list
{% endhighlight %}

### 重命名

RENAME key newkey

注:如果newkey已存在,则newkey的原值被覆盖

RENAMENX key newkey

注: nx--> not exists, 即, newkey不存在时,作改名动作

### 转移DB

MOVE key db

默认打开16个数据库,从0到15编号,可以从配置文件修改

{% highlight python %}
127.0.0.1:6379[15]> keys *
(empty list or set)
127.0.0.1:6379[15]> set name hehe
OK
127.0.0.1:6379[15]> set age 23
OK
127.0.0.1:6379[15]> keys *
1) "age"
2) "name"
127.0.0.1:6379[15]> move name 14
(integer) 1
127.0.0.1:6379[15]> keys *
1) "age"
127.0.0.1:6379[15]> select 14
OK
127.0.0.1:6379[14]> get name
"hehe"
{% endhighlight %}

### 随机返回key

RANDOMKEY

{% highlight python %}
127.0.0.1:6379[14]> randomkey
"name"
{% endhighlight %}

### 判断key是否存在

EXISTS key

{% highlight python %}
127.0.0.1:6379[14]> exists name
(integer) 1
127.0.0.1:6379[14]> exists noname
(integer) 0
{% endhighlight %}

### 设置key的生命周期

TTL key  查看ttl

EXPIRE key num  设置生命周期（秒）

PERSIST key  设置永久

{% highlight python %}

127.0.0.1:6379[14]> ttl name
(integer) -1
127.0.0.1:6379[14]> expire name 600
(integer) 1
127.0.0.1:6379[14]> ttl name
(integer) 597
127.0.0.1:6379[14]> persist name
(integer) 1
127.0.0.1:6379[14]> ttl name
(integer) -1
{% endhighlight %}


## 字符串

### 赋值

SET key value [ex 秒数]/[px 毫秒数] [nx]/[xx]

nx 表示key不存在时，执行操作
xx 表示key 存在时，执行操作

如果 ex 和 px 同时写，以后面的有效期为准

例如： `set a 1 ex 10 px 9000` 实际有效期是9000毫秒

MSET key1 value1 key2 value2 一次性设置多个键值

例如： `mset key1 v1 key2 v2`


### 取值

GET key 获取key的值

MGET key1 key1 获取多个key

{% highlight python %}
127.0.0.1:6379[14]> mget key1 key2
1) "v1"
2) "v2"
{% endhighlight %}


### 其他的值操作

SETRANGE key offset value

把字符串的offset偏移字节改为value

如果偏移字节大于字符串长度，空缺部分，自动补`\x00`

{% highlight python %}
127.0.0.1:6379[14]> set name mouren
OK
127.0.0.1:6379[14]> get name
"mouren"
127.0.0.1:6379[14]> setrange name 0 x
(integer) 6
127.0.0.1:6379[14]> get name
"xouren"
127.0.0.1:6379[14]> setrange name 10 x
(integer) 11
127.0.0.1:6379[14]> get name
"xouren\x00\x00\x00\x00x"
{% endhighlight %}

APPEND key value

将value追加到key的原值后面

如果没有key，则设置key为value

{% highlight python %}
127.0.0.1:6379[14]> set name hello
OK
127.0.0.1:6379[14]> append name world
(integer) 10
127.0.0.1:6379[14]> get name
"helloworld"
127.0.0.1:6379[14]> get none
(nil)
127.0.0.1:6379[14]> append none nothing
(integer) 7
127.0.0.1:6379[14]> get none
"nothing"
{% endhighlight %}


GETRANGE key start stop

获取字符串中 start 到stop 范围的值
{% highlight python %}
127.0.0.1:6379[14]> getrange name 0 2
"hel"
127.0.0.1:6379[14]> getrange name  2 -1
"lloworld"
{% endhighlight %}


GETSET key value

返回旧值，并设置新值

{% highlight python %}
127.0.0.1:6379[14]> set name hello
OK
127.0.0.1:6379[14]> getset name world
"hello"
127.0.0.1:6379[14]> get name
"world"
{% endhighlight %}

STRLEN key

获得字符串长度

{% highlight python %}
127.0.0.1:6379[14]> get none
"nothing"
127.0.0.1:6379[14]> strlen none
(integer) 7
{% endhighlight %}


### 递增数字

INCR key

指定key的值加1，并返回加1后的值

{% highlight python %}
redis 127.0.0.1:6379> incr num
(integer) 1
redis 127.0.0.1:6379> get num
"1"
redis 127.0.0.1:6379> type num
string
redis 127.0.0.1:6379> incr num
(integer) 2
{% endhighlight %}

当操作的键不存在默认键值为0，所以第一次递增后结果为1.

当键值不是整数redis会提示错误

{% highlight python %}
redis 127.0.0.1:6379> set foo xx
OK
redis 127.0.0.1:6379> incr foo
(error) ERR value is not an integer or out of range
{% endhighlight %}

自己通过GET 和 SET 实现 incr 函数，会有问题

{% highlight python %}
def incr(key):
     value = GET key
     if not value:
          value = 0
     value += 1
     SET key, value
{% endhighlight %}

当多个客户端连接的时候，可能出现竞态条件。

比如A,客户端都操作同一个键，恰好均取到键值一样，比如“5”，各自将键值增加到“6”，最后得到的是“6”，而不是想要的“7”。

所有的redis命令均为原子操作，无论多少客户端同时连接，均不会出现上述情况。


INCRBY key num

指定key的值，加num

{% highlight python %}
127.0.0.1:6379[14]> set n 1
OK
127.0.0.1:6379[14]> incrby n 10
(integer) 11
127.0.0.1:6379[14]> get n
"11"
{% endhighlight %}

INCRBYFLOAT key floatnum

指定key值，增加float的数值

{% highlight python %}
127.0.0.1:6379[14]> set f 1.9
OK
127.0.0.1:6379[14]> incrbyfloat f 2.3
"4.2"
127.0.0.1:6379[14]> get f
"4.2"
{% endhighlight %}


DECR key

指定key的值，减少1

DECRBY key num

指定key的值，减少num

{% highlight python %}
127.0.0.1:6379[14]> get n
"11"
127.0.0.1:6379[14]> decr n
(integer) 10
127.0.0.1:6379[14]> decrby n 5
(integer) 5
127.0.0.1:6379[14]> get n
"5"
{% endhighlight %}


### 位操作

GETBIT key offset

SETBIT key offset value

BITCOUNT key [start] [end]

BITOP operation destkey key [key ...]


一个字节由8个二进制位组成

举例：
{% highlight python %}
127.0.0.1:6379> SET foo bar
OK
{% endhighlight %}

bar 的3个字母对应的ASCII码分别为98、97和114，转换为二进制分别为1100010、1100001、1110010

所以foo键的二进制位结构如下

![redis_bit.png](/files/redis_bit.png)

GETBIT命令可以获得一个字符串类型键指定二进制位的值（0或者1），索引从0开始。超出长度的默认位值为0



{% highlight python %}
127.0.0.1:6379> getbit foo 18
(integer) 1
127.0.0.1:6379> getbit foo 20
(integer) 0
127.0.0.1:6379> getbit foo 200
(integer) 0
{% endhighlight %}


SETBIT可以设置字符串类型键指定位置的二进制位的值，返回值是该位置旧的值

offset最大2^32-1,可推出最大的的字符串为512M

{% highlight python %}
127.0.0.1:6379> setbit foo 6 0
(integer) 1
127.0.0.1:6379> setbit foo 7 1
(integer) 0
127.0.0.1:6379> get foo
"aar"
{% endhighlight %}

如果设置位置超过键值的二进制长度，自动将中间位置设置为0

{% highlight python %}
127.0.0.1:6379> setbit nofoo 10 1
(integer) 0
127.0.0.1:6379> getbit nofoo 5
(integer) 0
127.0.0.1:6379> get nofoo
"\x00 "
{% endhighlight %}

BITCOUNT 命令可以获得字符串类型中值为1的二进制位个数
{% highlight python %}
127.0.0.1:6379> bitcount foo
(integer) 10
{% endhighlight %}

可以指定开始和结束的字节数 比如只统计'a' 或者'aa'
{% highlight python %}
127.0.0.1:6379> bitcount foo 0 0
(integer) 3
127.0.0.1:6379> bitcount foo 0 1
(integer) 6
{% endhighlight %}

BITOP可以对多个字符串类型键进行位运算，并将结果存储在destkey中。支持的运算操作包括AND,OR,XOR,NOT

注意: 对于NOT操作, key不能多个

{% highlight python %}
127.0.0.1:6379> set foo1 bar
OK
127.0.0.1:6379> set foo2 aar
OK
127.0.0.1:6379> bitop or res foo1 foo2
(integer) 3
127.0.0.1:6379> get res
"car"
{% endhighlight %}


## 列表类型

列表类型内部是使用双链表实现的。

LPUSH key value [value ...]

RPUSH key value [value ...]

向列表左、右插入元素

{% highlight python %}
127.0.0.1:6379[14]> lpush list a
(integer) 1
127.0.0.1:6379[14]> lpush list b c
(integer) 3
127.0.0.1:6379[14]> rpush list d
(integer) 4
127.0.0.1:6379[14]> lrange list 0 -1
1) "c"
2) "b"
3) "a"
4) "d"
{% endhighlight %}


LPOP key

RPOP key

从列表的左、右弹出元素

{% highlight python %}
127.0.0.1:6379[14]> rpop list
"d"
127.0.0.1:6379[14]> lpop list
"c"
{% endhighlight %}


LRANGE key start stop

获取列表的片段

LLEN key

获取元素的个数


{% highlight python %}
127.0.0.1:6379[14]> lrange list 0 -1
1) "b"
2) "a"
127.0.0.1:6379[14]> llen list
(integer) 2
{% endhighlight %}

LREM key count value

删除列表中指定的值

* 当 count＞0时LREM命令会从列表左边开始删除前count个值为value的元素；
* 当 count＜ 0时LREM 命令会从列表右边开始删除前count个值为value的元素；
* 当 count=0是LREM命令会删除所有值为value的元素

{% highlight python %}
127.0.0.1:6379[14]> rpush numbers 2 1 0 2 3 1 3 2
(integer) 8
127.0.0.1:6379[14]> lrange numbers 0 -1
1) "2"
2) "1"
3) "0"
4) "2"
5) "3"
6) "1"
7) "3"
8) "2"
# 右侧删除
127.0.0.1:6379[14]> lrem numbers -1 3
(integer) 1
127.0.0.1:6379[14]> lrange numbers 0 -1
1) "2"
2) "1"
3) "0"
4) "2"
5) "3"
6) "1"
7) "2"
# 全部删除
127.0.0.1:6379[14]> lrem numbers 0 1
(integer) 2
127.0.0.1:6379[14]> lrange numbers 0 -1
1) "2"
2) "0"
3) "2"
4) "3"
5) "2"
# 左侧删除
127.0.0.1:6379[14]> lrem numbers 2 2
(integer) 2
127.0.0.1:6379[14]> lrange numbers 0 -1
1) "0"
2) "3"
3) "2"
{% endhighlight %}


LINDEX key index

LSET key index value

获得和设置指定索引元素

{% highlight python %}
127.0.0.1:6379[14]> lrange numbers 0 -1
1) "0"
2) "3"
3) "2"
127.0.0.1:6379[14]> lindex numbers 1
"3"
127.0.0.1:6379[14]> lset numbers 1 8
OK
127.0.0.1:6379[14]> lindex numbers 1
"8"
{% endhighlight %}


LSTRIM key start end

只保留列表指定片段

{% highlight python %}
127.0.0.1:6379[14]> lrange numbers 0 -1
1) "0"
2) "8"
3) "2"
127.0.0.1:6379[14]> ltrim numbers 0 1
OK
127.0.0.1:6379[14]> lrange numbers 0 -1
1) "0"
2) "8"
{% endhighlight %}


LINSERT  key AFTER/BEFORE search value

从左到右查找search，找到的话插入value

一旦找到一个search后,命令就结束了,因此不会插入多个value

{% highlight python %}
127.0.0.1:6379[14]> lrange numbers 0 -1
1) "0"
2) "8"
127.0.0.1:6379[14]> linsert numbers after 3 9
(integer) -1
127.0.0.1:6379[14]> lrange numbers 0 -1
1) "0"
2) "8"
127.0.0.1:6379[14]> linsert numbers after 8 9
(integer) 3
127.0.0.1:6379[14]> lrange numbers 0 -1
1) "0"
2) "8"
3) "9"
{% endhighlight %}

RPOPLPUSH source destination

从source右边弹出元素，放入destination左侧

{% highlight python %}
127.0.0.1:6379[14]> lpush source 1 2 3 4 5 6 7
(integer) 7
127.0.0.1:6379[14]> rpoplpush source destination
"1"
127.0.0.1:6379[14]> rpoplpush source destination
"2"
127.0.0.1:6379[14]> lrange destination 0 -1
1) "2"
2) "1"
127.0.0.1:6379[14]> lrange source 0 -1
1) "7"
2) "6"
3) "5"
4) "4"
5) "3"
{% endhighlight %}

BRPOP key timeout

BLPOP key timeout

等待弹出key元素，timeout为超时时间，0 则一直等待

可应用到 长轮询ajax，在线聊天，缓存队列等

注意: LRANGE 会随着取数的变大导致性能急剧下降。

## HASH 类型

### 赋值和取值

HSET key field value

HGET key field

HMSET key field value [field value ...]

HMGET key field [field ...]

HGETALL key

{% highlight python %}
127.0.0.1:6379[14]> hset car price 50w
(integer) 1
127.0.0.1:6379[14]> hset car name bmw
(integer) 1
127.0.0.1:6379[14]> hget car name
"bmw"
127.0.0.1:6379[14]> hmget car name price
1) "bmw"
2) "50w"
127.0.0.1:6379[14]> hgetall
(error) ERR wrong number of arguments for 'hgetall' command
127.0.0.1:6379[14]> hgetall car
1) "price"
2) "50w"
3) "name"
4) "bmw"

{% endhighlight %}

### 字段操作

HEXISTS key field

判断字段是否存在

{% highlight python %}

127.0.0.1:6379[14]> hgetall car
1) "price"
2) "50w"
3) "name"
4) "bmw"
127.0.0.1:6379[14]> hexists car model
(integer) 0
127.0.0.1:6379[14]> hexists car name
(integer) 1
{% endhighlight %}

HSETNX key field value

当字段不存在时，赋值

{% highlight python %}

127.0.0.1:6379[14]> hgetall car
1) "price"
2) "50w"
3) "name"
4) "bmw"
127.0.0.1:6379[14]> hsetnx car name auto
(integer) 0
127.0.0.1:6379[14]> hsetnx car age 5
(integer) 1
{% endhighlight %}

HDEL key field [field ..]

删除字段

{% highlight python %}
127.0.0.1:6379[14]> hdel car name
(integer) 1
127.0.0.1:6379[14]> hgetall car
1) "price"
2) "50w"
3) "age"
4) "5"
{% endhighlight %}

HINCRBY key field num

增加字段值，若字段不存在则默认为0
{% highlight python %}
127.0.0.1:6379[14]> hincrby car length 30
(integer) 30
127.0.0.1:6379[14]> hincrby car length 3
(integer) 33
{% endhighlight %}

HINCRBYFLOAT key field floatnum

增加浮点值

{% highlight python %}
127.0.0.1:6379[14]> hincrbyfloat car age 1.2
"6.2"
{% endhighlight %}

HKEYS key

HVALS key

值获取 字段名 或者 值

{% highlight python %}
127.0.0.1:6379[14]> hkeys car
1) "price"
2) "age"
3) "length"

127.0.0.1:6379[14]> hvals car
1) "50w"
2) "5"
3) "33"
{% endhighlight %}


HLEN key

获取字段数量

{% highlight python %}
127.0.0.1:6379[14]> hlen car
(integer) 3
{% endhighlight %}

## 集合类型

集合是无序的、唯一的。最多存储2^32 - 1 个元素

内部使用值为空的hash实现，所以插入、删除、判断元素的时间复杂度为O(1)

SADD key number [number ...]

SREM key number [number ...]

增加或者删除元素

{% highlight python %}

127.0.0.1:6379[14]> sadd letters a
(integer) 1
127.0.0.1:6379[14]> sadd letters a b c
(integer) 2
127.0.0.1:6379[14]> smembers letters
1) "b"
2) "a"
3) "c"
127.0.0.1:6379[14]> srem letters c
(integer) 1
127.0.0.1:6379[14]> smembers letters
1) "b"
2) "a"
{% endhighlight %}

SMEMBERS key

返回集合中的元素

{% highlight python %}

127.0.0.1:6379[14]> smembers letters
1) "b"
2) "a"
{% endhighlight %}

SISMEMBER key number

判断是否存在某个元素

{% highlight python %}

127.0.0.1:6379[14]> sismember letters a
(integer) 1
127.0.0.1:6379[14]> sismember letters c
(integer) 0
{% endhighlight %}


SPOP key

随机删除一个元素

{% highlight python %}

127.0.0.1:6379[14]> smembers letters
1) "a"
2) "c"
3) "d"
4) "f"
5) "b"
6) "e"
127.0.0.1:6379[14]> spop letters
"f"
127.0.0.1:6379[14]> spop letters
"a"
{% endhighlight %}

SRANDMEMBER key

随机返回一个值

{% highlight python %}

127.0.0.1:6379[14]> smembers letters
127.0.0.1:6379[14]> srandmember letters
"c"
127.0.0.1:6379[14]> srandmember letters
"e"
{% endhighlight %}

SCARD key

集合含有元素个数

{% highlight python %}

127.0.0.1:6379[14]> smembers letters
1) "d"
2) "b"
3) "e"
4) "c"
127.0.0.1:6379[14]> scard letters
(integer) 4
{% endhighlight %}

SMOVE source destination value

将一个集合的值，移动到另外一个集合

{% highlight python %}
127.0.0.1:6379[14]> smembers letters
1) "d"
2) "b"
3) "e"
4) "c"
127.0.0.1:6379[14]> smove letters newletters d
(integer) 1
127.0.0.1:6379[14]> smembers newletters
1) "d"
127.0.0.1:6379[14]> smembers letters
1) "b"
2) "e"
3) "c"
{% endhighlight %}

### 集合操作

SDIFF key [key ...] 差集

SINTER key [key ...] 并集

SUNION key [key ...] 交集

SINTERSTORE dest key [key ...] 差集并存储到dest

SINTERSTORE key [key ...] 并集并存储到dest

SUNIONSTORE key [key ...] 交集并存储到dest

{% highlight python %}

127.0.0.1:6379[14]> sadd s1 1 2 3 4 5 6
(integer) 6
127.0.0.1:6379[14]> sadd s2 3 4 5 6 7
(integer) 5
127.0.0.1:6379[14]> sadd s3 5 6 7 8 9
(integer) 5
127.0.0.1:6379[14]> sdiff s1 s2
1) "1"
2) "2"
127.0.0.1:6379[14]> sdiff s1 s2 s3
1) "1"
2) "2"
127.0.0.1:6379[14]> sinter s1 s2
1) "3"
2) "4"
3) "5"
4) "6"
127.0.0.1:6379[14]> sinter s1 s2  s3
1) "5"
2) "6"
127.0.0.1:6379[14]> sunion s1 s2
1) "1"
2) "2"
3) "3"
4) "4"
5) "5"
6) "6"
7) "7"
127.0.0.1:6379[14]> sdiffstore s1diffs2 s1 s2
(integer) 2
127.0.0.1:6379[14]> smembers s1diffs2
1) "1"
2) "2"

{% endhighlight %}

## 有序集合类型

有序集合在集合类型的基础上为每个元素关联了一个分数

ZADD key score member [score member ...]

增加元素

重新add可以覆盖原来的分数，分数可以是float类型

{% highlight python %}
127.0.0.1:6379[14]> zadd ranking 88 tom 89 peter 100 edward
(integer) 3
127.0.0.1:6379[14]> zadd ranking 88.8 tom
(integer) 0
127.0.0.1:6379[14]> zscore ranking tom
"88.799999999999997"
{% endhighlight %}


ZSCORE key member

获得元素分数

{% highlight python %}
127.0.0.1:6379[14]> zscore ranking peter
"89"
{% endhighlight %}

ZRANGE key start stop [WITHSCORES]  # 从小到大顺序

ZREVRANGE key start stop [WITHSCORES]  # 逆序

获取排名在某个范围的元素

默认返回元素，可以增加 WITHSCORES 参数则连同分数一起返回

{% highlight python %}

127.0.0.1:6379[14]> zadd age 12 lily 20 tom 25 jerry 30 jack
(integer) 4
127.0.0.1:6379[14]> zrange age 0 3
1) "lily"
2) "tom"
3) "jerry"
4) "jack"
127.0.0.1:6379[14]> zrevrange age 0 3
1) "jack"
2) "jerry"
3) "tom"
4) "lily"
127.0.0.1:6379[14]> zrevrange age 0 3 withscores
1) "jack"
2) "30"
3) "jerry"
4) "25"
5) "tom"
6) "20"
7) "lily"
8) "12"

{% endhighlight %}


ZRANGEBYSCORE key min max [WITHSCORES] [LIMIT offset count]

获取指定分数范围的元素，包含min 和max

-inf和+inf分别表示负无穷和正无穷

如果希望分数范围不包含端点值， 可以在分数前加上“(”符号

ZREVRANGEBYSCORE key max min [WITHSCORES] [LIMIT offset count]

会得到与 ZRANGEBYSCORE 相反的排序，注意 max 和min的位置也是相反的

{% highlight python %}
127.0.0.1:6379[14]> zrangebyscore age 15 28
1) "tom"
2) "jerry"
127.0.0.1:6379[14]> zrangebyscore age 15 28 withscores
1) "tom"
2) "20"
3) "jerry"
4) "25"
127.0.0.1:6379[14]> zrangebyscore age 15 28 withscores limit 1
(error) ERR syntax error
127.0.0.1:6379[14]> zrangebyscore age 15 28 withscores limit  0 1
1) "tom"
2) "20"
# 获取年龄大于15的
127.0.0.1:6379[14]> zrangebyscore age 15 +inf
1) "tom"
2) "jerry"
3) "jack"
# 如果希望分数范围不包含端点值， 可以在分数前加上“(”符号
127.0.0.1:6379[14]> zrangebyscore age 12 28
1) "lily"
2) "tom"
3) "jerry"
127.0.0.1:6379[14]> zrangebyscore age (12 28
1) "tom"
2) "jerry"

# 相反的顺序
127.0.0.1:6379[14]> zrevrangebyscore age 28 15 withscores
1) "jerry"
2) "25"
3) "tom"
4) "20"
{% endhighlight %}



ZINCRBY key increment member

增加一个元素的分数，increment 也可以是float类型

如果指定的元素不存在， Redis在执行命令前会先建立它并将它的分数赋为0再执行操作

{% highlight python %}

127.0.0.1:6379[14]> zscore age tom
"20"
127.0.0.1:6379[14]> zincrby age 5 tom
"25"
127.0.0.1:6379[14]> zincrby age 5.5 tom
"30.5"
# 不存在
127.0.0.1:6379[14]> zincrby age 5 tom2
"5"
{% endhighlight %}

ZCARD key

获得有序集合元素数量

{% highlight python %}
127.0.0.1:6379[14]> zcard age
(integer) 5
{% endhighlight %}


ZCOUNT key min max

获得min到max直接元素的数量



{% highlight python %}
127.0.0.1:6379[14]> zrange age 0 5 withscores
 1) "tom2"
 2) "5"
 3) "lily"
 4) "12"
 5) "jerry"
 6) "25"
 7) "jack"
 8) "30"
 9) "tom"
10) "30.5"
127.0.0.1:6379[14]> zcount age 5 25
(integer) 3
{% endhighlight %}

ZREM key member [member ...]

删除元素

{% highlight python %}
127.0.0.1:6379[14]> zrem age tom2 tom
(integer) 2
127.0.0.1:6379[14]> zrange age 0 5 withscores
1) "lily"
2) "12"
3) "jerry"
4) "25"
5) "jack"
6) "30"
{% endhighlight %}


ZREMRANGEBYRANK key start stop

按照排名顺序删除元素

{% highlight python %}
127.0.0.1:6379[14]> zrange age 0 5 withscores
1) "lily"
2) "12"
3) "jerry"
4) "25"
5) "jack"
6) "30"
127.0.0.1:6379[14]> zremrangebyrank age 0 2
(integer) 3
127.0.0.1:6379[14]> zrange age 0 5 withscores
(empty list or set)
{% endhighlight %}


ZREMRANGEBYSCORE key min max

按照分数区间删除元素

{% highlight python %}
127.0.0.1:6379[14]> zadd age 10 tom 15 lily 20 kate 25 jerry
(integer) 4
127.0.0.1:6379[14]> zrange age 0 5
1) "tom"
2) "lily"
3) "kate"
4) "jerry"
127.0.0.1:6379[14]> zremrangebyscore age 10 15
(integer) 2

{% endhighlight %}


ZRANK key member  # 从小到大排名

ZREVRANK key member # 从大到小

获取元素排名（排名从0开始）

{% highlight python %}

127.0.0.1:6379[14]> zadd age 10 tom 15 lily 20 kate 25 jerry
(integer) 2
127.0.0.1:6379[14]> zrank age tom
(integer) 0
127.0.0.1:6379[14]> zrevrank age tom
(integer) 3
127.0.0.1:6379[14]> zrevrank age lily
(integer) 2

{% endhighlight %}


ZINTERSTORE destination numkeys key [key …] [WEIGHTS weight [weight …]] [AGREGATE SUM\MIN\MAX]

求交集；并集为 ZUNIONSTORE

结果存储在destination键中， 返回值为destination键中的元素个数

destination键中元素的分数是由 AGGREGATE参数决定的：

* SUM时（也就是默认值），参与计算的有序集合元素的和
* MINI时，参与计算的有序集合元素的最小值
* MAX时，参与计算的有序集合元素的最大值

ZINTERSTORE命令还能够通过WEIGHTS参数设置每个集合的权重，

每个集合在参与计算时元素的分数会被乘上该集合的权重

{% highlight python %}
# 默认sum
127.0.0.1:6379[14]> zadd z1 2 a 3 b 4 c
(integer) 3
127.0.0.1:6379[14]> zadd z2 5 a 6 b 7 c
(integer) 1
127.0.0.1:6379[14]> zinterstore z 2 z1 z2
(integer) 3
127.0.0.1:6379[14]> zrange z 0 10 withscores
1) "a"
2) "7"
3) "b"
4) "9"
5) "c"
6) "11"
# MAX
127.0.0.1:6379[14]> zadd z2 1 a
(integer) 0
127.0.0.1:6379[14]> zrange z2 0 5 withscores
1) "a"
2) "1"
3) "b"
4) "6"
5) "c"
6) "7"
7) "d"
8) "8"
127.0.0.1:6379[14]> zinterstore z 2 z1 z2 aggregate max
(integer) 3
127.0.0.1:6379[14]> zrange z 0 5  withscores
1) "a"
2) "2"
3) "b"
4) "6"
5) "c"
6) "7"
# MIN
127.0.0.1:6379[14]> zinterstore z 2 z1 z2 aggregate min
(integer) 3
127.0.0.1:6379[14]> zrange z 0 5 withscores
1) "a"
2) "1"
3) "b"
4) "3"
5) "c"
6) "4"
# WEIGHTS
127.0.0.1:6379[14]> zinterstore z 2 z1 z2 weights 1 2
(integer) 3
127.0.0.1:6379[14]> zrange z 0 5 withscores
1) "a"
2) "4"
3) "b"
4) "15"
5) "c"
6) "18"
{% endhighlight %}


# 三、特性

## 事务

Redis通过 MULTI、EXEC、WATCH 等命令实现事务功能。

事务一般经历三个阶段：

* 事务开始
* 命令入队
* 事务执行

MULTI 命令的执行意味着事务的开始

处于事务开启状态，后续的操作就会被放入队列（除了EXEC DISCARD WIATCH MULTI）

EXEC 执行事务


注意：

当检测到语法错误的时候，不执行任务队列的任何命令。

如果语法没有错误，实际命令错误，无法检查，则不执行该出错命令，其他命令执行

错误的避免，由程序员负责

{% highlight python %}
# 关注列表，与被关注列表
127.0.0.1:6379[14]> sadd user:1:following 2
QUEUED
127.0.0.1:6379[14]> sadd user:2:followers 2
QUEUED
127.0.0.1:6379[14]> exec
1) (integer) 1
2) (integer) 1

# 语法错误，不执行
127.0.0.1:6379[14]> multi
OK
127.0.0.1:6379[14]> set user mouren
QUEUED
127.0.0.1:6379[14]> set ue
(error) ERR wrong number of arguments for 'set' command
127.0.0.1:6379[14]> exec
(error) EXECABORT Transaction discarded because of previous errors.
127.0.0.1:6379[14]> get user
(nil)

# 命令错误，其他语句执行
127.0.0.1:6379[14]> multi
OK
127.0.0.1:6379[14]> set user mouren
QUEUED
127.0.0.1:6379[14]> sadd user xx
QUEUED
127.0.0.1:6379[14]> set user1 mouren1
QUEUED
127.0.0.1:6379[14]> exec
1) OK
2) (error) WRONGTYPE Operation against a key holding the wrong kind of value
3) OK
127.0.0.1:6379[14]> mget user user1
1) "mouren"
2) "mouren1"

{% endhighlight %}



WATCH 是个乐观锁，可以再EXEC命令执行前，监视任意数据库键

并在EXEC执行时，检查被监视的键是否改动过，如果改动过则拒绝执行事务

{% highlight python %}
127.0.0.1:6379[14]> set key 1
OK
127.0.0.1:6379[14]> watch key
OK
127.0.0.1:6379[14]> set key 2
OK
127.0.0.1:6379[14]> multi
OK
127.0.0.1:6379[14]> set key 3
QUEUED
127.0.0.1:6379[14]> exec
(nil)
127.0.0.1:6379[14]> get key
"2"
{% endhighlight %}


DISCARD 提前结束事务

{% highlight python %}
127.0.0.1:6379[14]> set key 1
OK
127.0.0.1:6379[14]> multi
OK
127.0.0.1:6379[14]> set key 2
QUEUED
127.0.0.1:6379[14]> discard
OK
127.0.0.1:6379[14]> exec
(error) ERR EXEC without MULTI
127.0.0.1:6379[14]> get key
"1"
{% endhighlight %}

UNWATCH 取消所有监控

执行EXEC命令后会取消对所有键的监控，

如果不想执行事务中的命令也可以使用UNWATCH命令来取消监控

比如实现类似HSETNX命令

{% highlight python %}

def hsetxx(key, field, value):
    WATCH key
    isFieldExists = HEXISTS key field
    if isFieldExists:
        MULTI
        HSET key field value
        EXEC
    else:
        UNWATCH
    return isFieldExists


{% endhighlight %}


## 发布/订阅消息

SUBSCRIBE 、 UNSUBSCRIBE 和 PUBLISH 三个命令实现了发布与订阅信息泛型

正在订阅频道的客户端不应该发送除 SUBSCRIBE 和 UNSUBSCRIBE 之外的其他命令。

其中， SUBSCRIBE 可以用于订阅更多频道， 而 UNSUBSCRIBE 则可以用于退订已订阅的一个或多个频道


订阅

SUBSCRIBE channel1 [channel2 ...]

{% highlight python %}
127.0.0.1:6379> subscribe foo bar
Reading messages... (press Ctrl-C to quit)
1) "subscribe"
2) "foo"
3) (integer) 1
1) "subscribe"
2) "bar"
3) (integer) 2
1) "message"
2) "foo"
3) "xxx"
1) "message"
2) "bar"
3) "barrrrr"
{% endhighlight %}

发布

PUBLISH channel message

{% highlight python %}
127.0.0.1:6379> publish foo xxx
(integer) 1
{% endhighlight %}

取消订阅 不带参数全部取消

UNSUBSCRIBE [channel1 ....]


Redis 的发布与订阅实现支持模式匹配（pattern matching）：

客户端可以订阅一个带 * 号的模式， 如果某个/某些频道的名字和这个模式匹配，

那么当有信息发送给这个/这些频道的时候， 客户端也会收到这个/这些频道的信息。

`redis> PSUBSCRIBE news.*`的客户端将收到来自 news.art.figurative 、 news.music.jazz 等频道的信息

客户端订阅的模式里面可以包含多个 glob 风格的通配符， 比如 * 、 ? 和 [...] ， 等等


订阅一个或多个符合给定模式的频道

PSUBSCRIBE pattern [pattern ...]

{% highlight python %}
127.0.0.1:6379> psubscribe fo*
Reading messages... (press Ctrl-C to quit)
1) "psubscribe"
2) "fo*"
3) (integer) 1
1) "pmessage"
2) "fo*"
3) "foo"
4) "222"
{% endhighlight %}

指示客户端退订所有给定模式

PUNSUBSCRIBE [pattern [pattern ...]]

注意：

* 使用 PUNSUBSCRIBE命令只能退订通过PSUBSCRIBE命令订阅的规则， 不会影响直接通过SUBSCRIBE命令订阅的频道；

* 同样UNSUBSCRIBE命令也不会影响通过PSUBSCRIBE命令订阅的规则

* PUNSUBSCRIBE命令退订某个规则时不会将其中 的通配符展开， 而是进行严格的字符串 匹配， 所以PUNSUBSCRIBE*
无法退订channel.*规则， 而是必须使用 PUNSUBSCRIBE channel.*才能退订


## 排序

SORT 排序命令

redis可以对列表类型、集合类型、有序集合类型进行排序

{% highlight python %}
127.0.0.1:6379[3]> lpush mylist 4 2 6 1 3 7
(integer) 6
127.0.0.1:6379[3]> sort mylist
1) "1"
2) "2"
3) "3"
4) "4"
5) "6"
6) "7"
{% endhighlight %}

有序列表会忽略掉分数

{% highlight python %}
127.0.0.1:6379[3]> zadd myzset 50 2 40 3 20 1 60 5
(integer) 4
127.0.0.1:6379[3]> sort myzset
1) "1"
2) "2"
3) "3"
4) "5"
{% endhighlight %}

通过ALPHA参数实现按照字典顺序排列非数字元素

{% highlight python %}

127.0.0.1:6379[3]> lpush strings  a c d e f B C A
(integer) 8
127.0.0.1:6379[3]> sort strings
(error) ERR One or more scores can't be converted into double
127.0.0.1:6379[3]> sort strings alpha
1) "a"
2) "A"
3) "B"
4) "c"
5) "C"
6) "d"
7) "e"
8) "f"
{% endhighlight %}

SORT命令默认是按照从小到大的顺序排列，可以通过DESC参数改为从大到小

{% highlight python %}
127.0.0.1:6379[3]> sort mylist desc
1) "7"
2) "6"
3) "4"
4) "3"
5) "2"
6) "1"

{% endhighlight %}

SORT命令还支持LIMIT参数来返回指定范围的结果

用法和SQL语句一样， LIMIT offset count

{% highlight python %}
127.0.0.1:6379[3]> sort mylist desc limit 1 3
1) "6"
2) "4"
3) "3"

{% endhighlight %}

BY 参数

很多时候列表存储的元素代表的对象的ID，单纯排序无意义。

我们需要根据ID代表的对象的属性进行排序。

BY 参数的语法为“BY参考键”。

其中参考键可以是字符串类型键或者是散列类型键的某个字段（表示为键名 -＞字段名 ）

如果提供了BY参数， SORT命令将不再依据元素自身的值进行排序，

而是对每个元素使用元素的值替换参考键中的第一个“＊ ”并获取其值，

然后依据该值对元素排序


例如：
{% highlight python %}
127.0.0.1:6379[3]> lpush ids 2 1 3
(integer) 3
127.0.0.1:6379[3]> set itemscore:1 50
OK
127.0.0.1:6379[3]> set itemscore:2 100
OK
127.0.0.1:6379[3]> set itemscore:3 -10
OK
127.0.0.1:6379[3]> sort ids by itemscore:* desc
1) "2"
2) "1"
3) "3"
{% endhighlight %}

当参考键不含有 * 的时候，不会进行排序操作
{% highlight python %}
127.0.0.1:6379[3]> sort ids
1) "1"
2) "2"
3) "3"
127.0.0.1:6379[3]> sort ids by xx
1) "3"
2) "1"
3) "2"
{% endhighlight %}


如果几个元素的参考键值相同， 则SORT命令会再比较元素本身的值来决定元素的顺序


{% highlight python %}
127.0.0.1:6379[3]> lpush ids 4
(integer) 4
127.0.0.1:6379[3]> set itemscore:4 50
OK
127.0.0.1:6379[3]> sort ids by itemscore:* desc
1) "2"
2) "4"
3) "1"
4) "3"
{% endhighlight %}

当某个元素的参考键不存在时， 会默认参考键的值为0

{% highlight python %}
127.0.0.1:6379[3]> lpush ids 5
(integer) 5
127.0.0.1:6379[3]> sort ids by itemscore:* desc
1) "2"
2) "4"
3) "1"
4) "5"
5) "3"
{% endhighlight %}

参考键虽然支持散列类型， 但是"*"只能在 "->"符号前面（即键名 部分）才有用，

在 "->"后（即字段名部分） 会被当 成字段名 本身而不会作为占位符被元素的值替換， 即常量键名

{% highlight python %}
127.0.0.1:6379[3]> hset obj3 age -18
(integer) 1
127.0.0.1:6379[3]> hset obj2 age 100
(integer) 0
127.0.0.1:6379[3]> hset obj1 age 10
(integer) 0
127.0.0.1:6379[3]> lpush objs 1 2 3
(integer) 3
127.0.0.1:6379[3]> sort objs by obj*->age
1) "3"
2) "1"
3) "2"

# 相当于取obj1 obj2 obj3 的 ag* 属性，结果没有取到
127.0.0.1:6379[3]> sort objs by obj*->ag*
1) "1"
2) "2"
3) "3"
{% endhighlight %}


GET 参数

GET 参数不影响排序，使得SORT命令的返回结果不再是元素的自身值，而是GET参数中指定的值



{% highlight python %}
127.0.0.1:6379> hmset obj1 name obj1 value 10
OK
127.0.0.1:6379> hmset obj2 name obj2 value 100
OK
127.0.0.1:6379> hmset obj3 name obj3 value 1
OK
127.0.0.1:6379> lpush objs 1 2 3
(integer) 3
127.0.0.1:6379> sort objs by obj*->value get obj*->name
1) "obj3"
2) "obj1"
3) "obj2"
{% endhighlight %}

GET 参数可以有多个

{% highlight python %}
127.0.0.1:6379> sort objs by obj*->value get obj*->name get obj*->value
1) "obj3"
2) "1"
3) "obj1"
4) "10"
5) "obj2"
6) "100"
{% endhighlight %}

GET #会返回元素本身的值

{% highlight python %}
127.0.0.1:6379> sort objs by obj*->value get obj*->name get obj*->value get #
1) "obj3"
2) "1"
3) "3"
4) "obj1"
5) "10"
6) "1"
7) "obj2"
8) "100"
9) "2"
{% endhighlight %}

STORE 参数

默认情况，sort会返回结果，使用STORE参数可以放置到一个key里面

保存后的键的类型为列表类型， 如果键已经存在则会覆盖它。

加上STORE参数后SORT命令的返回值为结果的个数

{% highlight python %}
127.0.0.1:6379> sort objs by obj*->value get obj*->name get obj*->value get # store storekey
(integer) 9
127.0.0.1:6379> lrange storekey 0 -1
1) "obj3"
2) "1"
3) "3"
4) "obj1"
5) "10"
6) "1"
7) "obj2"
8) "100"
9) "2"
{% endhighlight %}




# 四、系统相关

## 常用系统命令

显示服务器时间，时间戳的秒数、微妙数

TIME

{% highlight python %}
127.0.0.1:6379> time
1) "1454035871"
2) "898820"
{% endhighlight %}

当前数据库的key的数量

DBSIZE

{% highlight python %}
127.0.0.1:6379> keys *
1) "name"
2) "a"
127.0.0.1:6379> dbsize
(integer) 2
{% endhighlight %}

认证

AUTH passwd

配置文件中 requirepass 参数配置密码

如 requirepass passwd

{% highlight python %}
127.0.0.1:6379[3]> config set requirepass passwd
OK
127.0.0.1:6379[3]> get fo
(error) NOAUTH Authentication required.
127.0.0.1:6379[3]> auth passwd
OK
127.0.0.1:6379[3]> get fo
(nil)
{% endhighlight %}

选择数据库

SELECT index

{% highlight python %}
127.0.0.1:6379> dbsize
(integer) 2
127.0.0.1:6379> select 3
OK
127.0.0.1:6379[3]> dbsize
(integer) 0
{% endhighlight %}

INFO

查看信息

{% highlight python %}
127.0.0.1:6379> info
# Server
redis_version:2.8.19
redis_git_sha1:00000000
redis_git_dirty:0
redis_build_id:baa5f1acbdf97a21
redis_mode:standalone
os:Linux 2.6.32-431.3.1.el6.x86_64 x86_64
arch_bits:64
multiplexing_api:epoll
gcc_version:4.4.7
process_id:9615
run_id:e558cca431cc853faba172c2111c998a82374abd
tcp_port:6379
uptime_in_seconds:3340
uptime_in_days:0
hz:10
lru_clock:11196739
config_file:

# Clients
connected_clients:3
client_longest_output_list:0
client_biggest_input_buf:0
blocked_clients:0

# Memory
used_memory:852848
used_memory_human:832.86K
used_memory_rss:7954432
used_memory_peak:852848
used_memory_peak_human:832.86K
used_memory_lua:35840
mem_fragmentation_ratio:9.33
mem_allocator:jemalloc-3.6.0

# Persistence
loading:0
rdb_changes_since_last_save:2
rdb_bgsave_in_progress:0
rdb_last_save_time:1454033975
rdb_last_bgsave_status:ok
rdb_last_bgsave_time_sec:-1
rdb_current_bgsave_time_sec:-1
aof_enabled:0
aof_rewrite_in_progress:0
aof_rewrite_scheduled:0
aof_last_rewrite_time_sec:-1
aof_current_rewrite_time_sec:-1
aof_last_bgrewrite_status:ok
aof_last_write_status:ok

# Stats
total_connections_received:3
total_commands_processed:15
instantaneous_ops_per_sec:0
total_net_input_bytes:346
total_net_output_bytes:1470
instantaneous_input_kbps:0.00
instantaneous_output_kbps:0.00
rejected_connections:0
sync_full:0
sync_partial_ok:0
sync_partial_err:0
expired_keys:0
evicted_keys:0
keyspace_hits:1
keyspace_misses:0
pubsub_channels:0
pubsub_patterns:0
latest_fork_usec:0

# Replication
role:master
connected_slaves:0
master_repl_offset:0
repl_backlog_active:0
repl_backlog_size:1048576
repl_backlog_first_byte_offset:0
repl_backlog_histlen:0

# CPU
used_cpu_sys:0.83
used_cpu_user:4.63
used_cpu_sys_children:0.00
used_cpu_user_children:0.00

# Keyspace
db0:keys=2,expires=0,avg_ttl=0
{% endhighlight %}



BGREWRITEAOF 后台进程重写AOF

BGSAVE       后台保存rdb快照

SAVE         保存rdb快照

LASTSAVE     上次保存时间

SLAVEOF host port   运行时动态地修改复制(replication)功能

FLUSHALL  清空所有库所有键

FLUSHDB  清空当前库所有键

SHOWDOWN [save/nosave] 关闭所有服务器 安全的关闭服务

* 停止所有客户端
* 如果有至少一个保存点在等待，执行 SAVE 命令
* 如果 AOF 选项被打开，更新 AOF 文件
* 关闭 redis 服务器(server)


CONFIG GET parameter 获取配置参数信息

CONFIG SET parameter value 动态调整配置参数，无须重启

CONFIG REWRITE  重新写入到配置文件 要求版本大于2.8

### 不小心flushall后，如何补救

如果不小心运行了flushall, 立即 shutdown nosave ,关闭服务器

然后 手工编辑aof文件, 去掉文件中的 “flushall ”相关行, 然后开启服务器,就可以导入回原来数据.

如果,flushall之后,系统恰好bgrewriteaof了,那么aof就清空了,数据丢失



## 监视器

MONITOR 命令使得客户端成为一个监视器，实时接收并打印服务器当前处理命令请求信息

每当一个客户端向服务器发送一条命令的时候，服务器除了处理命令外，还将命令请求信息发给所有监视器

监视器
{% highlight python %}
127.0.0.1:6379> monitor
OK
1454035240.579372 [0 127.0.0.1:40395] "ping"
1454035250.168335 [0 127.0.0.1:40395] "set" "name" "hehe"
1454035251.736271 [0 127.0.0.1:40395] "get" "name"
1454035264.867648 [0 127.0.0.1:40395] "keys" "*"
1454035267.664166 [0 127.0.0.1:40395] "dbsize"
{% endhighlight %}

客户端请求
{% highlight python %}
127.0.0.1:6379> ping
PONG
127.0.0.1:6379> set name hehe
OK
127.0.0.1:6379> get name
"hehe"
127.0.0.1:6379> keys *
1) "name"
2) "a"
127.0.0.1:6379> dbsize
(integer) 2
127.0.0.1:6379> wrong command
(error) ERR unknown command 'wrong'
127.0.0.1:6379>
{% endhighlight %}


当一个客户端从普通客户端变为监视器时，该客户端的REDIS_MONITOR标识会被打开

服务器将所有监视器都记录在monitors链表中

每次处理命令请求时，服务器都会遍历monitors链表，将相关信息发送给监视器


## 慢查询日志

慢查询日志记录执行时间超过指定时长的命令请求，用户可以据此监控和优化查询速度

跟慢查询日志相关的配置：

* slowlog-log-slower-than 指定执行时间超过多少【微秒】的命令请求会记录到日志

* slowlog-max-len 执行服务器最多保存多少条慢查询日志

所有的慢查询日志保持在slowlog链表中，每个链表节点都包含一个slowlogEntry结果，每个slowlogEntry结构代表一条慢查询日志

打印和删除慢查询日志可以通过遍历slowlog链表完成

slowlog链表的长度就是服务器保存慢查询日志的数量

新的慢查询日志添加到slowlog链表的表头，如果日志数量超过slowlog-max-len，多出的日志会被删除


SLOWLOG GET [num] 获取慢查询日志

SLOWLOG LEN 获取当前慢查询日志数量

SLOWLOG RESET  清空慢查询日志


{% highlight python %}
127.0.0.1:6379[3]> config set slowlog-log-slower-than 0
OK
127.0.0.1:6379[3]> config set slowlog-max-len 4
OK
127.0.0.1:6379[3]> set name mouren
OK
127.0.0.1:6379[3]> set age 18
OK
127.0.0.1:6379[3]> set addr beijing
OK
127.0.0.1:6379[3]> set phone 110
OK
127.0.0.1:6379[3]> slowlog get
1) 1) (integer) 5
   2) (integer) 1454048497
   3) (integer) 6
   4) 1) "set"
      2) "phone"
      3) "110"
2) 1) (integer) 4
   2) (integer) 1454048484
   3) (integer) 5
   4) 1) "set"
      2) "addr"
      3) "beijing"
3) 1) (integer) 3
   2) (integer) 1454048476
   3) (integer) 6
   4) 1) "set"
      2) "age"
      3) "18"
4) 1) (integer) 2
   2) (integer) 1454048472
   3) (integer) 6
   4) 1) "set"
      2) "name"
      3) "mouren"
127.0.0.1:6379[3]> set newname mouren11
OK
127.0.0.1:6379[3]> slowlog get
1) 1) (integer) 7
   2) (integer) 1454048533
   3) (integer) 7
   4) 1) "set"
      2) "newname"
      3) "mouren11"
2) 1) (integer) 6
   2) (integer) 1454048516
   3) (integer) 15
   4) 1) "slowlog"
      2) "get"
3) 1) (integer) 5
   2) (integer) 1454048497
   3) (integer) 6
   4) 1) "set"
      2) "phone"
      3) "110"
4) 1) (integer) 4
   2) (integer) 1454048484
   3) (integer) 5
   4) 1) "set"
      2) "addr"
      3) "beijing"

# 本身获取或者清空的命令也被保存到了慢查询日志
127.0.0.1:6379[3]> slowlog len
(integer) 4
127.0.0.1:6379[3]> slowlog reset
OK
127.0.0.1:6379[3]> slowlog len
(integer) 1
127.0.0.1:6379[3]> slowlog get
1) 1) (integer) 11
   2) (integer) 1454048715
   3) (integer) 4
   4) 1) "slowlog"
      2) "len"
2) 1) (integer) 10
   2) (integer) 1454048713
   3) (integer) 4
   4) 1) "slowlog"
      2) "reset"
{% endhighlight %}


## redis的过期键删除策略

键过期，一般有三种不同的删除策略：

* 定时删除：在设置键过期的时间的同时，创建一个定时器，让定时器在键的过期时间来临时，立刻执行对键的删除操作。

* 惰性删除：放任键过期不管，但每次获取键的值的时候，检测键是否过期，如果过期，就删除，不过期就返回值。

* 定期删除： 每隔一段时间，程序对数据库进行一次检查，删除里面过期的键。

定期删除在redis.conf 的hz选项，默认为10 （即1秒执行10次，100ms一次）


redis服务器使用的是惰性删除和定期删除两种策略。

如果用的是主从，读写分离的话。主expire，从get并没有检测键是否过期。这点要注意下。（3.2版本解决这个问题了）

参考
http://blog.csdn.net/u012538947/article/details/52540313

## AOF、RDB和复制功能对键的过期处理

### RDB 文件保存

执行SAVE命令或者BGSAVE命令会创建一个新的RDB文件，程序会对键进行检查，已过期的键，不会保存在RDB文件中。


### RDB文件载入

启动redis服务器时，如果服务器开启了RDB功能，那么服务器将对RDB文件进行载入：

* 如果服务器以主服务器运行，载入RDB文件时，会对文件中保存的键进行检查，只载入微过期键

* 如果服务器已从服务器模式运行，载入RDB文件时，会全部载入。当从服务器进行数据同步的时候，从服务器的数据库会被清空，所以全部载入不会造成影响。


### AOF文件写入

AOF 写入时，如果键过期还没有被删除，对AOF文件不会产生影响。

当过期键被惰性删除或者定期删除后，程序会向AOF文件追加一条DEL命令

### AOF 重写

程序会对键进行检查，已过期键不会保存到重写的AOF文件中。

### 复制

当服务器处于复制模式下，从服务器的过期键删除动作由主服务器控制：

* 当主服务器删除一个过期键后，显示的向所有从服务器发送一个DEL命令，告知从服务器删除这个过期键

* 从服务器在执行客户端发送的读命令时，即使碰到过期键也不会删除

* 从服务器只有接受主服务器发送的DEL命令后，才会删除过期键


实测： 主服务器在键过期后，会及时通知从服务器，发送DEL命令。


{% highlight python %}
#主服务器
127.0.0.1:6379> set yyy ssss ex 10
OK
# 从服务器
127.0.0.1:7777> monitor
OK
1454313407.538669 [0 127.0.0.1:6379] "PING"
1454313410.980391 [0 127.0.0.1:6379] "set" "yyy" "ssss" "ex" "10"
1454313417.694862 [0 127.0.0.1:6379] "PING"
1454313421.074451 [0 127.0.0.1:6379] "DEL" "yyy"
{% endhighlight %}


## 持久化

### RDB持久化

可以将redis在内存中的数据库状态保存到磁盘里面

手动运行 `SAVE`或者 `BGSAVE`命令可以触发RDB持久化文件生成

SAVE 命令会阻塞redis服务器进行，直到RDB文件创建完毕，阻塞期间，服务器不处理任何命令请求。

BGSAVE 命令会派生一个子进程，然后由子进程负责创建RDB文件，服务器进程继续处理命令请求


RDB文件的载入是在服务器启动时，自动执行的。

但是如果服务器开启了AOF持久化功能，则优先载入AOF。

只有AOF关闭时，才使用RDB文件还原数据

持久化命令的冲突：

* SAVE 执行期间，不执行其他命令
* BGSAVE 执行期间，拒绝SAVE命令和BGSAVE命令
* BGSAVE 执行期间，BGREWRITEAOF命令会延迟到BGSAVE命令执行完毕
* BGREWRITEAOF执行期间，拒绝BGSAVE命令

Rdb快照的配置选项

* save 900 1      // 900内,有1条写入,则产生快照
* save 300 1000   // 如果300秒内有1000次写入,则产生快照
* save 60 10000  // 如果60秒内有10000次写入,则产生快照
* 这3个选项都屏蔽,则rdb禁用
* stop-writes-on-bgsave-error yes  // 后台备份进程出错时,主进程是否停止写入
* rdbcompression yes    // 导出的rdb文件是否压缩
* rdbchecksum   yes //  导入rbd恢复时数据时,要不要检验rdb的完整性
* dbfilename dump.rdb  //导出来的rdb文件名
* dir ./  //rdb的放置路径


### AOF持久化

AOF保存的命令是以redis的命令请求协议格式保存的

BGREWRITEAOF命令 触发AOF 文件重写功能，解决AOF文件体积过大问题

AOF重写过程：

在执行BGREWRITEAOF命令时，redis服务器会维护一个AOF缓冲区，该缓冲区会在子进程创建新AOF期间记录服务器执行的所有命令。

当子进程完成创建新的AOF文件的工作后，服务器会将缓冲区的所有内容追加到新AOF文件末尾，使得新旧两个AOF文件所保存的数据库状态一致。

最后服务器用新的AOF文件替换旧的AOF文件，完成AOF文件重写操作。

![AOF](/files/redis_aof.png)

Aof 的配置：

* appendonly no # 是否打开 aof日志功能

* appendfsync always   # 每1个命令,都立即同步到aof. 安全,速度慢

* appendfsync everysec # 折衷方案,每秒写1次

* appendfsync no      # 写入工作交给操作系统,由操作系统判断缓冲区大小,统一写入到aof. 同步频率低,速度快,

* no-appendfsync-on-rewrite  yes # 正在导出rdb快照的过程中,要不要停止同步aof

* auto-aof-rewrite-percentage 100 #aof文件大小比起上次重写时的大小,增长率100%时,重写

* auto-aof-rewrite-min-size 64mb #aof文件,至少超过64M时,重写

## 虚拟内存

使用VM把不经常访问到的数据交换到磁盘上，腾出内存空间用于其他需要访问的数据。

注意几个地方：

1，redis的VM在设计上为了保证key的查找速度，只会将value交换等我swap文件中。

所以，内存问题由于太多的value很小的key造成的，无法解决。可以选择list、set等数据结构解决。

2，VM解决部分数据的访问，而不全部。

如果经常需要访问全部数据，redis服务器会结束掉阻塞的客户端，从磁盘拿数据。

3，dump数据会变的非常慢，因为redis需要从磁盘获取数据，以便保存到RDB文件中。

可以采用AOF做持久化

4，VM会影响复制，因为当新的slave连接的时候，master需要执行BGSAVE保存RDB




相关配置：

vm-enabled yes # 开启VM功能

vm-swap-file /tmp/redis.swap # 交换出来的value的保存的文件路径

vm-max-memory 1000000 # redis使用的最大内存上限，超过上线后redis开始交换value到磁盘文件

vm-page-size 32  # 每个页面的大小32个字节

vm-pages 134217728  # 最多在文件中使用多数页面，交换文件大小=vm-page-size*vm-pages

vm-max-threads 4 # 用于执行value对象换入换出的工作线程数量。0则交换过程在主进程进行，会阻塞客户端请求，大于0推荐设置为服务器核数



# 五、多数据库相关


## 复制

执行SLAVEOF 命令或者 设置slaveof 选项，让一个服务器去复制另一个服务器

被复制的服务器叫 master ，进行复制的服务器叫slave

2.8版本以前使用旧的复制

复制功能分为同步和命令传播两个操作：

* 同步操作用于将从服务器的数据库状态更新至主服务器所处的数据库状态

* 命令传播用于在主服务器状态被修改，导致主从服务器数据库状态不一致的时候，让主从服务器数据库重新回到一致状态


从服务器对主服务器同步操作需要向主服务器发送SYNC命令完成：

1) 从服务器向主服务器发送SYNC命令

2) 收到SYNC命令的主服务器执行BGSAVE命令，在后台生成一个RDB文件，并使用一个缓冲区记录从现在开始执行的所有命令

3) 主服务器将生成的RDB文件发送给从服务器，从服务器接受并载入RDB文件，将自己的数据库状态更新到主服务器执行BGSAVE时的数据库状态

4) 主服务器将记录在缓冲区里的所有命令发送给从服务器，从服务器执行这些命令，将自己数据库状态更新至主服务器最新状态

命令传播：

当主服务器执行客户端发送的写命令时，会造成主从服务器状态不一直，主服务器会将造成不一致的那条写命令发送给从服务器。

旧版复制的缺点：

从服务器断线重新连接的时候，仍然会完整的执行整个过程，仍要生成RDB文件同步，效率低下。



2.8版本开始用PSYNC命令代替SYNC命令执行复制的同步操作

PSYNC命令具有完整同步和部分同步两种模式：

* 完整同步：和SYNC一样

* 部分同步，用于处理短线重连后复制情况：主服务器将主服务器连接断开期间执行的写命令，发送给从服务器，从服务器执行这些命令。


## sentinel 哨兵

sentinel是redis高可用的解决方案：

由一个或者多个sentinel组成的sentinel系统可以监视任意多个主服务器以及这些主服务器属下的所有从服务器，

并在被监视的主服务器下线时，自动将下线主服务器属下的某个从服务器升级为新的主服务器


启动sentinel服务：

redis-server /path/to/sentinel.conf --sentinel

运行一个 Sentinel 所需的最少配置：

{% highlight python %}
sentinel monitor mymaster 127.0.0.1 6379 2  # 给主机起名字mymaster 多个主机不重复，当2个sentinel实例都认为master失效，则认为失效
sentinel down-after-milliseconds mymaster 60000  # 多少毫秒链接不到master认为失效
sentinel failover-timeout mymaster 180000
sentinel parallel-syncs mymaster 1 # 执行故障转移时， 最多可以有多少个从服务器同时对新的主服务器进行同步，通过将这个值设为 1 来保证每次只有一个从服务器处于不能处理命令请求的状态
sentinel client-reconfig-script mymaster /var/redis/reconfig.sh # 在重新配置new master,new slave过程,可以触发的脚本
{% endhighlight %}

redis slave服务可以配置`slave-priority 100`，在master失效后，sentinel根据这个值来决定优先使用哪个slave切换为master


每个 Sentinel 都需要定期执行的任务：

* 每个 Sentinel 以每秒钟一次的频率向它所知的主服务器、从服务器以及其他 Sentinel 实例发送一个 PING 命令。

* 如果一个实例（instance）距离最后一次有效回复 PING 命令的时间超过 down-after-milliseconds 选项所指定的值， 那么这个实例会被 Sentinel 标记为主观下线。 一个有效回复可以是： +PONG 、 -LOADING 或者 -MASTERDOWN 。

* 如果一个主服务器被标记为主观下线， 那么正在监视这个主服务器的所有 Sentinel 要以每秒一次的频率确认主服务器的确进入了主观下线状态。

* 如果一个主服务器被标记为主观下线， 并且有足够数量的 Sentinel （至少要达到配置文件指定的数量）在指定的时间范围内同意这一判断， 那么这个主服务器被标记为客观下线。

* 在一般情况下， 每个 Sentinel 会以每 10 秒一次的频率向它已知的所有主服务器和从服务器发送 INFO 命令。 当一个主服务器被 Sentinel 标记为客观下线时， Sentinel 向下线主服务器的所有从服务器发送 INFO 命令的频率会从 10 秒一次改为每秒一次。

* 当没有足够数量的 Sentinel 同意主服务器已经下线， 主服务器的客观下线状态就会被移除。 当主服务器重新向 Sentinel 的 PING 命令返回有效回复时， 主服务器的主管下线状态就会被移除


可以自动发现 Sentinel 和从服务器：

* Sentinel 可以通过发布与订阅功能来自动发现正在监视相同主服务器的其他 Sentinel ， 这一功能是通过向频道 __sentinel__:hello 发送信息来实现的

* Sentinel 可以通过询问主服务器来获得所有从服务器的信息



当一个主服务器被判断为客观下线时，监视这个下线服务器的各个sentinel会进行协商，选举出一个领头sentinel，对下主线服务器执行故障转移操作。


Sentinel 使用以下规则来选择新的主服务器：

* 在失效主服务器属下的从服务器当中， 那些被标记为主观下线、已断线、或者最后一次回复 PING 命令的时间大于五秒钟的从服务器都会被淘汰。

* 在失效主服务器属下的从服务器当中， 那些与失效主服务器连接断开的时长超过 down-after 选项指定的时长十倍的从服务器都会被淘汰。

* 在经历了以上两轮淘汰之后剩下来的从服务器中， 我们选出复制偏移量（replication offset）最大的那个从服务器作为新的主服务器； 如果复制偏移量不可用， 或者从服务器的复制偏移量相同， 那么带有最小运行 ID 的那个从服务器成为新的主服务器



# 六、实际应用

## 升级redis版本

可以安装新版本的redis，并用新的配置文件，开启新的端口

将新版本的redis服务作为老版本的slave

数据同步问题解决了

客户端配置修改为新版本服务器端口

取消复制 slaveof no one

关闭老版本redis服务

## 访问频率控制

要求：每分钟每个用户最多访问100个页面

思路：

每个用户使用 rate:limiting:IP 作为键，

每次访问则使用INCR命令递增，同时设置键生命周期为1分钟

{% highlight python %}

isKeyExists = EXISTS rate:limiting:IP
if isKeyExists:
    times = INCR rate:limiting:IP
    if times > 100:
        print "deny!"
        exit
else:
    INCR rate:limiting:IP
    EXPIRE rate:limiting:IP 60

{% endhighlight %}

有个问题 如果执行到倒数第二行，突然退出，则删除该键前最多访问10次

使用事务来修正
{% highlight python %}

isKeyExists = EXISTS rate:limiting:IP
if isKeyExists:
    times = INCR rate:limiting:IP
    if times > 10:
        print "deny!"
        exit
else:
    MULTI
    INCR rate:limiting:IP
    EXPIRE rate:limiting:IP 60
    EXEC

{% endhighlight %}


还有问题：

如果用户在一分钟的第一秒访问1次，最后一秒访问9次；

在下分钟的第一秒内访问10次，则两秒内访问了19次，突破了每分钟10次限制

解决：

使用列表记录用户最近的访问时间

一旦键中的元素超过10个， 就判断时间最早的元素距现在的时间是否小于1分钟。

如果是则表示用户最近1分钟的访问次数超过了10次；

如果不是就将现在的时间加入到列表中， 同时把最早的元素删除。

{% highlight python %}

listLength = LLEN rate:limiting:IP

if listLength < 10:
    LPUSH rate:limiting:IP now()
else:
    time = LINDEX rate:limiting:IP -1
    if now() - time < 60:
        print "deny!"
        exit
    else:
        LPUSH rate:limiting:IP now()
        LTRIM rate:limiting:IP 0 9


{% endhighlight %}



## 大量用户统计活跃用户

1亿个用户，每个用户登录则记为活跃，否则不活跃

统计每周、每月、连续多少天活跃用户来做活动

思路：使用位图法记录用户是否活跃

* 用户登录，每天按照日期生成一个位图，登录用户的ID位上的bit值设置为1
* 把一周的位图 and 计算

{% highlight python %}

127.0.0.1:6379[14]> setbit mon 100000000 0
(integer) 0
127.0.0.1:6379[14]> setbit mon 3 1
(integer) 0
127.0.0.1:6379[14]> setbit mon 5 1
(integer) 0
127.0.0.1:6379[14]> setbit mon 7 1
(integer) 0
127.0.0.1:6379[14]> setbit tue 100000000 0
(integer) 0
127.0.0.1:6379[14]> setbit tue 3 1
(integer) 0
127.0.0.1:6379[14]> setbit tue 5 1
(integer) 0
127.0.0.1:6379[14]> setbit tue 8 1
(integer) 0
127.0.0.1:6379[14]> setbit wen 100000000 0
(integer) 0
127.0.0.1:6379[14]> setbit wen 3 1
(integer) 0
127.0.0.1:6379[14]> setbit wen 4 1
(integer) 0
127.0.0.1:6379[14]> setbit wen 6 1
(integer) 0
127.0.0.1:6379[14]> bitop and res mon tue wen
(integer) 12500001
127.0.0.1:6379[14]> bitcount res
(integer) 1

{% endhighlight %}

优点：

* 节约空间，1亿个用户，1亿个bit，大约10M字符
* 方便计算


## 安全队列

* rpoplpush task bak
* 接收返回值，并做业务处理
* 如果成功 rpop bak 清除任务
* 如果不成功，下次从bak取任务


## 限制列表中元素个数

只保留100条日志

LPUSH logs newlog

LTRIM logs 0 99

## python调用redis存放pickle数据

{% highlight python %}

def dumps_zip(value):
    pickled_value = pickle.dumps(value, 2)
    zip_pickled_value = zlib.compress(pickled_value)
    return zip_pickled_value


def loads_zip(zip_pickled_value):
    pickled_value = zlib.decompress(zip_pickled_value)
    value = pickle.loads(pickled_value)
    return value


if settings.REDIS_DATA_ZIP:
    r_loads = loads_zip
    r_dumps = dumps_zip
else:
    r_loads = pickle.loads
    r_dumps = lambda v: pickle.dumps(v, 2)


class PickledRedis(StrictRedis):
    """
    a pickled redis client
    """
    def get(self, name):
        pickled_value = super(PickledRedis, self).get(name)
        if pickled_value is None:
            return None
        return r_loads(pickled_value)

    def set(self, name, value, ex=None, px=None, nx=False, xx=False):
        return super(PickledRedis, self).set(name, r_dumps(value), ex, px, nx, xx)
{% endhighlight %}


## 分布式锁的实现

利用SETNX 很容易实现分布式锁

SETNTX lock_key now+expires+1

* 返回1，则客户端获得锁，并设置时间，最后可以 del lock_key 释放锁
* 返回0，则被其他用户占用

问题：持有锁的客户端失败或者崩溃不能释放锁，怎么办？

不能通过简单的del来删除锁，然后setnx一次，多个客户端检测锁超时的时候，会出现竞态

1. C0 操作超时，但持有锁。
2. C1 C2 检测lock_key时间戳，发现过期
3. C1 发送del lock_key
4. C1 发送 SETNX lock_key 成功了
5. C2 发送del lock_key
6. C2 发送 SETNX lock_key 成功了
7. 结果C1 C2都拿到了锁

解决方案：

1. C3 发现lock_key过期
2. GETSET lock_key now+expires+1
3. C3 拿到的时间戳仍然是过期的，说明C3 可以获得锁
4. C4 同时操作，GETSET 由于C3重新设置了时间，C4获得的时间是没有过期的，不能获得锁


{% highlight python %}
class RedisLock(object):
    def __init__(self, r, key, expires=3, timeout=3):
        """
        Distributed locking using Redis SETNX and GETSET.

        Usage::

            with RedisLock('my_lock'):
                print "Critical section"

        :param  expires     We consider any existing lock older than
                            ``expires`` seconds to be invalid in order to
                            detect crashed clients. This value must be higher
                            than it takes the critical section to execute.
        :param  timeout     If another client has already obtained the lock,
                            sleep for a maximum of ``timeout`` seconds before
                            giving up. A value of 0 means we never wait.
        """
        self.redis = r
        self.key = key
        self.timeout = timeout
        self.expires = expires

    def __enter__(self):
        timeout = self.timeout
        while timeout >= 0:
            expires = time.time() + self.expires + 1

            if self.redis.setnx(self.key, expires):
                # We gained the lock; enter critical section
                return

            current_value = self.redis.get(self.key)

            # We found an expired lock and nobody raced us to replacing it
            if current_value and float(current_value) < time.time() and \
                self.redis.getset(self.key, expires) == current_value:
                    return

            timeout -= 1
            time.sleep(1)

        raise LockTimeout("Timeout whilst waiting for lock")

    def __exit__(self, exc_type, exc_value, traceback):
        self.redis.delete(self.key)


class LockTimeout(BaseException):
    pass
{% endhighlight %}


## redis监控

集成到django admin上的[rediaboard](https://github.com/ionelmc/django-redisboard)


## 压力测试

{% highlight python %}

redis-benchmark -h 127.0.0.1 -p 6379 -t set,lpush -n 100000 -q

{% endhighlight %}

http://www.tutorialspoint.com/redis/redis_benchmarks.htm


## 数据导入导出

```python

# 安装redis-dump
yum install ruby rubygems ruby-devel   //安装rubygems 以及相关包
gem sources --add http://upyun.gems.ruby-china.org/
gem sources -l
gem sources --remove http://rubygems.org/
gem install redis-dump -V

# redis-dump 导出数据
redis-dump –u 127.0.0.1:6379 > test.json (导出redis 默认数据库的数据，默认数据库为0)
如果指定15数据库的数据：
redis-dump –u 127.0.0.1:6379 –d 15 >test.json

# redis-load 还原数据
cat test.json | redis-load
如果导出时指定了数据库
cat test.json | redis-load –d 15

# key多的话，报错。。。。。方案舍弃
```

直接存rdb

```python

CONFIG GET dir

BGSAVE


stop redis server

copy the dump.rdb to target host

start redis server

```