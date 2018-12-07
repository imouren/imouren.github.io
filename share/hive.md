# HIVE分享

## hive是什么

*hive*是基于Hadoop的一个数据仓库工具，可以将结构化的数据文件映射为一张数据库表，并提供简单的sql查询功能，可以将sql语句转换为MapReduce任务进行运行。--百度百科

1. Hive 由 Facebook 实现并开源

2. 是基于 Hadoop 的一个数据仓库工具

3. 可以将结构化的数据映射为一张数据库表

4. 并提供 HQL(Hive SQL)查询功能

5. 底层数据是存储在 HDFS 上

6. Hive的本质是将 SQL 语句转换为 MapReduce 任务运行

7. 使不熟悉 MapReduce 的用户很方便地利用 HQL 处理和计算 HDFS 上的结构化的数据，适用于离线的批量数据计算


## hive的架构

![架构](hive.png)

HiveQL 通过命令行或者客户端提交，经过 Compiler 编译器，运用 MetaStore 中的元数 据进行类型检测和语法分析，生成一个逻辑方案(Logical Plan)，然后通过的优化处理，产生 一个 MapReduce 任务。


## hive相关操作

1. 创建数据库

```
DROP TABLE IF EXISTS orange.result_week;
```

2. 创建表

```
CREATE EXTERNAL TABLE IF NOT EXISTS orange.result_week (
    days INT,
    play_second INT,
    play_rate FLOAT,
    video_num INT,
    video_rate FLOAT,
    user_num INT,
    user_rate FLOAT,
    per_play_second FLOAT,
    per_video_num FLOAT,
    per_play_second_day FLOAT,
    per_video_num_day FLOAT
)
PARTITIONED BY (year STRING, month STRING, day STRING)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
LOCATION '/dw/logs/recommend/ott/userPlayTime/';
```

内部表和外部表的区别：

* 删除内部表，删除表元数据和数据

* 删除外部表，删除元数据，不删除数据

3. 创建分区

```
ALTER TABLE orange.result_week ADD IF NOT EXISTS PARTITION(year='2018', month='11', day='04') LOCATION '/dw/logs/recommend/ott/userPlayTime/2018/11/04/weekUser/';
```

4. sqoop

创建hive表 导入数据到hive表

```
sqoop import --connect jdbc:mysql://10.1.x.x/vmis --username vmis_read_only  --password xxx --table  fv_channel --hive-import  --warehouse-dir /home/hadoop/hive/warehouse --create-hive-table --hive-database vmis
```

## hive查询

fbuffer-- app版本分布

```
select app, ve, count(*) n from gotyou2.fbuffer
where year = '2017' and month = '08' and day = '22'
group by app, ve;
```

fbuffer--地域统计

```
select app, area, count(*) n
from
(
select TRANSFORM (ip, app, fudid, vid, tid, ok, lian) USING 'python fbuffer_areax/area.py' AS (area, app, fudid, vid, tid, ok, lian) from gotyou2.fbuffer
where year = '2017' and month = '08' and day = '22'
and ok = 0
) t
group by app, area;
```


fbuffer-取播放数大于100的

```
SELECT r.app, v.video_id, v.name, r.n FROM vmis.fv_video v
JOIN
(
SELECT app, vid, count(*) n FROM gotyou2.fbuffer
WHERE year = '2017' and month = '08' and day = '16'
and ok = 0
GROUP BY app, vid HAVING n >300
) r
ON v.video_id = r.vid
ORDER BY r.app, r.n DESC
;
```


分位数

```python

select percentile(cast(priority as BIGINT), array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0))
from gotyou2.topic_click
where year='2017' and month='09' and day = '10'
and  app = 'aphone_v_smart'
and page = 'play'
;
```

排重

```python

insert overwrite table store
  select t.p_key,t.sort_word from
    ( select p_key,
           sort_word ,
           row_number()over(distribute by p_key sort by sort_word) as rn
     from store) t
     where t.rn=1;
```

## python操作hive

```python
import pyhs2

HIVE_CONF = {
    'db_host': '10.1.6.35',
    'user': 'hive',
    'password': 'hive',
    'database': 'gotyou2'
}


class HiveClient(object):
    def __init__(self, db_host, user, password, database, port=10000, auth_mechanism="PLAIN"):
        self._db_host = db_host
        self._user = user
        self._password = password
        self._database = database
        self._port = port
        self._authMechanism = auth_mechanism
        self._conn = ''

    def __enter__(self):
        self._conn = pyhs2.connect(
            host=self._db_host,
            port=self._port,
            authMechanism=self._authMechanism,
            user=self._user,
            password=self._password,
            database=self._database
        )
        return self

    def query(self, sql, query=True):
        with self._conn.cursor() as cursor:
            cursor.execute(sql)
            if query:
                return cursor.fetch()
            else:
                return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn:
            self._conn.close()


def query_example():
    statement = "select * from gotyou2.fbuffer limit 10"
    with HiveClient(**HIVE_CONF) as hive:
        result = hive.query(statement)
    return result
```

## UDF函数

stp有些字符被urlencode了

```python
import sys
import urllib

for line in sys.stdin:
    app, stp, fudid = line.strip('\n').split('\t')
    stp = urllib.unquote(stp)
    print "%s\t%s\t%s" % (app, stp, fudid)
```


IP转换为城市

```
hdfs:///udf/fbuffer_areax/area.py
```


## 其他产品

* Spark SQL

* Hive on Spark

把Spark作为Hive的一个计算引擎，将Hive的查询作为Spark的任务提交到Spark集群上进行计算。通过该项目，可以提高Hive查询的性能，同时为已经部署了Hive或者Spark的用户提供了更加灵活的选择，从而进一步提高Hive和Spark的普及率。

* impala  http://impala.apache.org/

把执行计划表现为一棵完整的执行计划树，可以更自然地分发执行计划到各个Impalad执行查询，而不用像Hive那样把它组合成管道型的map->reduce模式，以此保证Impala有更好的并发性和避免不必要的中间sort与shuffle。

* Presto https://prestodb.io/
Facebook的科学家和分析师一直依靠Hive来做数据分析。
数据量大，嫌慢，开发了一个分布式的查询引擎Presto。
