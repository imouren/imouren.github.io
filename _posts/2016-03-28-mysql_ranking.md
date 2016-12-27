---
layout: post
title:  "MYSQL查询排名或者某分类聚合"
date:   2016-03-28 09:05:31
categories: mysql
tags: mysql
---

* content
{:toc}

## 查询排名

```python
select count(1) as 排名 from 表名 where 分数字段 >= (select 分数字段 from 表名 where 姓名字段 = '用户名')
```

## 查询某分类数大于2的

fv_video_category 表存储了 视频ID和视频分类的对应关系

```python
select count(*) from (select count(video_id) from fv_video_category group by video_id having count(video_id) > 0) as cnt ;
```

## in 语句按照 in 的顺序排序

```python

SELECT id, name
FROM mytable
WHERE name IN ('B', 'A', 'D', 'E', 'C')
ORDER BY FIELD(name, 'B', 'A', 'D', 'E', 'C')

```

## 聚合类和limit的结合使用

```python
# 得到全部数据的sum(disable)
select sum(disable) from fv_video order by create_time desc limit 100000; 

select sum(disable) from (select disable from fv_video order by create_time desc limit 100000) a;
```

## 随机取数据

```python

SELECT * FROM fun_theme_content WHERE id >= ((SELECT MAX(id) FROM fun_theme_content)-(SELECT MIN(id) FROM fun_theme_content)) * RAND() + (SELECT MIN(id) FROM fun_theme_content)  LIMIT 10;

```






