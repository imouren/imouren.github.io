# django-ORM

## 查询null和空字符串

```python
# null查询
Book.objects.filter(author__isnull=True).count()

# 空字符串查询
Book.objects.filter(title__exact='').count()
# exact 是隐性的查询类型
Book.objects.filter(title='').count()

# 查询不是null也不为空字符串
Book.objects.exclude(author__isnull=True).exclude(title__exact='').count()

```

## admin change-list 外键数据过多

比较有很多用户

```python
raw_id_fields = ["user"]
```


## first() VS exists()

```python
# first()
Collect.objects.filter(user_id=user_id).filter(object_type=1).filter(object_id=object_id).filter(status=1).first()

SELECT `v_collect`.`id`, `v_collect`.`user_id`, `v_collect`.`object_type`, `v_collect`.`object_id`, `v_collect`.`status`, `v_collect`.`update_time`, `v_collect`.`create_time` FROM `v_collect` WHERE (`v_collect`.`user_id` = 1 AND `v_collect`.`object_type` = 1 AND `v_collect`.`object_id` = 1912 AND `v_collect`.`status` = 1) ORDER BY `v_collect`.`id` ASC LIMIT 1

# exists()
Collect.objects.filter(user_id=user_id).filter(object_type=1).filter(object_id=object_id).filter(status=1).exists()

SELECT (1) AS `a` FROM `v_collect` WHERE (`v_collect`.`user_id` = 1 AND `v_collect`.`object_type` = 1 AND `v_collect`.`object_id` = 1912 AND `v_collect`.`status` = 1) LIMIT 1

```

在查询关联关系时，使用exists()
比如通过主键查询一个MODEL是否是一个QuerySet的成员

```python

entry = Entry.objects.get(pk=123)

if some_queryset.filter(pk=entry.pk).exists():
    do something...

```

## select_related 得到关联查询的数据

```python

video = Video.objects.select_related("user", "topic").get(pk=vid)
```

## 随机选取元素


```python

# 直接用户数据库引擎的随机，数据大可能会慢
def get_random():
    return Category.objects.order_by("?").first()

# 采用逻辑进行选择
def get_random3():
    max_id = Category.objects.all().aggregate(max_id=Max("id"))['max_id']
    while True:
        pk = random.randint(1, max_id)
        category = Category.objects.filter(pk=pk).first()
        if category:
            return category
```