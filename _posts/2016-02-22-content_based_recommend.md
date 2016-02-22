---
layout: post
title:  "内容推荐算法简单应用"
date:   2016-02-22 14:05:31
categories: 推荐 python
tags: 推荐 python
---

* content
{:toc}



应用到aphone端的电影，上线验证几个月，效果与 基于物品的协同过滤 相当，明显高于编辑配置版块



## 思路

根据电影自身的属性，并赋予一定的权重，

通过一定的相似算法得出类似电影，最终取最相近的K个推荐给用户。

* 属性和权重的选择

这里经过测试最终采用 标签、地域、分类 信息。

导演、演员、剧情等信息，影视库中资料太分散，采用后效果不明显，弃用。

权重方面，标签更为精准，稍微提升些权重。

* 相似算法

相似算法一般有，余弦定理、欧式距离、皮尔逊相关度评价等。

由于没有用户的评价信息，以观看记录为基础数据推荐。

这样每个电影对用户来说都是一样，采用计数即可。

* 平台特性

本身平台可播电影不多，质量也一般，只选取评分略高电影推荐。

## 代码实现

配置不同的属性不同的权重

{% highlight python %}
# 标签推荐配置
K = 25
WEIGHT = {
    "tag": 12,
    "area": 10,
    "category": 10,
}
{% endhighlight %}


为推荐准备电影信息，应用时要存入redis缓存


{% highlight python %}
def get_recommend_tag():
    """
    为推荐准备电影的信息
    """
    mids = get_mids()
    movie_area = get_movie_area(mids)
    movie_tag = get_movie_tag(mids)
    movie_category = get_movie_category(mids)
    tag_items = defaultdict(list)
    item_tags = defaultdict(list)
    for mid in mids:
        tags = []
        tags.extend(movie_area[mid])
        tags.extend(movie_tag[mid])
        tags.extend(movie_category[mid])
        item_tags[mid] = tags
        for tag in tags:
            tag_items[tag].append(mid)
    return tag_items, item_tags


def get_mids():
    conn = connections['poseidon_media']
    cursor = conn.cursor()
    sql = """select fm_media.media_id from
        fm_media, fm_media_channel
        where fm_media.disable=0 and fm_media.isvip=0 and fm_media_channel.channel_id = 1
        and fm_media.media_id = fm_media_channel.media_id and score >= 6.5
        order by fm_media.release_year desc"""
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
    except Exception, e:
        print e
        rows = []
    finally:
        cursor.close()
    return [int(x[0]) for x in rows]


def get_movie_area(mids):
    """
    获得所有电影地域
    """
    res = defaultdict(list)
    conn = connections['poseidon_media']
    cursor = conn.cursor()
    mids = ",".join(str(x) for x in mids)
    sql = """select media_id, area_id from fm_media_area
        where media_id in (%s)""" % mids
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
    except Exception, e:
        print e
        rows = []
    finally:
        cursor.close()
    for row in rows:
        media_id, area_id = row
        area_id = "area_" + str(area_id)
        res[media_id].append(area_id)
    return res


def get_movie_tag(mids):
    """
    获得所有电影标签
    """
    res = defaultdict(list)
    conn = connections['poseidon_media']
    cursor = conn.cursor()
    mids = ",".join(str(x) for x in mids)
    sql = """select media_id, tag_id from fm_media_tag
        where media_id in (%s)""" % mids
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
    except Exception, e:
        print e
        rows = []
    finally:
        cursor.close()
    for row in rows:
        media_id, tag_id = row
        tag_id = "tag_" + str(tag_id)
        res[media_id].append(tag_id)
    return res


def get_movie_category(mids):
    """
    获得所有电影类型
    """
    res = defaultdict(list)
    conn = connections['poseidon_media']
    cursor = conn.cursor()
    mids = ",".join(str(x) for x in mids)
    sql = """select media_id, category_id from fm_media_category
        where media_id in (%s)""" % mids
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
    except Exception, e:
        print e
        rows = []
    finally:
        cursor.close()
    for row in rows:
        media_id, category_id = row
        # 其他 8
        if category_id == 8:
            continue
        category_id = "category_" + str(category_id)
        res[media_id].append(category_id)
    return res
{% endhighlight %}


根据用户的观影历史，进行推荐

{% highlight python %}
# 根据用户历史 和 电影的 地域、标签、分类 来推荐相关电影
# mids为用户观影历史
def get_multi_movies_recommend_tags(mids, tag_items, item_tags, weight=WEIGHT, k=K):
    max_tags = 10
    mids_tags = {}
    for mid in mids[:30]:
        tags = item_tags[mid]
        for tag in tags:
            if tag in mids_tags:
                mids_tags[tag] += 1
            else:
                mids_tags[tag] = 1
    res = {}
    sorted_mids_tags = sorted(mids_tags.iteritems(), key=operator.itemgetter(1), reverse=True)
    for tag, n in sorted_mids_tags[:max_tags]:
        items = tag_items[tag]
        key = tag.split("_")[0]
        w = weight.get(key, 0)
        for item in items:
            if item in mids:
                continue
            if item in res:
                res[item] += n*w
            else:
                res[item] = n*w
    sorted_res = sorted(res.iteritems(), key=operator.itemgetter(1), reverse=True)
    return [x[0] for x in sorted_res[:k]]
{% endhighlight %}
