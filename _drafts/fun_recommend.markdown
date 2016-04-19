---
layout: post
title:  "公司推荐系统"
date:   2016-04-13 15:05:31
categories: 推荐
tags: 推荐
---

* content
{:toc}


简述下，在公司参与推荐系统，大体也是公司的现有的推荐系统实现。



## 网页个性首页

“视频号”战略下的个性首页，上线运营一年左右，现在未在用。


### 基于规则的推荐

根据用户的 客户端类型、客户端IP、上次访问时间、当前访问时间 来给用户分组；

根据用户分组获取用户的 版块、模板、策略、排序 信息；

根据策略值，取不同数据源获取数据。

### 个性化推荐

个性化推荐，被做成策略，直接通过策略取个性化接口中的数据或者其他源的个性数据。

### 开放接口

接受公司其他团队数据。

* 返回数据遵循指定格式
* 第三方接口请求接入时，会填入 URL、频率等信息
* 人工审核接口
* 自动根据URL、爬取频率尽量数据的获取，并缓存

自动爬取逻辑
```python

def load_ingegration_data():
    objs = ThirdPartyIntegration.objects.filter(is_active=True)
    now = datetime.now()
    for obj in objs:
        # 确保已经自动获取group_id
        if not obj.group_id or obj.group_id not in settings.PROGRAM_NHF_EXT_THIRD_PARTY:
            continue
        # 对比上次调用时间，来进行调用
        last_time = now - timedelta(minutes=obj.interval)
        if obj.last_invoke_time is None or obj.last_invoke_time <= last_time:
            obj.last_invoke_time = now
            obj.save()
            # 创建多线程执行
            t = threading.Thread(target=crawl_data, args=(obj.group_id, obj.url))
            t.start()

```

### 敏感词和黑名单

过滤掉指定敏感词和黑名单的视频。


判断是否存在敏感词和黑名单的算法
```python

class Filter(object):

    def __init__(self):
        self.mids = load_black_mid()
        self.black_word = load_black_word()

    def is_black_mid(self, mid):
        return mid in self.mids

    def has_black_word(self, sentence):
        if any(self.black_word):
            maxlength, dicts, first, others = self.black_word
            for index, char in enumerate(sentence):
                if char not in first:
                    continue
                for j in xrange(min(maxlength, len(sentence)-index)):
                    if j != 0 and sentence[index+j] not in others:
                        break
                    word = sentence[index:index+j+1]
                    if word in dicts:
                        return True
        return False


def load_black_word():
    objs = BlackWord.objects.all()
    maxlength = 0
    dicts = {}
    first = {}
    others = {}
    for obj in objs:
        word = obj.word
        maxlength = max(maxlength, len(word))
        if word and word not in dicts:
            dicts[word] = None
            first[word[0]] = None
            for char in word[1:]:
                others[char] = None
    data = (maxlength, dicts, first, others)
    return data
```

### 策略

* 运营策略：取特定DB取数据
* 15分钟、1天 的 最新、最热、上升最快的视频和媒体
* 订阅
* 基于物品的协同过滤 视频和媒体
* 电视剧、动漫的追剧

其中运营策略中的 “热播大片” 和 “搞笑” 版块，点击率高。

个性化策略中 “基于物品的过滤媒体” 点击率高。

### 绩效指标

页面转化率，实际提示30%-50%。

## 移动端频道页

首页、电影、电视剧、卡通、聚合频道等流量大的页面都已经上线。

目前仍在AB测试中。即人工运营与自动运营进行对比测试。

具体策略：

* 基于物品的协同过滤
* 基于内容的推荐（地域、频道、分类信息）
* 个性主题（给定主题规则自动生产主题内容，根据用户观看历史推荐主题）
* 电视剧、动漫等特定频道的观看历史追剧
* 相关媒体频道的热门片花
* 频道最热
* 最新更新
* 融合进人工运营好的版块

其中 基于内容的推荐、基于物品的协同过滤在媒体的表现很好。

最热微电影、协同过滤微电影表现很好。

电视剧、动漫的观看历史推荐（追剧）不错。

绩效指标采用：页面转换率。

不同频道页均有不同程度提升，电影提升30%左右；电视剧、动漫提示50%左右；聚合提示60%左右


## 移动端视频页

新上线的，数据全部由自动运营给出。

这里用到了如下几类策略：

* 基于item的协同过滤
* 基于题材：根据用户观看历史，找出用户喜爱的题材。推荐给用户题材热榜数据。
* 基于标题相似（效果差，问题多，未在线上使用）
* 最新
* 最热（15分钟、昨日）

目前，没有策略相关的上报信息，无法评估具体策略如何。

绩效指标采用：人均VV
