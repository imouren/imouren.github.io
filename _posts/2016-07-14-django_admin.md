---
layout: post
title:  "django admin"
date:   2016-07-14 09:05:31
categories: python django
tags: python django
---

* content
{:toc}

## 去这个网站找

https://djangosnippets.org/

## 编辑的时候按照指定field的属性

fields = ["field2", "field1"]

## 需要显示的字段

list_display = ["word", "link", "channel", "shop"]

## 搜索的字段

search_fields = ["word", "channel__name", "shop__name"]

其中 channel__name 为foreignkey的name字段

## 右侧筛选栏

list_filter = ["channel__name", ]

## 日期搜索

date_hierarchy = 'publication_date'

## 替换 select 为 search

https://docs.djangoproject.com/en/1.9/ref/contrib/admin/#django.contrib.admin.ModelAdmin.raw_id_fields

raw_id_fields = ("merchant",)

## 多选框

```python

@python_2_unicode_compatible
class Shop(models.Model):
    shop_id = models.IntegerField(u"门店ID", unique=True)
    merchant = models.ForeignKey(Merchant, verbose_name=u"商户")
    name = models.CharField(u"门店名称", max_length=128)


@python_2_unicode_compatible
class ShopGroup(models.Model):
    shop_group_id = models.IntegerField(u"门店组ID", unique=True)
    name = models.CharField(u"门店组名称", max_length=128)
    shops = models.ManyToManyField(Shop, db_table='shop_group_ship')


class ShopGroupAdmin(admin.ModelAdmin):
    list_display = ["shop_group_id", "name"]
    filter_horizontal = ('shops',)

```

## 多选框，不重复分组

```python

def formfield_for_manytomany(self, db_field, request, **kwargs):
    if db_field.name == "shops":
        path = request.get_full_path()
        shop_group_pk = utils.r1("shopgroup/(\d+)/", path)
        # 你希望用到的 这里采用 有分配组的其他 + 当前组的
        kwargs["queryset"] = dbutils.get_nogroup_shops(shop_group_pk)
    return super(ShopGroupAdmin, self).formfield_for_manytomany(db_field, request, **kwargs)


```

## 对app和model进行排序

https://github.com/mishbahr/django-modeladmin-reorder

## 权限设置

```python

class DefaultSearchWordsAdmin(admin.ModelAdmin):
    form = forms.DefaultSearchWordsForm

    list_display = ["word", "link", "channel", "shop", "start_time", "end_time"]
    fields = list_display
    search_fields = ["word", "channel__name", "shop__name"]
    list_filter = ["channel__name", ]

    def has_add_permission(self, request):
        objs = models.DefaultSearchWords.objects.all()
        if len(objs) >= constants.DEFAULT_SEARCH_WORDS_MAX:
            return False
        else:
            return True

```

## 表单验证

```python

class DefaultSearchWordsForm(forms.ModelForm):
    def clean_link(self):
        link = self.cleaned_data["link"]
        if link:
            try:
                res = requests.get(link)
                flag = res.ok
            except:
                flag = False
            if not flag:
                msg = u"链接无法访问"
                raise forms.ValidationError(msg)
        return link

    def clean(self):
        cleaned_data = super(DefaultSearchWordsForm, self).clean()
        link = cleaned_data.get("link", "")
        word = cleaned_data.get("word", "")
        if not link and word.startswith("test"):
            msg = u"无链接，且关键字物品少于%s个" % constants.SEARCH_WORD_GOODS_MIN
            raise forms.ValidationError(msg)
        return cleaned_data

```

## 自定义admin的模板

比如想在admin最后加上一个些其他网站的链接，方便访问

```python

# yourapp/templates/admin/index_custom.html

{% extends "admin/index.html" %}

{% block sidebar %}
  {{ block.super }}

<div class="module" style="float: left; width: 498px">
    <table style="width: 100%">
        <caption>Custom Links</caption>
        <tbody>
            <tr>
                <th scope="row"><a href="http://www.dmall.com" target="_blank">dmall</a></th>
            </tr>
        </tbody>
    </table>
</div>
{% endblock %}


# urls.py

from django.contrib import admin

admin.site.index_template = 'admin/index_custom.html'
admin.autodiscover()

```


## manytomany 中间表自定义

https://docs.djangoproject.com/en/dev/topics/db/models/#extra-fields-on-many-to-many-relationships

```python

from django.db import models

class Person(models.Model):
    name = models.CharField(max_length=128)

    def __str__(self):              # __unicode__ on Python 2
        return self.name

class Group(models.Model):
    name = models.CharField(max_length=128)
    members = models.ManyToManyField(Person, through='Membership')

    def __str__(self):              # __unicode__ on Python 2
        return self.name

class Membership(models.Model):
    person = models.ForeignKey(Person, on_delete=models.CASCADE)
    group = models.ForeignKey(Group, on_delete=models.CASCADE)
    date_joined = models.DateField()
    invite_reason = models.CharField(max_length=64)

```