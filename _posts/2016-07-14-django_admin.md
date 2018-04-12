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

## 不显示修改按钮

list_display_links = None

## 不显示 删除 的action

```python

def get_actions(self, request):
    actions = super(ChannelAdmin, self).get_actions(request)
    del actions['delete_selected']
    return actions

```

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


## 对super用户特殊处理

```python

def is_superuser(request):
    if request.user.is_active and request.user.is_superuser:
        return True
    else:
        return False

class StoreAdmin(admin.ModelAdmin):
    list_display = ["store_id", "name", "vender"]
    fields = list_display
    list_filter = ["vender__name", ]
    search_fields = ["name", ]
    raw_id_fields = ("vender",)
    readonly_fields = models.Store._meta.get_all_field_names()

    def get_readonly_fields(self, request, obj=None):
        if is_superuser(request):
            return []
        else:
            return super(StoreAdmin, self).get_readonly_fields(request, obj)

    def get_list_display_links(self, request, list_display):
        list_display = super(ChannelAdmin, self).get_list_display_links(request, list_display)
        if not is_superuser(request):
            list_display = None
        return list_display

    def get_actions(self, request):
        actions = super(StoreAdmin, self).get_actions(request)
        if not is_superuser(request):
            del actions['delete_selected']
        return actions

    def has_add_permission(self, request):
        return is_superuser(request)

    def has_delete_permission(self, request, obj=None):
        return is_superuser(request)

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

{ % extends "admin/index.html" % }

{ % block sidebar % }
  { { block.super } }

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
{ % endblock % }


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

## 添加用户和权限

```python

CMS_PERMS = ["change_channel", "change_vender", "change_store",
            "change_storegroup", "add_storegroup", "delete_storegroup",
            "change_defaultsearchwords", "add_defaultsearchwords", "delete_defaultsearchwords",
            "change_hotsearchwords", "add_hotsearchwords", "delete_hotsearchwords",
            "change_positionpush", "add_positionpush", "delete_positionpush",
            "change_categorypush", "add_categorypush", "delete_categorypush",
            "change_blackskus", "add_blackskus", "delete_blackskus",
            ]

def add_custom_user():
    # add user
    try:
        user = User.objects.get(username=settings.CMS_USER)
    except:
        user = None
    if not user:
        user = User.objects.create_user(settings.CMS_USER, settings.CMS_MAIL, settings.CMS_PASSWD)
        user.is_active = True
        user.is_staff = True
        user.save()
    # add permission
    for perm in settings.CMS_PERMS:
        if not user.has_perm("search_cms." + perm):
            permission = Permission.objects.get(codename=perm)
            user.user_permissions.add(permission)
            get_object_or_404(User, pk=user.id)

```


## 在模板中使用settings的数据

在你的app下面创建一个`context_processors.py` 文件

```python

# -*- coding: UTF-8 -*-
from django.conf import settings


def global_settings(request):
    # return any necessary values
    return {
        'ERP_SITE': settings.ERP_SITE,
    }

```

添加自定义的 context processor

```python

TEMPLATE_CONTEXT_PROCESSORS = (
    ...
    'app.context_processors.global_settings',
)

```

在模板中使用数据

```python

<tr>
    <th scope="row"><a href={{ ERP_SITE }} target="_blank">ERP</a></th>
</tr>

```

## 字段增加链接

```python

class AlbumProgramRelateAdmin(admin.ModelAdmin):
    list_display = ["album", "programx", "period", "area_group", "number", "percent", "priority"]
    fields = list_display
    list_filter = ["album", ]
    search_fields = ["album__album_name", "program__program_name"]

    def programx(self, obj):
        return '<a href="/admin/gotyou2/program/?program_id=%s" target="_blank">%s</a>' % (obj.program.program_id, obj.program)
    programx.allow_tags = True
    programx.short_description = u'策略'

```

## 不用many2many实现多选框

https://github.com/kelvinwong-ca/django-select-multiple-field

## 级联的实现

django-smart-selects

https://github.com/digi604/django-smart-selects

## django-select2

https://github.com/asyncee/django-easy-select2

## django admin 模板

参考

https://djangopackages.org/grids/g/admin-styling/

https://github.com/rosarior/awesome-django

列出前几个

https://github.com/divio/djangocms-admin-style

https://github.com/sehmaschine/django-grappelli

https://github.com/django-admin-bootstrapped/django-admin-bootstrapped

https://github.com/darklow/django-suit (选用的这个)
https://github.com/django-ckeditor/django-ckeditor (富文本编辑器用这个)

django-image-cropping easy_thumbnails 图片的好使的
https://github.com/divio/django-filer
versatileimagefield (未测试)


## django 查看sql语句

```python
qs.query.__str__()

Video.objects.values("topic").annotate(total=Count('topic'))
有默认排序的需要自己加个排序，否则会自动加上自动排序的
Video.objects.values("topic").annotate(total=Count('topic')).order_by("topic")
```

## django admin 字符串转数字排序

```python
class XAdmin(admin.ModelAdmin):
    list_display = ['trade_rmb']
    list_filter = []

    def get_queryset(self, request):
        queryset = super(SYFTerminalAdmin, self).get_queryset(request)
        queryset = queryset.extra({'trade_rmb': "CAST(trade_rmb as DECIMAL)"})
        return queryset
```