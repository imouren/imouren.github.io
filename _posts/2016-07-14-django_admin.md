---
layout: post
title:  "django admin"
date:   2016-07-14 09:05:31
categories: python django
tags: python django
---

* content
{:toc}

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