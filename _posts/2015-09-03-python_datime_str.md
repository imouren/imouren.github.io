---
layout: post
title:  "python 时间转换函数"
date:   2015-09-03 18:05:31
categories: 工具代码 python
tags: 工具代码 python
---

* content
{:toc}



{% highlight python %}

def datetime_to_string(adatetime, format_str="%Y-%m-%d %H:%M:%S"):
    return adatetime.strftime(format_str)


def string_to_datetime(time_str, format_str="%Y-%m-%d %H:%M:%S"):
    return datetime.strptime(time_str, format_str)


def string_to_timestamp(time_str, format_str="%Y-%m-%d %H:%M:%S"):
    adatetime = string_to_datetime(time_str, format_str)
    timestamp = time.mktime(adatetime.timetuple())
    return timestamp


def timestamp_to_string(stamp, format_str="%Y-%m-%d %H:%M:%S"):
    return time.strftime(format_str, time.localtime(stamp))


def datetime_to_timestamp(adatetime):
    return time.mktime(adatetime.timetuple())


{% endhighlight %}


