---
layout: post
title:  "pandas统计数据并导出excel"
date:   2016-08-14 09:05:31
categories: pandas
tags: pandas
---

* content
{:toc}

## 统计首页楼层数据并导出

```python

# -*- coding:utf-8 -*-
import sys
from functools import partial
from datetime import datetime, timedelta

import pyhs2
import pandas as pd
import numpy as np

ROUND_N = 4

date_str = "20160816"

query = 'select action_type, title, vender_id  from app_home_site_click where date = "%s"' % date_str


HIVE_OXEYE = {
    'host': 'host',
    'user': 'user',
    'password': 'passwd',
    'database': 'data',
    'authMechanism': "PLAIN",
    'port': 10000
}


def datetime_to_string(adatetime, format_str="%Y%m%d"):
    return adatetime.strftime(format_str)


def get_date_str(adatetime=None):
    if adatetime is None:
        adatetime = datetime.now() - timedelta(1)
    date_str = datetime_to_string(adatetime)
    return date_str


def get_daily_data(date_str):
    print "loading data from hive %s ..." % date_str
    query = 'select action_type, title, vender_id  from app_home_site_click where date = "%s"' % date_str
    conn = pyhs2.connect(**HIVE_OXEYE)
    cur = conn.cursor()
    cur.execute(query)

    column_names = [a['columnName'] for a in cur.getSchema()]
    data = pd.DataFrame(cur.fetch(), columns=column_names)

    cur.close()
    conn.close()
    return data


def get_total_statistics(data):
    print "get total statistics"
    action_cnt = data["action_type"].value_counts()

    # 总计的点击
    click_total_cnt = action_cnt.sum()

    # 加入购物车 action_type = 2
    click_add_cart_cnt = action_cnt["2"]

    # 查看详情 action_type = 1
    click_view_detail_cnt = action_cnt["1"]

    assert((click_add_cart_cnt + click_view_detail_cnt) == click_total_cnt)

    add_cart_rate = round(click_add_cart_cnt / float(click_total_cnt) * 100, ROUND_N)

    view_detail_rate = round(click_view_detail_cnt / float(click_total_cnt) * 100, ROUND_N)

    # index_multi = pd.MultiIndex.from_tuples([("0", u"全部楼层")], names=['vender_id', 'title'])

    # total_res = pd.DataFrame({"addcart_rate": add_cart_rate,
    #     "viewdetail_rate": view_detail_rate
    # }, index=index_multi)
    return click_total_cnt, add_cart_rate, view_detail_rate


def _get_rates(total_cnt, typ, alist):
    click_cnt = sum(a == typ for a in alist)
    rate = round(click_cnt / float(total_cnt) * 100, ROUND_N)
    return rate


def get_dim_statistics(data, click_total_cnt):
    print "get dim statistics"
    group = data["action_type"].groupby([data["vender_id"], data["title"]])

    vd_rate_func = partial(_get_rates, click_total_cnt, "1")
    ac_rate_func = partial(_get_rates, click_total_cnt, "2")

    res = group.agg([("addcart_rate", ac_rate_func), ("viewdetail_rate", vd_rate_func)])
    return res


def join_total(res, add_cart_rate, view_detail_rate, date_str):
    res.ix[("0", "全部楼层"), :] = [add_cart_rate, view_detail_rate]
    return res


def export_excel(res_date_list):
    print "export to excel"
    writer = pd.ExcelWriter('homeclick.xlsx')
    for res, date_str in res_date_list:
        res.to_excel(writer, date_str)
    writer.save()


def main(days=1):
    res_date_list = []
    for day in xrange(1, days + 1):
        adatetime = datetime.now() - timedelta(day)
        date_str = get_date_str(adatetime)
        data = get_daily_data(date_str)
        click_total_cnt, add_cart_rate, view_detail_rate = get_total_statistics(data)
        res = get_dim_statistics(data, click_total_cnt)
        res = join_total(res, add_cart_rate, view_detail_rate, date_str)
        res_date_list.append((res, date_str))
    export_excel(res_date_list)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        days = 1
    else:
        days_arg = sys.argv[1]
        if days_arg.isdigit():
            days = int(days_arg)
        else:
            days = 1
    print "latest days %s" % days
    main(days)


```

分组TOPN

```
df = pd.DataFrame({'class':['a','a','b','b','a','a','b','c','c'],'score':[3,5,6,7,8,9,10,11,14]})

df.sort_values(['class','score'],ascending=[1,0],inplace=True)
grouped = df.groupby(['class']).head(2).reset_index(drop=True)
```
