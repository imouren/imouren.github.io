---
layout: post
title:  "hive数据导入pandas"
date:   2016-08-13 09:05:31
categories: hive pandas
tags: hive pandas
---

* content
{:toc}

## hive数据导入pandas

参考：https://github.com/pydata/pandas/issues/5919


```python

import pyhs2
import pandas as pd

query = 'select action_type, first_orderno, title, vender_id  from app_home_site_click where date = "20160816" limit 100'

HIVE_OXEYE = {
    'host': 'ip',
    'user': 'user',
    'password': 'passwd',
    'database': 'oxeye',
    'authMechanism': "PLAIN",
    'port': 10000
}


conn = pyhs2.connect(**HIVE_OXEYE)
cur = conn.cursor()
cur.execute(query)

if cur.getSchema() is None:
    cur.close()
    conn.close()
    return None

columnNames = [a['columnName'] for a in  cur.getSchema()]
print columnNames
columnNamesStrings = [a['columnName'] for a in  cur.getSchema() if a['type']=='STRING_TYPE']
output =  pd.DataFrame(cur.fetch(), columns=columnNames)

cur.close()
conn.close()
return output

```
