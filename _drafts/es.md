### 不同index不同权重

http://nocf-www.elastic.co/guide/en/elasticsearch/reference/current/search-request-index-boost.html

```python
GET /_search
{
    "indices_boost" : {
        "index1" : 1.4,
        "index2" : 1.3
    }
}
```


### filter 查询新格式

https://www.elastic.co/guide/en/elasticsearch/reference/5.0/query-dsl-filtered-query.html

```python

GET _search
{
  "query": {
    "bool": {
      "must": {
        "match": {
          "text": "quick brown fox"
        }
      },
      "filter": {
        "term": {
          "status": "published"
        }
      }
    }
  }
}
```

### 单个index不支持多个type

https://www.elastic.co/guide/en/elasticsearch/reference/6.0/removal-of-types.html
