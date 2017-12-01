---
layout: post
title:  "python生成云图"
date:   2017-11-08 10:05:31
categories:  python 工具代码
tags:  python
---

* content
{:toc}

## 生成云图

```python
# -*- coding: utf-8 -*-
from wordcloud import WordCloud
import cPickle as pickle
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def wordcloudplot(txt):
    d = os.path.dirname(__file__)
    filename = os.path.join(d, "./image/love.jpg")
    alice_mask = np.array(Image.open(filename).convert("L"))
    path = r'msyh.ttf'
    wordcloud = WordCloud(font_path=path,
                        background_color="black",
                        margin=5,
                        # width=1800, height=800,
                        mask=alice_mask,
                        max_words=2000,
                        max_font_size=60,
                        random_state=42)
    if isinstance(txt, dict):
        wordcloud = wordcloud.generate_from_frequencies(txt)
    else:
        wordcloud = wordcloud.generate(txt)
    wordcloud.to_file('tags.jpg')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


def main():
    with open('tags.txt') as f:
        txt = f.read()
        txt = txt.decode('utf-8')
        wordcloudplot(txt)


def test():
    txt = u"""
    哈哈  哈哈  呵呵呵 啊是砥砺奋进 阿斯蒂芬  车 啊啊  哈哈哈 刚 阿里 是
    """
    wordcloudplot(txt)


def test2():
    txt = {
        u"哈哈": 100,
        u"美女": 50,
        u"搞笑": 60,
        u"汽车": 220,
        u"体验": 20,
        u"士大夫": 55,
        u"计算机": 5,
        u"挂钩": 100,
        u"水果": 50,
        u"信息": 60,
        u"头条": 290,
        u"头疼": 20,
        u"阿夫": 155,
        u"考虑": 51,
        u"哈哈1": 100,
        u"美女1": 50,
        u"搞笑1": 60,
        u"汽车1": 220,
        u"体验1": 20,
        u"士大夫1": 55,
        u"计算机1": 5,
        u"挂钩1": 100,
        u"水果1": 50,
        u"信息1": 60,
        u"头条1": 290,
        u"头疼1": 20,
        u"阿夫1": 155,
        u"考虑1": 51,
    }
    wordcloudplot(txt)


def main_pkl():
    data = {}
    with open("tags.pkl", "rb") as f:
        data = pickle.load(f)
    wordcloudplot(data)

if __name__ == '__main__':
    main()

```


