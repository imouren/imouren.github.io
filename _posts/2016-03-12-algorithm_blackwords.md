---
layout: post
title:  "脏字过滤算法"
date:   2016-03-12 16:05:31
categories: 算法 工具代码
tags: 算法 工具代码
---

* content
{:toc}


之前游戏聊天有个需求，要求脏字要用星号代替。

一般来说，过滤字库很大，而玩家在游戏中聊天的字数并不多。

如果每说一句话，就遍历整个过滤词库，代价太大了。

这里采取空间换时间的方式，存取词库中每个词的首字和其他字来判定是否为脏字。


在后来的类似场景中，也用这个来过滤。

```python
# -*- coding: utf-8 -*-

class Filter(object):

    def __init__(self):
        self.maxlength = 0
        self.dicts = {}
        self.first = {}
        self.others = {}


    def good_sentence(self, sentence):
        for index, char in enumerate(sentence):
            # 首字是否为脏字
            if char not in self.first:
                continue
            # 根据脏字最大长度优化
            for j in xrange(min(self.maxlength, len(sentence)-index)):
                # 判断下个字是否为脏字 j=0 首字符已经判断过
                if j != 0 and sentence[index+j] not in self.others:
                    break
                # 判定脏字
                word = sentence[index:index+j+1]
                if word in self.dicts:
                    sentence = sentence.replace(word, "*"*len(word))
        return sentence


    def load_black_words(self, words):
        for word in words:
            self.maxlength = max(self.maxlength, len(word))
            if word and word not in self.dicts:
                self.dicts[word] = None
                self.first[word[0]] = None
                for char in word[1:]:
                    self.others[char] = None

words = ["hell", "sb", "shit", "nima"]

f = Filter()
f.load_black_words(words)

sentence = "hello, you are a sb, shit!"

print f.good_sentence(sentence)

# ****o, you are a **, ****!

```


## 一段项目中的代码

```python
# -*- coding: utf-8 -*-
import threading
from content.dbutils import get_black_words

RELOAD_TIME = 1800


class Filter(object):

    def __init__(self):
        self.init()

    def init(self):
        self.maxlength = 0
        self.dicts = {}
        self.first = {}
        self.others = {}

    def good_sentence(self, sentence):
        for index, char in enumerate(sentence):
            if char not in self.first:
                continue
            for j in xrange(min(self.maxlength, len(sentence) - index)):
                if j != 0 and sentence[index + j] not in self.others:
                    break
                word = sentence[index:index + j + 1]
                if word in self.dicts:
                    sentence = sentence.replace(word, "*" * len(word))
        return sentence

    def load_black_words(self, words):
        for word in words:
            self.maxlength = max(self.maxlength, len(word))
            if word and word not in self.dicts:
                self.dicts[word] = None
                self.first[word[0]] = None
                for char in word[1:]:
                    self.others[char] = None

    def delete_black_words(self, words):
        for word in words:
            if word and word in self.dicts:
                self.dicts.pop(word)

    def reload_black_words(self, words):
        self.init()
        self.load_black_words(words)


sentence_filter = Filter()
sentence_filter.load_black_words(get_black_words())


def auto_load_words():
    global timer
    sentence_filter.load_black_words(get_black_words())
    timer = threading.Timer(RELOAD_TIME, auto_load_words)
    timer.start()


timer = threading.Timer(RELOAD_TIME, auto_load_words)
timer.start()

```