# 脏词过滤算法

### 1. 简单粗暴先来下

直接字符串查找，找到的话，进行处理。比如替换为 ** 或者 提示用户。

**优点**：

简单粗暴

**缺点**：

性能太差，不适合实时处理的业务

### 2. 基于分词的过滤

将脏字库，作为词库；对要进行处理的文本进行分词，匹配到词库内容则进行相关处理。

可以把脏字作为字典的方式存入缓存，或者直接使用ElasticSearch。

**优点：**

简单且高效

**缺点：**

依赖分词精准性，不适合处理人工易读，机器分词差的情形。

### 3. 以前做游戏写过的一个实现

**需求：**

在游戏聊天中，将匹配到脏词词库的词，替换为*

**分析：**

脏词词库比较大，而聊天信息一般比较简单，实时性要求很高。

**方案：**

空间换时间，提前将脏词词库载入内存，并进行简单处理。

内存应该不用太担心，按照UTF8的编码，一个汉字三个字节，1M可以存储近35万汉字呢。

1. 将所有词存入字典

2. 将所有词的首个字存入字典

3. 将所有词的所有字存入字典

4. 开始遍历被处理的字符串

5. 判断字符是否在首字的字典中

   如果在，则继续判断下个字符是否在所有字的字典中

   ​	如果在，则继续判断下个字符

   ​	同时判断刚才在的那些字符的词是否在词的字典中

**贴下代码**

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
            # first character
            if char not in self.first:
                continue
            # bad word max len optimize
            for j in xrange(min(self.maxlength, len(sentence)-index)):
                if j != 0 and sentence[index+j] not in self.others:
                    break
                # bad word
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



### 4. 其他扩展

* KMP 单模式匹配算法
* WM 多模式匹配算法
* TTMP 是Terminator Triggered Multi-Pattern 脏词过滤算法
* DFA 有穷自动机检测脏词