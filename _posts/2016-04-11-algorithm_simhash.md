---
layout: post
title:  "simhash算法"
date:   2016-04-11 15:05:31
categories: 算法
tags: 算法
---

* content
{:toc}


传统比较两个文本相似性的方法，大多是将文本分词之后，转化为特征向量距离的度量，常见的有余弦夹角算法、欧式距离、Jaccard相似度、最长公共子串、编辑距离等

两两比较固然能很好地适应，但这种方法的一个最大的缺点就是，无法将其扩展到海量数据。

simhash是locality sensitive hash（局部敏感哈希）的一种，Google就是基于此算法实现网页文件查重的。



在信息领域，两个长度相等的字符串的海明距离是在相同位置上不同的字符的个数，也就是将一个字符串替换成另一个字符串需要的替换的次数。

例如：

* toned和roses的海明距离是3
* 1011101和1001001的海明距离是2
* 2173896和2233796的海明距离是3


海明距离的原理恰恰符合这种局部敏感的相似。

simhash的实现原理：

1. 选择simhash的位数，请综合考虑存储成本以及数据集的大小，比如说32位
2. 将simhash的各位初始化为0
3. 提取原始文本中的特征，一般采用各种分词的方式。比如对于"the cat sat on the mat"，采用两两分词的方式得到如下结果：["th", "he", "e ", " c", "ca", "at", "t ", " s", "sa", " o", "on", "n ", " t", " m", "ma"]
4. 使用传统的32位hash函数计算各个word的hashcode
5. 对各word的hashcode的每一位，如果该位为1，则simhash相应位的值加1；否则减1
6. 对最后得到的32位的simhash，如果该位大于1，则设为1；否则设为0

![simhash.jpg](/files/simhash.jpg)


通过大量测试，simhash用于比较大文本，比如500字以上效果都还蛮好，距离小于3的基本都是相似，误判率也比较低。

但是如果我们处理的是微博信息，最多也就140个字，使用simhash的效果并不那么理想。看如下图，在距离为3时是一个比较折中的点，在距离为10时效果已经很差了，不过我们测试短文本很多看起来相似的距离确实为10。

![simhash_1.png](/files/simhash_1.png)


短文本相似处理的思路：



Twitter中对于热门事件的中文表达多种多样，转发时又往往会追加很多格式的各种内容，造成了相似内容的短文本的信息指纹不同，在这种情况下，段国成认为需要先为短文本抽取标签，再通过标签来计算相似度进行锐推合并。

1. 短文本抽取标签的方法：

    * 剔除所有英文、数字、标点字符；

    * 剔除所有Twitter用户名；

    * 分词，并标注词性，仅保留实体词性，如名词、动词；（技巧一！）

    * 过滤掉常用实体词（常用实体词是通过对历史锐推训练而得，即建立自己的停止词表）；（技巧二！）

    * 计算保留实体词的词频，并以此为权重，选择权重大的词语作为标签；

    * 标签数组长度大于一个阈值（如3），才认为是有信息量的锐推，否则忽略。（技巧三！）



2. 合并相似短文本（即我们所说的Retweet）方法：

    * 以每个标签作为Shingle特征（即核心思想还是“一个数据段分成若干Shingle，每个Shingle计算出一个hash值，然后组成一个以hash值为元素的数组，以此作为特征值或叫信息指纹来表示这个数据段”）；

    * 以网页去重领域中改进的Shingle方法来进行计算近期内（目的：缩小计算范围）锐推的相似度（Shingle算法的核心思想是将文本相似性问题转换为集合的相似性问题，改进的Shingle方法的时间复杂度近似于线性）；

    * 对相似程度大于一定阈值的、不同信息指纹的热门锐推，再次扫描出来，将它们的热度进行合并。



[python版实现](https://github.com/leonsim/simhash)


[参考来源](http://www.lanceyan.com/tech/arch/simhash_hamming_distance_similarity.html)
[参考来源](http://www.cnblogs.com/zhengyun_ustc/archive/2012/06/12/sim.html)
