---
layout: post
title:  "基于概率论的分类方法：朴素贝叶斯"
date:   2015-11-30 09:05:31
categories: python 机器学习
tags: python
---

* content
{:toc}


概率论是许多机器学习算法的基础，所以深入理解这一主题十分重要。

## 1，基于贝叶斯决策理论的分类方法

优点：在数据较少的情况下，依然有效，可以处理多类别问题。
缺点：对于输入数据的准备方式较为敏感。


简单了解下贝叶斯决策理论：

有一个数据集，它由两类数据组成。假设有一个描述两类数据的统计参数，我们用p1(x, y)表示数据点(x, y)属于类别1的概率，用p2(x, y)表示数据点(x, y)属于类别2的概率，那么对于一个新的数据点(x, y)，可以用下面的规则来判定它的类别：

* 如果p1(x, y) > p2(x, y)， 那么类别为1
* 如果p2(x, y) > p1(x, y)， 那么类别为2

也就是说，我们会选择高概率对应的类别。这是贝叶斯决策理论的核心思想，即选择具有高概率的决策。

## 2，条件概率

有一个装有7块石头的罐子，3块是灰色的，4块是黑色的。

从里面随机取一块石头，取到灰色和黑的概率为3/7 和 4/7；

我们使用p(gray)表示取得灰色石头的概率，灰色石头数目除以所有石头数目

![bayes1.png](/files/bayes1.png)

如果将7块石头放到两个桶中，如图，那么概率应该如何计算呢？

![bayes2.png](/files/bayes2.png)



要计算p(gray)，你已经想到计算从B桶获得灰色石头概率的方法，这就是所谓**条件概率(conditional probability)**。

这个概率可以记做p(gray\|bucketB)，我们称之为“在已知石头出自B桶的条件下，取出灰色石头的概率”，

不难得出p(gray\|bucketA)为2/4，p(gray\|bucketB)为1/3

计算公式如下：

$$p(gray|bucketB) = p(gray and bucketB)/p(bucketB)$$

验证下公式的合理性：

* B桶中灰色石头数量除以两个桶中石头总是的到p(gray and bucketB) 为1/7
* B桶有3个石头，总数为7个 p(bucketB) 为 3/7
* 综合得出$p(gray\|bucketB)$ = $p(gray and bucketB)/p(bucketB)$ = 1/3

另外一种有效计算条件概率的方法称为贝叶斯准则。如果已知`p(x|c)`，要求`p(c|x)`可以使用下面的公式：

$$p(c|x) = \frac{p(x|c)p(c)}{p(x)}$$

## 3，使用条件概率来分类

上面提到过贝叶斯决策理论要求计算两个概率值p1(x, y) 和 p2(x, y)，其实是为了尽量简化描述，真正计算的是$p(c_1\|x, y)$ 和$p(c_2\|x, y)$。

$p(c_1\|x, y)$代表的意义为：给定某个由x, y 表示的数据点，那么该数据点来自类别$c_1$的概率

注意这些概率与刚才给出的概率$p(x,y\|c_1)$并不一样，不过可以通过贝叶斯准则来交互概率中条件与结果：

$$p(c_i|x,y) = \frac{p(x,y|c_i)p(c_i)}{p(x,y)}$$

* 如果$p(c_1\|x, y)$ > $p(c_2\|x,y)$，那么属于类别$c_1$
* 如果$p(c_1\|x, y)$ < $p(c_2\|x,y)$，那么属于类别$c_2$


## 4，使用朴素贝叶斯进行文档分类

朴素贝叶斯是贝叶斯分类器的一个扩展，用于文档分类的常用算法。

由统计学知，如果每个特征需要N个样本，那么对10个特征将需要$N^10$个样本，对于包含1000个特征的词汇表需要$N^1000$个样本。

如果特征直接互相独立，那么样本数就可以从$N^1000$减少到1000*N。独立指的是统计意义上的独立，即一个特征或者单词的出现的可能性与它和其他单词相邻没有关系。这个假设正式朴素贝叶斯分类器中的朴素(naive)的含义。

式朴素贝叶斯分类器另外一个假设就是：所有的特征值同等重要。

尽管上述建设存在一些小的瑕疵，但是朴素贝叶斯的实际效果很好。


## 5，使用Python进行文本分类

### 词汇表到向量的转换函数

{% highlight python %}
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec
{% endhighlight %}

看下结果

{% highlight python %}
In [6]: listOPosts, listClasses = bayes.loadDataSet()

In [7]: myVocablist = bayes.createVocabList(listOPosts)

In [8]: myVocablist
Out[8]:
['cute', 'love', 'help', 'garbage', 'quit', 'I', 'problems', 'is', 'park',
'stop', 'flea', 'dalmation', 'licks', 'food', 'not', 'him', 'buying', 'posting',
'has', 'worthless', 'ate', 'to', 'maybe', 'please', 'dog', 'how', 'stupid',
'so', 'take', 'mr', 'steak', 'my']

In [11]: bayes.setOfWords2Vec(myVocablist, listOPosts[0])
Out[11]:
[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
{% endhighlight %}

### 训练算法：从词向量计算概率

我们重写贝叶斯准则，将之前的x,y替换为W。用W表示一个向量，即由多少个数值组
成。这个例子中，数值个数与词汇表的单词个数相同。

$$p(c_i|W) = \frac{p(W|c_i)p(c_i)}{p(W)}$$

对于不同的分类p(W)是一致的，只需要比较 $p(W\|c_i)p(c_i)$即可。

通过类别i中的文档数，除以总文档数可以得出$p(c_i)$的概率

$p(W\|c_i)$概率的计算可以通过朴素贝叶斯假设，每个特征值都是独立的，可以分拆成
$p(w_0,w_1,...,w_n\|c_i)$，即可以用$p(w_0\|c_i)p(w_1\|c_i)...p(w_n\|c_i)$计算得出

朴素贝叶斯训练函数

{% highlight python %}
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = zeros(numWords); p1Num = zeros(numWords)
    p0Denom = 0.0; p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive
{% endhighlight %}

参数trainMatrix,trainCategory为文档矩阵，和类别。

使用函数进行分类前，还有解决函数中的一些缺陷。

* 计算$p(w_0\|c_i)p(w_1\|c_i)...p(w_n\|c_i)$时，如果一个概率为0，最后乘积也为0；可以将所有词的出现次数初始化为1，并将分母初始化为2
* 由于很多小的乘数相乘，会造成下溢出或者得不到正确答案。
* 可以通过对数避免，同时采用自然对数进行处理不会有任何损失。

给出f(x) 和 ln(f(x)) 的曲线，他们在相同的区域内同时增加或者减少，并且在相同点上取到极值。虽然最终取值不同，但不影响最终结果。

![bayes3.png](/files/bayes3.png)

修改后代码：
{% highlight python %}
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones()
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive
{% endhighlight %}

在到代数中有$ln(a*b)=ln(a)+ln(b)$

写最终分类和测试函数
{% highlight python %}
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
{% endhighlight %}

测试结果

{% highlight python %}
In [12]: bayes.testingNB()
['love', 'my', 'dalmation'] classified as:  0
['stupid', 'garbage'] classified as:  1
{% endhighlight %}

### 文档词袋模型

现在，我们将词的是否出现作为一个特征，叫做**词集模型**

能够表达单词出现次数的叫做 **词袋模型**

使用词袋模型
{% highlight python %}
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
{% endhighlight %}

## 6，示例：使用朴素贝叶斯过滤垃圾邮件

### 准备数据：切分文本

使用split()分割
{% highlight python %}
In [13]: mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'

In [14]: mySent.split()
Out[14]:
['This', 'book','is', 'the', 'best', 'book', 'on', 'Python', 'or', 'M.L.',
'I', 'have', 'ever', 'laid', 'eyes', 'upon.']
{% endhighlight %}
标点也被当做单词了。使用正则解决
{% highlight python %}
In [15]: import re
In [16]: regEx = re.compile(r'\W*')
In [17]: listOfTokens = regEx.split(mySent)
{% endhighlight %}

然后去掉空字符，并统一大小写
{% highlight python %}
 [tok.lower() for tok in listOfTokens if len(tok) > 0]
{% endhighlight %}

### 测试算法：使用朴素贝叶斯进行交叉验证

文件解析以及完整的垃圾邮件过滤测试函数
{% highlight python %}
def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = range(50); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error",docList[docIndex]
    print 'the error rate is: ',float(errorCount)/len(testSet)
    #return vocabList,fullText
{% endhighlight %}

50封邮件，其中40封作为训练集，10封作为测试集。

## 7，使用朴素贝叶斯分类器从个人广告中获取区域倾向

### 收集数据：导入RSS源

这里使用第三方包`feedparser`，可以使用`pip install feedparser`安装

使用下
{% highlight python %}
In [22]: ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
In [23]: len(ny['entries'])
Out[23]: 25
{% endhighlight %}

RSS源分类器以及删除高频词
{% highlight python %}
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V
{% endhighlight %}

实践证明，去掉高频词，会提供准确率。
{% highlight python %}
In [29]: ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')

In [30]: sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')

In [31]: vocabList, pSF, pNY = bayes.localWords(ny, sf)
the error rate is:  0.45

# 保留所有词汇
In [32]: reload(bayes)
Out[32]: <module 'bayes' from 'bayes.py'>

In [33]: vocabList, pSF, pNY = bayes.localWords(ny, sf)
the error rate is:  0.7
{% endhighlight %}

### 分析数据，显示地域相关词汇

{% highlight python %}
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -4.6 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -4.6 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]
{% endhighlight %}

运行：
{% highlight python %}
In [53]: bayes.getTopWords(ny, sf)
the error rate is:  0.4
SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**
enjoy
interests
going
smoke
share
man
friend
need
one
educated
fun
NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**
around
cool
don
well
very
even
ever
more
native
through
its
seeking
platonic
any
normal
email
much
upscale
make
hang
person
thanks
indonesian
know
about
getting
there
head
chat
other
age
{% endhighlight %}





from: [《机器学习实践》](https://github.com/pbharrin/machinelearninginaction)
