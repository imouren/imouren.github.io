---
layout: post
title:  "决策树"
date:   2015-11-26 09:05:31
categories: python 机器学习
tags: python
---

* content
{:toc}


调查表明决策树是最常用的数据挖掘算法。

KNN 可以完成很多分类任务，但它的最大缺点是无法给出数据的内在含义，决策树的主要优势就在于数据形式非常容易理解。

下面的过滤邮件流程图就可以做为一个决策树：

![决策树](http://chuantu.biz/t2/20/1448430086x-1566638320.png)


## 1. 决策树的构造

优点：计算复杂度不高，输出结果易于理解，对中间值的缺失不敏感，可以处理不相关特征数据。

缺点： 可能产生过度匹配问题

适用数据类型： 数值型和标称型

划分数据集的最大原则是：将无序的数据变得更加有序。


### 信息增益

划分数据集之前之后信息发生的变化称为信息增益，获得信息增益最高的特征是最好的选择。

集合信息的度量方式称为`香农熵`或者简称为`熵`。

信息增益：infomation gain

熵： entropy  为信息的期望值

信息的定义：
:   如果待分类的事务可能划分在多个分类中，则符合$x_i$的信息定义为

$$l(x_i)=-log_2^{p(x_i)}$$

其中$p(x_i)$是选择该分类的概率。

为了计算熵，我们需要计算所有类别所有可能值包含的信息期望值，

$$H = -\sum_{i=1}^np(x_i)log_2^{p(x_i)}$$

其中n为分类的数目

使用python计算香农熵
{% highlight python %}
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt
{% endhighlight %}

这里我们海洋生物是否鱼类做一个例子，看如何用熵来分类：

| 序号 |不浮出水面是否可以生存| 是否有脚蹼| 是否鱼类|
| -- | --| -- |--|
|1| 是|是| 是|
|2| 是|是| 是|
|3| 是|否| 否|
|4| 否|是| 否|
|5| 否|是| 否|

获得数据集：
{% highlight python %}
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels
{% endhighlight %}
进行测试：
{% highlight python %}
In [45]: myDat, labels = trees.createDataSet()

In [46]: myDat
Out[46]: [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]

In [47]: trees.calcShannonEnt(myDat)
Out[47]: 0.9709505944546686

In [48]: myDat[0][-1] = "maybe"

In [49]: myDat
Out[49]: [[1, 1, 'maybe'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]

In [50]: trees.calcShannonEnt(myDat)
Out[50]: 1.3709505944546687
{% endhighlight %}

熵越高，则混合数据越多。如上，新增加分类后，熵的值变大。

### 划分数据集

我们对每个特征划分数据集的结果计算一次信息熵，然后判断按照哪个特性划分数据集是最好的划分方式。

划分数据集：
{% highlight python %}
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
{% endhighlight %}
三个参数： 待划分的数据集，划分数据集的特征，特征返回值

测试下：
{% highlight python %}
In [51]: reload(trees)
Out[51]: <module 'trees' from 'trees.pyc'>

In [52]: myDat, labels = trees.createDataSet()

In [53]: myDat
Out[53]: [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]

In [55]: trees.splitDataSet(myDat, 0, 1)
Out[55]: [[1, 'yes'], [1, 'yes'], [0, 'no']]

In [56]: trees.splitDataSet(myDat, 0, 0)
Out[56]: [[1, 'no'], [1, 'no']]
{% endhighlight %}

接下来，遍历整个数据集，循环计算香农熵和分类函数，找到最好的特征划分方式。

{% highlight python %}
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer
{% endhighlight %}

解释下：
信息增益 = 熵（前一状态）- 熵（后一状态）
信息增益越大说明后一状态的熵约小，所以 信息增益越大 的划分最好。

看下，我们的数据，哪个划分最好：
{% highlight python %}
In [71]: reload(trees)
Out[71]: <module 'trees' from 'trees.pyc'>

In [72]: myDat, lables = trees.createDataSet()

In [73]: trees.chooseBestFeatureToSplit(myDat)
Out[73]: 0

In [74]: myDat
Out[74]: [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
{% endhighlight %}
看出是以“不浮出水面是否可以生存”合适，人工辨别也是这样。

### 构建决策树

原理如下：
:  得到原始数据集，基于最好的属性划分数据集，由于特征值可能多于两个，因此可能大于两个分支的划分。第一次划分后，数据传递到树分支的下一个节点，在这个节点上，我们再次划分。我们可以使用递归来处理。

递归结束的条件：程序遍历完所有划分数据集的属性，或者每个分支下的实例都具有相同分类。

如果数据已经处理完所有分类，但类标签仍然不是唯一的，此时我们需要决定如何定义叶子节点。通常使用多数表决的方式来处理。
{% highlight python %}
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
{% endhighlight %}

创建树的代码：
{% highlight python %}
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # 类别相同，停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]#stop splitting when all of the classes are equal
    # 遍历完所有特征值，多数表决
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree
{% endhighlight %}

测试下我们的数据构建的树：
{% highlight python %}
In [86]: reload(trees)
Out[86]: <module 'trees' from 'trees.py'>

In [87]: myDat, labels = trees.createDataSet()

In [88]: myTree = trees.createTree(myDat, labels)

In [89]: myTree
Out[89]: {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
{% endhighlight %}

## 使用Matplotlib 注解绘制树形图

决策树的主要优点是直观易于理解，如果不能将其直观的显示出来，就无法发挥其优势。

直接贴画图的代码吧，找时间专门看画图的时候，在详细看逻辑和方法。
{% highlight python %}
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

{% endhighlight %}

调用`treePlotter.createPlot(myTree)` 看下我们画出的决策树：

![mytree.png](http://chuantu.biz/t2/20/1448453097x-1566638320.png)


## 测试和存储分类器

使用决策树的分类函数：

{% highlight python %}
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel
{% endhighlight %}

通过入口key，并从lables获取index；通过向量的index值判断分支。

决策树的存储：

保留创建好的决策树，可以节约很多计算时间。存储的话，看自己的应用吧。

使用`pickle` 或者 `json` 持久化到文件，或者 直接存入 缓存 等等吧。

代码略。

## 例子：使用决策树预测隐形眼镜类型

隐形眼镜类型包括：硬材质，软材质，不适合佩戴

数据集`lenses.txt`很小，就列出了
{% highlight python %}
young   myope   no  reduced no lenses
young   myope   no  normal  soft
young   myope   yes reduced no lenses
young   myope   yes normal  hard
young   hyper   no  reduced no lenses
young   hyper   no  normal  soft
young   hyper   yes reduced no lenses
young   hyper   yes normal  hard
pre myope   no  reduced no lenses
pre myope   no  normal  soft
pre myope   yes reduced no lenses
pre myope   yes normal  hard
pre hyper   no  reduced no lenses
pre hyper   no  normal  soft
pre hyper   yes reduced no lenses
pre hyper   yes normal  no lenses
presbyopic  myope   no  reduced no lenses
presbyopic  myope   no  normal  no lenses
presbyopic  myope   yes reduced no lenses
presbyopic  myope   yes normal  hard
presbyopic  hyper   no  reduced no lenses
presbyopic  hyper   no  normal  soft
presbyopic  hyper   yes reduced no lenses
presbyopic  hyper   yes normal  no lenses
{% endhighlight %}

测试下：
{% highlight python %}
In [99]: fr = open("lenses.txt")

In [100]: lenses = [inst.strip().split("\t") for inst in fr.readlines()]

In [101]: lenseLabels = ["age", "prescript", "astigmatic", "tearRate"]

In [103]: lensesTree = trees.createTree(lenses, lenseLabels)

In [104]: lensesTree
Out[104]:
{'tearRate': {'normal': {'astigmatic': {'no': {'age': {'pre': 'soft',
      'presbyopic': {'prescript': {'hyper': 'soft', 'myope': 'no lenses'}},
      'young': 'soft'}},
    'yes': {'prescript': {'hyper': {'age': {'pre': 'no lenses',
        'presbyopic': 'no lenses',
        'young': 'hard'}},
      'myope': 'hard'}}}},
  'reduced': 'no lenses'}}

In [105]: treePlotter.createPlot(lensesTree)
{% endhighlight %}
![tree_test.png](http://chuantu.biz/t2/20/1448502339x-1566638323.png)



from: [《机器学习实践》](https://github.com/pbharrin/machinelearninginaction)