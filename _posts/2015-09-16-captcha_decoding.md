---
layout: post
title:  "破解验证码"
date:   2015-09-16 15:05:31
categories: python
tags: python captcha
---


具体参考 [decoding-captchas]

文件下载 [captcha.zip] python文件[crackit.py]

### 读取图片
{% highlight python %}
im = Image.open("captcha.gif")
{% endhighlight %}

### 图片降噪
{% highlight python %}
#(将图片转换为8位像素模式)
im = im.convert("P")

#打印颜色直方图
#print im.histogram()

#输出如下
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
1, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0
, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 1, 0, 0, 1,
0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 3, 1, 3, 3, 0,
 0, 0, 0, 0, 0, 1, 0, 3, 2, 132, 1, 1, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 15, 0
, 1, 0, 1, 0, 0, 8, 1, 0, 0, 0, 0, 1, 6, 0, 2, 0, 0, 0, 0, 18, 1, 1, 1, 1, 1, 2,
 365, 115, 0, 1, 0, 0, 0, 135, 186, 0, 0, 1, 0, 0, 0, 116, 3, 0, 0, 0, 0, 0, 21,
 1, 1, 0, 0, 0, 2, 10, 2, 0, 0, 0, 0, 2, 10, 0, 0, 0, 0, 1, 0, 625]
{% endhighlight %}

颜色直方图的每一位数字都代表了在图片中含有对应位的颜色的像素的数量。
每个像素点可表现256种颜色，你会发现白点是最多
（白色序号255的位置，也就是最后一位，可以看到，有625个白色像素）。
红像素在序号200左右，我们可以通过排序，得到有用的颜色。

{% highlight python %}
his = im.histogram()
values = {}

for i in range(256):
    values[i] = his[i]

for j,k in sorted(values.items(),key=lambda x:x[1],reverse = True)[:10]:
    print j,k

# 输出如下
255 625
212 365
220 186
219 135
169 132
227 116
213 115
234 21
205 18
184 15
{% endhighlight %}

我们得到了图片中最多的10种颜色，
其中 220 与 227 才是我们需要的红色和灰色，可以通过这一讯息构造一种黑白二值图片。

{% highlight python %}
im2 = Image.new("P", im.size, 255)
for x in range(im.size[0]):
    for y in range(im.size[1]):
        pix = im.getpixel((x, y))
        if pix == 220 or pix == 227: # these are the numbers to get
            im2.putpixel((x, y), 0)

im2.show()
{% endhighlight %}

### 图片切割

对图片进行纵向切割

{% highlight python %}
# 找图片位置
inletter = False
foundletter=False
start = 0
end = 0

letters = []

for x in range(im2.size[0]):
    for y in range(im2.size[1]):
        pix = im2.getpixel((x, y))
        if pix != 255:
            inletter = True
    if foundletter == False and inletter == True:
        foundletter = True
        start = x

    if foundletter == True and inletter == False:
        foundletter = False
        end = x
        letters.append((start, end))

    inletter=False
#print letters

# 保存图片
count = 0
for letter in letters:
    m = hashlib.md5()
    im3 = im2.crop((letter[0], 0, letter[1], im2.size[1]))
    m.update("{}{}".format(time.time(), count))
    im3.save("{}.gif".format(m.hexdigest()))
    count += 1
{% endhighlight %}

处理多个图片，构建一个图片集合。

### AI与向量空间图像识别

在这里我们使用向量空间搜索引擎来做字符识别，它具有很多优点：

+ 不需要大量的训练迭代
+ 不会训练过度
+ 你可以随时加入／移除错误的数据查看效果
+ 很容易理解和编写成代码
+ 提供分级结果，你可以查看最接近的多个匹配
+ 对于无法识别的东西只要加入到搜索引擎中，马上就能识别了。

当然它也有缺点，例如分类的速度比神经网络慢很多，它不能找到自己的方法解决问题等等。

{% highlight python %}
#用 Python 类实现向量空间：
import math

class VectorCompare:
    #计算矢量大小
    def magnitude(self, concordance):
        total = 0
        for word, count in concordance.iteritems():
            total += count ** 2
        return math.sqrt(total)

    #计算矢量之间的 cos 值
    def relation(self, concordance1, concordance2):
        relevance = 0
        topvalue = 0
        for word, count in concordance1.iteritems():
            if concordance2.has_key(word):
                topvalue += count * concordance2[word]
        return topvalue / (self.magnitude(concordance1) * self.magnitude(concordance2))

# 它会比较两个 python 字典类型并输出它们的相似度（用 0～1 的数字表示）

#将图片转换为矢量
def buildvector(im):
    d1 = {}
    count = 0
    for i in im.getdata():
      d1[count] = i
      count += 1
    return d1
{% endhighlight %}

### 加载训练集

直接使用给出的训练集
{% highlight python %}
iconset = ['0','1','2','3','4','5','6','7','8','9','0','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

imageset = []

for letter in iconset:
  for img in os.listdir('./iconset/%s/'%(letter)):
    temp = []
    if img != "Thumbs.db" and img != ".DS_Store":
      temp.append(buildvector(Image.open("./iconset/%s/%s"%(letter,img))))
    imageset.append({letter:temp})
return imageset
{% endhighlight %}

### 通过向量比较猜测图片

{% highlight python %}
guess = []
for image in imageset:
    for x, y in image.iteritems():
        if len(y) != 0:
            guess.append((v.relation(y[0], buildvector(im3)),x))
guess.sort(reverse=True)
print guess[0]
{% endhighlight %}


[crackit.py]: /files/crackit.py
[captcha.zip]: /files/python_captcha.zip
[decoding-captchas]: http://www.boyter.org/decoding-captchas/

