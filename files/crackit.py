#-*- coding:utf8 -*-
import math
import os
import hashlib
import time
from PIL import Image


def get_white_pic(im):
    im = im.convert("P")
    im2 = Image.new("P", im.size, 255)
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            pix = im.getpixel((x, y))
            if pix == 220 or pix == 227: # these are the numbers to get
                im2.putpixel((x, y), 0)
    return im2


def get_small_pic(im2):
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

    imgs = []
    for letter in letters:
        im3 = im2.crop((letter[0], 0, letter[1], im2.size[1]))
        imgs.append(im3)
    return imgs


class VectorCompare:
    """
    它会比较两个 python 字典类型并输出它们的相似度（用 0～1 的数字表示
    """
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


#将图片转换为矢量
def buildvector(im):
  d1 = {}
  count = 0
  for i in im.getdata():
    d1[count] = i
    count += 1
  return d1


#加载训练集
def get_imageset():
    iconset = ['0','1','2','3','4','5','6','7','8','9','0','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

    imageset = []

    for letter in iconset:
      for img in os.listdir('./iconset/%s/'%(letter)):
        temp = []
        if img != "Thumbs.db" and img != ".DS_Store":
          temp.append(buildvector(Image.open("./iconset/%s/%s"%(letter,img))))
        imageset.append({letter:temp})
    return imageset


def guss_small_pic(im3, imageset):
    guess = []
    for image in imageset:
        for x, y in image.iteritems():
            if len(y) != 0:
                guess.append((v.relation(y[0], buildvector(im3)),x))
    guess.sort(reverse=True)
    return guess[0]


if __name__ == "__main__":
    v = VectorCompare()
    im = Image.open("captcha.gif")
    im2 = get_white_pic(im)
    imgs = get_small_pic(im2)
    imageset = get_imageset()
    result = []
    for img3 in imgs:
        r = guss_small_pic(img3, imageset)
        result.append(r[1])
    print "".join(result)
