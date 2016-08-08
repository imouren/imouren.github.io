---
layout: post
title:  "python AES加密解密"
date:   2015-09-08 10:05:31
categories:  python
tags:  python
---

* content
{:toc}



## 安装 pycrypto

windows:

http://www.voidspace.org.uk/python/pycrypto-2.6.1/

linux:

pip install pycrypto

## 代码

```python

BS = AES.block_size
pad = lambda s: s + (BS - len(s) % BS) * chr(BS - len(s) % BS)
unpad = lambda s : s[:-ord(s[len(s)-1:])]

import base64
from Crypto.Cipher import AES
from Crypto import Random

class AESCipher(object):
    def __init__( self, key ):
        self.key = key

    def encrypt( self, raw ):
        raw = pad(raw)
        iv = Random.new().read( AES.block_size )
        cipher = AES.new( self.key, AES.MODE_CBC, iv )
        return base64.b64encode( iv + cipher.encrypt( raw ) )

    def decrypt( self, enc ):
        enc = base64.b64decode(enc)
        iv = enc[:16]
        cipher = AES.new(self.key, AES.MODE_CBC, iv )
        return unpad(cipher.decrypt( enc[16:] ))


```