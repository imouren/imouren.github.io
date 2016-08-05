---
layout: post
title:  "python调用java"
date:   2015-09-07 15:05:31
categories:  python
tags:  python
---

* content
{:toc}



from http://baojie.org/blog/2014/06/16/call-java-from-python/

Some of my notes on calling Java from Python, only lightly edited from the raw notes. Short, mostly installation script and hello world code, but should serve the purpose.

Short answer: Jpype works pretty well, but Pyjnius is faster and
simpler than JPype
Summary

2013-05-21T22:38:11 (PDT) Pyjnius is faster and simpler than JPype

    JCC, javabridge, Jpype and Jnius are all JNI wrappers.

2012-06-14T10:33:00 (PDT) Jpype works pretty well. I can call Stanford parser and OpenNLP from Python

2012-05-05T17:57:57 (PDT) close for now. At lease I can use Jpype. Reopen a Py4j task in the future if Jpype is not enough

Resources mentioned in this post are listed as visual bookmarks on Memect

