---
layout: post
title:  "python代码片段"
date:   2015-10-08 10:05:31
categories:  python
tags:  python
---

* content
{:toc}

## 不打印警告信息

```python
import warnings
warnings.filterwarnings("ignore")
```


## 数字格式化为百分百

```python

from numbers import Number

def as_percent(v, precision='0.2'):
    """Convert number to percentage string."""
    if isinstance(v, Number):
        return "{ {:{}%} }".format(precision).format(v)
    else:
        raise TypeError("Numeric type required")

as_percent(0.5)  # '50.00%'

```


## 线程池

https://www.metachris.com/2016/04/python-threadpool/

```python

import sys
IS_PY2 = sys.version_info < (3, 0)

if IS_PY2:
    from Queue import Queue
else:
    from queue import Queue

from threading import Thread


class Worker(Thread):
    """ Thread executing tasks from a given tasks queue """
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()

    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
            try:
                func(*args, **kargs)
            except Exception as e:
                # An exception happened in this thread
                print(e)
            finally:
                # Mark this task as done, whether an exception happened or not
                self.tasks.task_done()


class ThreadPool:
    """ Pool of threads consuming tasks from a queue """
    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        for _ in range(num_threads):
            Worker(self.tasks)

    def add_task(self, func, *args, **kargs):
        """ Add a task to the queue """
        self.tasks.put((func, args, kargs))

    def map(self, func, args_list):
        """ Add a list of tasks to the queue """
        for args in args_list:
            self.add_task(func, *args)

    def wait_completion(self):
        """ Wait for completion of all the tasks in the queue """
        self.tasks.join()


if __name__ == "__main__":
    from random import randrange
    from time import sleep

    # Function to be executed in a thread
    def wait_delay(d):
        print("sleeping for (%d)sec" % d)
        sleep(d)

    # Generate random delays
    delays = [(randrange(3, 7),) for i in range(50)]

    # Instantiate a thread pool with 5 worker threads
    pool = ThreadPool(5)

    # Add the jobs in bulk to the thread pool. Alternatively you could use
    # `pool.add_task` to add single jobs. The code will block here, which
    # makes it possible to cancel the thread pool with an exception when
    # the currently running batch of workers is finished.
    pool.map(wait_delay, delays)
    pool.wait_completion()

```

## 自动获取文件编码读取文件内容

EF BB BF是被称为 Byte order mark (BOM)的文件标记，用来指出这个文件是UTF-8编码

```python

import chardet
import codecs

bytes = min(32, os.path.getsize(filename))
raw = open(filename, 'rb').read(bytes)

if raw.startswith(codecs.BOM_UTF8):
    encoding = 'utf-8-sig'
else:
    result = chardet.detect(raw)
    encoding = result['encoding']

infile = codecs.open(filename, mode, encoding=encoding)
data = infile.read()
infile.close()

print(data)

```


## 加密解密例子

```python
import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES

class AESCipher(object):

    def __init__(self, key):
        self.bs = 32
        self.key = hashlib.sha256(key.encode()).digest()

    def encrypt(self, raw):
        raw = self._pad(raw)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(raw))

    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size:])).decode('utf-8')

    def _pad(self, s):
        return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s)-1:])]
```


## 线程池

```python

import time
import functools
from multiprocessing.dummy import Pool as ThreadPool


def clock(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        name = func.__name__
        arg_lst = []
        if args:
            arg_lst.append(', '.join(repr(arg) for arg in args))
        if kwargs:
            pairs = ['%s=%r' % (k, w) for k, w in sorted(kwargs.items())]
            arg_lst.append(', '.join(pairs))
        arg_str = ', '.join(arg_lst)
        print('[%0.8fs] %s(%s) -> %r ' % (elapsed, name, arg_str, result))
        return result
    return clocked


def io_heavy(n):
    time.sleep(0.5)
    return n ** 2


@clock
def parallel_func(numbers, threads=2):
    pool = ThreadPool(threads)
    results = pool.map(io_heavy, numbers)
    pool.close()
    pool.join()
    return results


@clock
def func(numbers):
    results = []
    for n in numbers:
        results.append(io_heavy(n))
    return results


if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5]
    parallel_result = parallel_func(numbers, 4)
    func_result = func(numbers)
    print parallel_result == func_result

```


## 在函数内部获取函数名称

```python

import sys
import inspect


def my_name():
    print '1', sys._getframe().f_code.co_name
    print '2', inspect.stack()[0][3]


def get_current_function_name():
    print '5', sys._getframe().f_code.co_name
    return inspect.stack()[1][3]


class MyClass:
    def function_one(self):
        print '3', inspect.stack()[0][3]
        print '4', sys._getframe().f_code.co_name
        print "6 %s.%s invoked" % (self.__class__.__name__, get_current_function_name())


if __name__ == '__main__':
    my_name()
    myclass = MyClass()
    myclass.function_one()


1 my_name
2 my_name
3 function_one
4 function_one
5 get_current_function_name
6 MyClass.function_one invoked
```

## 生成exe文件

```python

pyinstaller -F pyfile.py
```

### 访问局域网文件

```python
with open('\\\\FX-JSJ510\\vmis_pic\\filename', 'wb') as f:
    process_some(f)
```