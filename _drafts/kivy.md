### kivy 安装（windows）

文档参考的是 Release 1.10.1.dev0

创建单独的安装环境

```python
执行Python出现LookupError: unknown encoding: cp65001错误
chcp 936 或者 chcp 1252


conda create --name kivy python=2.7
activate kivy
```

kivy安装

```python
# 安装工具
pip install --upgrade pip wheel setuptools

# 安装依赖

pip install docutils pygments pypiwin32 kivy.deps.sdl2 kivy.deps.glew
pip install kivy.deps.gstreamer

# 安装kivy
pip install kivy
pip install kivy_examples

# 第三方包
pip install cython
pip install pygame
pip install pyjnius
pip install plyer

# 测试运行下

python share\kivy-examples\demo\showcase\main.py

```


### 打包android

依赖

```python
pip install python-for-android
pip install buildozer  # 更方便
```

windows 需要安装Virtual Machine

下载给定的 https://kivy.org/#download 系统文件

建立共享目录在虚拟机位置 /media/fs_xx

尝试一下，进入你的app目录

```python
buildozer init  # 生成 buildozer.spec，根据自己的需要修改

buildozer -v android_new debug

https://kivy.org/#download
```

Ubuntu代理设置 用主机的代理


```python

vim ~/.bashrc

export http_proxy=http://username:password@proxyserver.net:port
export https_proxy=http://username:password@proxyserver.net:port

export _JAVA_OPTIONS='-Dhttp.proxyHost=host -Dhttp.proxyPort=port'
export _JAVA_OPTIONS='-Dhttps.proxyHost=host -Dhttps.proxyPort=port'

export JAVA_TOOL_OPTIONS='-Dhttp.proxyHost=host -Dhttp.proxyPort=port -Dfile.encoding=UTF-8'

export JAVA_TOOL_OPTIONS='-Dhttps.proxyHost=host-Dhttps.proxyPort=port -Dfile.encoding=UTF-8'



vim /etc/environment

export http_proxy=http://username:password@proxyserver.net:port
export https_proxy=http://username:password@proxyserver.net:port


vim /etc/apt/apt.conf

Acquire::http::Proxy "http://username:password@proxyserver.net:port";
Acquire::https::Proxy "http://username:password@proxyserver.net:port";

```


## notification 的错误

```python

https://github.com/kivy/plyer/issues/333

from plyer import notification as n
n.notify(title='title', message='m', ticker='r')

WindowsBalloonTip(**kwargs)
 TypeError: __init__() got an unexpected keyword argument 'ticker'

pip install -I https://github.com/kivy/plyer/zipball/master
```

## app的icon

把注释打开即可
icon.filename = %(source.dir)s/data/icon.png

