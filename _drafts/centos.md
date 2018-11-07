
## 安装系统依赖

centos 7.2 64位


yum groupinstall "Development Tools"
yum install gtk+-devel gtk2-devel

yum install openssl openssl-devel postgresql-devel -y
yum install sqlite-devel libffi-devel libxslt-devel -y
yum install libjpeg-devel freetype-devel -y

yum install cairo-devel libxml2-devel pango-devel pango libpng-devel freetype freetype-devel libart_lgpl-devel -y

yum install ntp vim-enhanced gcc gcc-c++ gcc-g77 flex bison autoconf automake bzip2-devel ncurses-devel zlib-devel libjpeg-devel libpng-devel libtiff-devel freetype-devel libXpm-devel gettext-devel  pam-devel -y

yum install -y rrdtool perl-rrdtool rrdtool-devel

yum install glibc-static -y

yum install gcc cmake make python-devel cairo-devel libxml2 libxml2-devel pango-devel pango libpng-devel freetype freetype-devel libart_lgpl-devel -y

yum install bzip2-devel zlib zlib-devel -y

yum install cyrus-sasl-devel cyrus-sasl-gssapi cyrus-sasl-md5 cyrus-sasl-plain -y

yum install screen -y

yum install lzo-devel -y

yum install mysql-devel

yum -y install python-pip
yum install pcre pcre-devel zlib-devel zlib openssl openssl-devel -y



## ipython

pip install ipython==5.3.0

IPYTHON 6要求python 3.2以上版本。


## github

edwordx

imouren@qq.com

m2 password

## virtualenv

pip install virtualenv

pip install virtualenvwrapper

which virtualenvwrapper.sh

vim ~/.bash_profile

append: source /usr/bin/virtualenvwrapper.sh

export WORKON_HOME=/opt/workon_home/

mkvirtualenv kaka -p python2.7


## memcached

```python
yum install memcached -y

/bin/systemctl start  memcached.service
```

## redis

```python
yum install tcl redis -y

/bin/systemctl start  redis.service
```


## tesseract-ocr
安装leptonica

```python
wget http://www.leptonica.org/source/leptonica-1.72.tar.gz
tar xvzf leptonica-1.72.tar.gz
cd leptonica-1.72/
./configure
make && make install
```

安装tesseract-ocr

```python
wget https://github.com/tesseract-ocr/tesseract/archive/3.04.zip
unzip 3.04.zip
cd tesseract-3.04/
./configure
make && make install
sudo ldconfig
```

部署模型

https://github.com/tesseract-ocr/tessdata

下载对应语言的模型文件

https://github.com/tesseract-ocr/tessdata.git

将模型文件移动到/usr/local/share/tessdata


export TESSDATA_PREFIX=/usr/local/share/tessdata/lang/

/usr/local/share/tessdata/configs/chars

内容
tessedit_char_whitelist 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ


