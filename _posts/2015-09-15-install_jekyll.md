---
layout: post
title:  "安装Jekyll"
date:   2015-09-15 11:05:31
categories: centos jekyll
tags: centos jekyll
---


### 更新ruby
{% highlight bash %}
curl --progress http://cache.ruby-lang.org/pub/ruby/2.1/ruby-2.1.2.tar.gz | tar xz
cd ruby-2.1.2
./configure
make
make install
yum install openssl
yum install ruby-devel
{% endhighlight %}

### 安装rubygems
{% highlight bash %}
wget https://rubygems.org/rubygems/rubygems-2.4.8.zip
unzip rubygems-2.4.8.zip
cd rubygems-2.4.8
ruby setup.rb
{% endhighlight %}

### 修改gem源
{% highlight bash %}
gem sources -a https://ruby.taobao.org/
gem sources -r https://rubygems.org/
{% endhighlight %}

### 安装jekyll
{% highlight bash %}
gem install jekyll
{% endhighlight %}

