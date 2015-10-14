---
layout: post
title:  "安装Jekyll"
date:   2015-09-15 11:05:31
categories: jekyll
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

### 运行jekyll
{% highlight bash %}
jekyll new yourblog # 创建自己的blog
jekyll serve -H 0.0.0.0 -P 4000
jekyll serve -H 0.0.0.0 -P 4000 --detach # 在后端运行
jekyll serve -H 0.0.0.0 -P 4000 --force_polling # 自动监控变化rebuild
{% endhighlight %}

通过 http://yourpcip:4000  来访问

### 目录结构
{% highlight bash %}
.
├── _config.yml
├── _drafts
|   ├── begin-with-the-crazy-ideas.textile
|   └── on-simplicity-in-technology.markdown
├── _includes
|   ├── footer.html
|   └── header.html
├── _layouts
|   ├── default.html
|   └── post.html
├── _posts
|   ├── 2007-10-29-why-every-programmer-should-play-nethack.textile
|   └── 2009-04-26-barcamp-boston-4-roundup.textile
├── _data
|   └── members.yml
├── _site
└── index.html
{% endhighlight %}

* `_config.yml`  配置文件
* `_drafts`  未发布文章放这
* `_posts`  发布文章放这
* `_site`  转换好的页面
