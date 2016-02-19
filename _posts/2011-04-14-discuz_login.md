---
layout: post
title:  "整合游戏账号登陆论坛"
date:   2011-04-14 12:24:31
categories: discuz
tags: discuz
---

* content
{:toc}

### 需求

游戏账号可以登录discuz论坛


### 思路
时间短，没有做API接口。直接使用数据库同步的方法。

* 用户登录游戏的时候，判断是否该账户在discuz_ucenter中，有则pass，没有则在uc_members中插入一条用户记录。
* 用户在修改游戏账号密码的时候，同时修改discuz_ucenter中对应该账户的密码。
* 用户在注册游戏账号的时候，同时写入discuz_ucenter中用户记录。

### 实现

#### 一、修改settings.py中的数据库配置，增加一个discuz论坛所在的数据库，键名称为"ucenter"

{% highlight python %}
DATABASES = {
    'default': {
        'ENGINE': 'mysql',
        'NAME': 'ppy',
        'USER': 'root',
        'PASSWORD': '123',
        'HOST': '',
    },
    'second': {
        'ENGINE': 'mysql',
        'NAME': 'ppy',
        'USER': 'root',
        'PASSWORD': '123',
        'HOST': ''
    },
    'ucenter': {
        'ENGINE': 'mysql',
        'NAME': 'discuz',
        'USER': 'root',
        'PASSWORD': '1232',
        'HOST': ''
    }
}
{% endhighlight %}

#### 二、程序代码改动

{% highlight python %}
#工具代码
def get_client_ip(request):
    try:
        real_ip = request.META['HTTP_X_FORWARDED_FOR']
        regip = real_ip.split(",")[0]
    except:
        regip = request.META.get['REMOTE_ADDR', '']
    return regip

def get_salt():
    samples = string.letters + string.digits
    salt = "".join(random.sample(samples, 6))
    return salt

#其他代码
    具体游戏相关，略

{% endhighlight %}


#### 三、discuz可能需要的改动

1，禁止用户注册，后台管理设置即可。

2，修改密码错误次数，方便测试。

打开 include 目录下的 misc.func.php，

找到`$return = (!$login ($timestamp – $login['lastupdate'] 900)) 4 max(0, 5 – $login['count']);`这句

`$login['lastupdate'] 900))`

这个是时间限制，900秒，就是15分钟，你可以改成60就是1分钟，输入错误了1分钟后可以登陆。

max(0, 5 – $login['count']);

这里的5就是错误次数，如果想让他永远不提示，就改999999999999

3，禁止用户修改密码

修改discuz根目录文件memcp.php

149行左右，修改为以下

{% highlight python %}
#$ucresult = uc_user_edit($discuz_user, $oldpassword, $newpassword, $emailnew, 0, $questionidnew, $answernew);
$ucresult = -7;
if($ucresult == -1) {
    showmessage('profile_passwd_wrong', NULL, 'HALTED');
} elseif($ucresult == -4) {
    showmessage('profile_email_illegal');
} elseif($ucresult == -5) {
    showmessage('profile_email_domain_illegal');
} elseif($ucresult == -6) {
    showmessage('profile_email_duplicate');
} elseif($ucresult == -7){
    showmessage('you can not change your password on this site.');
}
{% endhighlight %}

### 问题总结：

一、技术点

Discuz SNS及BBS的密码采用非明文方式，加密算法如下（采用mysql函数方式描述）：

`md5(concat(md5(‘password’),salt))`

二、系统问题

连接远程机器时候，有可能遭遇防火墙问题。

三、数据库问题

uc_members和cdb_members表结构中，username为char(15)，

如果用户名过长会有问题，可以修改username 为varchar(50)或者其他更合适的结构。
