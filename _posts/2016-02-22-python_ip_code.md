---
layout: post
title:  "python linux下 IP操作函数"
date:   2015-09-02 15:05:31
categories: 工具代码 python
tags: 工具代码 python
---

* content
{:toc}



{% highlight python %}
import socket
import fcntl
import struct


def get_ip_address(ifname='eth0'):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ipaddr = socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15]))[20:24])
    return ipaddr


def get_netmask(ifname='eth0'):
    return socket.inet_ntoa(fcntl.ioctl(
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
        35099,
        struct.pack('256s', ifname))[20:24])


def get_default_gateway_linux():
    with open("/proc/net/route") as fh:
        for line in fh:
            fields = line.strip().split()
            if fields[1] != '00000000' or not int(fields[3], 16) & 2:
                continue
            return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))


def get_dns(dns_number="1"):
    counter = 0
    with open("/etc/resolv.conf") as f:
        content = f.readlines()
        for line in content:
            if "nameserver " in line:
                counter= counter+1
                if (str(counter) == dns_number):
                    return line[11:len(line)-1];



{% endhighlight %}


