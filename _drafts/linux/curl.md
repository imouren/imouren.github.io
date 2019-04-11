# CURL命令 应用

来源：http://www.ruanyifeng.com/blog/2011/09/curl.html



## **一、查看网页源码**

直接在curl命令后加上网址，就可以看到网页源码。我们以网址www.sina.com为例（选择该网址，主要因为它的网页代码较短）

```python
# curl www.sina.com       

<html>
<head><title>301 Moved Permanently</title></head>
<body bgcolor="white">
<center><h1>301 Moved Permanently</h1></center>
<hr><center>nginx</center>
</body>
</html>
```

如果要把这个网页保存下来，可以使用`-o`参数，这就相当于使用wget命令了

```python
[root@localhost ~]# curl www.sina.com -o sina.html
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
178   178  178   178    0     0    933      0 --:--:-- --:--:-- --:--:--  1508
```

## **二、自动跳转**

有的网址是自动跳转的。使用`-L`参数，curl就会跳转到新的网址

也可以针对有302的网址

```python
$ curl -L www.sina.com  
```

键入上面的命令，结果就自动跳转为www.sina.com.cn

## **三、显示头信息**

`-i`参数可以显示http response的头信息，连同网页代码一起。

```python
[root@localhost ~]# curl -i www.sina.com


HTTP/1.1 301 Moved Permanently
Server: nginx
Date: Wed, 30 Jan 2019 02:52:06 GMT
Content-Type: text/html
Content-Length: 178
Connection: keep-alive
Location: http://www.sina.com.cn/
Expires: Wed, 30 Jan 2019 02:53:35 GMT
Cache-Control: max-age=120
Age: 31
Via: http/1.1 ctc.shanghai.ha2ts4.127 (ApacheTrafficServer/6.2.1 [cRs f ]), http/1.1 ctc.nanjing.ha2ts4.75 (ApacheTrafficServer/6.2.1 [cRs f ])
X-Via-Edge: 15488167266079ad066247c5e66ca39ddf757
X-Cache: HIT.75
X-Via-CDN: f=edge,s=ctc.nanjing.ha2ts4.77.nb.sinaedge.com,c=36.102.208.154;f=Edge,s=ctc.nanjing.ha2ts4.75,c=202.102.94.77

<html>
<head><title>301 Moved Permanently</title></head>
<body bgcolor="white">
<center><h1>301 Moved Permanently</h1></center>
<hr><center>nginx</center>
</body>
</html>
```

`-I`参数则是只显示http response的头信息

## **四、显示通信过程**

`-v`参数可以显示一次http通信的整个过程，包括端口连接和http request头信息

```python
[root@localhost ~]# curl -v www.sina.com
* About to connect() to www.sina.com port 80 (#0)
*   Trying 202.102.94.124... connected
* Connected to www.sina.com (202.102.94.124) port 80 (#0)
> GET / HTTP/1.1
> User-Agent: curl/7.19.7 (x86_64-redhat-linux-gnu) libcurl/7.19.7 NSS/3.27.1 zlib/1.2.3 libidn/1.18 libssh2/1.4.2
> Host: www.sina.com
> Accept: */*
> 
< HTTP/1.1 301 Moved Permanently
< Server: nginx
< Date: Wed, 30 Jan 2019 02:53:08 GMT
< Content-Type: text/html
< Content-Length: 178
< Connection: keep-alive
< Location: http://www.sina.com.cn/
< Expires: Wed, 30 Jan 2019 02:54:41 GMT
< Cache-Control: max-age=120
< Age: 27
< Via: http/1.1 ctc.shanghai.ha2ts4.127 (ApacheTrafficServer/6.2.1 [cRs f ]), http/1.1 ctc.nanjing.ha2ts4.75 (ApacheTrafficServer/6.2.1 [cRs f ])
< X-Via-Edge: 15488167886799ad066247c5e66ca51b0c8d6
< X-Cache: HIT.75
< X-Via-CDN: f=edge,s=ctc.nanjing.ha2ts4.63.nb.sinaedge.com,c=36.102.208.154;f=Edge,s=ctc.nanjing.ha2ts4.75,c=202.102.94.63
< 
<html>
<head><title>301 Moved Permanently</title></head>
<body bgcolor="white">
<center><h1>301 Moved Permanently</h1></center>
<hr><center>nginx</center>
</body>
</html>
* Connection #0 to host www.sina.com left intact
* Closing connection #0
```

如果你觉得上面的信息还不够，那么下面的命令可以查看更详细的通信过程

```python
$ curl --trace output.txt www.sina.com
或者
$ curl --trace-ascii output.txt www.sina.com
```

## **五、发送表单信息**

发送表单信息有GET和POST两种方法。GET方法相对简单，只要把数据附在网址后面就行。

```python
$ curl example.com/form.cgi?data=xxx
```

POST方法必须把数据和网址分开，curl就要用到--data参数。

```python
$ curl -X POST --data "data=xxx" example.com/form.cgi
```

如果你的数据没有经过表单编码，还可以让curl为你编码，参数是`--data-urlencode`。

```python
$ curl -X POST--data-urlencode "date=April 1" example.com/form.cgi
```

## **六、HTTP动词**

curl默认的HTTP动词是GET，使用`-X`参数可以支持其他动词

```python
$ curl -X POST www.example.com

$ curl -X DELETE www.example.com
```

## **七、文件上传**

假定文件上传的表单是下面这样：

```python
<form method="POST" enctype='multipart/form-data' action="upload.cgi">
　　　　<input type=file name=upload>
　　　　<input type=submit name=press value="OK">
　　</form>
```

你可以用curl这样上传文件

```python
$ curl --form upload=@localfilename --form press=OK [URL]
```

## **八、Referer字段**

有时你需要在http request头信息中，提供一个referer字段，表示你是从哪里跳转过来的。

```python
$ curl --referer http://www.example.com http://www.example.com
```

## **九、User Agent字段**

这个字段是用来表示客户端的设备信息。服务器有时会根据这个字段，针对不同设备，返回不同格式的网页，比如手机版和桌面版。

iPhone4的User Agent是

```python
Mozilla/5.0 (iPhone; U; CPU iPhone OS 4_0 like Mac OS X; en-us) AppleWebKit/532.9 (KHTML, like Gecko) Version/4.0.5 Mobile/8A293 Safari/6531.22.7
```

curl可以这样模拟

```python
$ curl --user-agent "[User Agent]" [URL]
```

## **十、cookie**

使用`--cookie`参数，可以让curl发送cookie

```python
$ curl --cookie "name=xxx" www.example.com
```

至于具体的cookie的值，可以从http response头信息的`Set-Cookie`字段中得到。

`-c cookie-file`可以保存服务器返回的cookie到文件，`-b cookie-file`可以使用这个文件作为cookie信息，进行后续的请求。

```python
$ curl -c cookies http://example.com
　　
$ curl -b cookies http://example.com
```

## **十一、增加头信息**

有时需要在http request之中，自行增加一个头信息。`--header`参数就可以起到这个作用

```python
$ curl --header "Content-Type:application/json" http://example.com
```

## **十二、HTTP认证**

```python
$ curl --user name:password example.com
```

