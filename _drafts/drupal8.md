### 安装PHP

wget http://am1.php.net/distributions/php-7.2.0.tar.gz

yum install gcc make gd-devel libjpeg-devel libpng-devel libxml2-devel bzip2-devel libcurl-devel -y

./configure --prefix=/usr/local/php-7.2 --with-config-file-path=/usr/local/php-7.2/etc --with-bz2 --with-curl --enable-ftp --enable-sockets --disable-ipv6 --with-gd --with-jpeg-dir=/usr/local --with-png-dir=/usr/local --with-freetype-dir=/usr/local --enable-gd-native-ttf --with-iconv-dir=/usr/local --enable-mbstring --enable-calendar --with-gettext --with-libxml-dir=/usr/local --with-zlib --with-pdo-mysql=mysqlnd --with-mysqli=mysqlnd --with-mysql=mysqlnd --enable-dom --enable-xml --enable-fpm --with-libdir=lib64 --enable-bcmath
make
make install


cp php.ini-production /usr/local/php-7.2/etc/php.ini
cp /usr/local/php-7.2/etc/php-fpm.conf.default /usr/local/php-7.2/etc/php-fpm.conf
cd /usr/local/php-7.2/etc/php-fpm.d
cp www.conf.default www.conf


/usr/local/php-7.2/sbin/php-fpm -R


netstat -nlt|grep 9000





for nginx
user root;

server
{
        listen       80;
        server_name  meizhan.funshoin.com;
        root   /root/meizhan;
        index  index.html index.htm index.php;

        location ~ .*\.(php|php5|php7)?$
        {
                fastcgi_pass  127.0.0.1:9000;
                fastcgi_index index.php;
                include /etc/nginx/fastcgi.conf;
                fastcgi_param SCRIPT_FILENAME /root/meizhan$fastcgi_script_name;
        }

        access_log /var/log/nginx/meizhan.log;

}