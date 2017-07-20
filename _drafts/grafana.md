### 安装和启动

```python
wget https://s3-us-west-2.amazonaws.com/grafana-releases/release/grafana-4.4.1-1.x86_64.rpm
sudo yum localinstall grafana-4.4.1-1.x86_64.rpm

sudo service grafana-server start  # 端口3000 默认用户名和组为admin

sudo /sbin/chkconfig --add grafana-server # 开机启动
```

### 插件安装

点击 install apps & plugs 就可以进入帮助页面进行安装了

例如：安装zabbix等

```python
grafana-cli plugins install alexanderzobnin-zabbix-app
grafana-cli plugins install grafana-piechart-panel
grafana-cli plugins install grafana-clock-panel
grafana-cli plugins install jdbranham-diagram-panel
grafana-cli plugins install natel-discrete-panel
grafana-cli plugins install grafana-worldmap-panel
grafana-cli plugins install savantly-heatmap-panel
```

需要重启才会生效
