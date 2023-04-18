# Landex
## Install Nginx and Basic Configuration

```bash
sudo apt-get install nginx
sudo service nginx start
sudo service nginx status
sudo systemctl enable nginx
```
open port 80 and port 443 on your system's firewall for HTTP and HTTPS traffic
```bash
sudo iptables -I INPUT -p tcp -m tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT -p tcp -m tcp --dport 443 -j ACCEPT
```
