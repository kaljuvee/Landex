# Land Index
## Install Requirements
Launch a browser on localhost:8501
```bash
git pull https://github.com/manelreghima/Landex.git
pip install -r requiments.txt
streamlit run Home.py
```
## AWS Deployment Instructions
### Install required packages on AWS Ubuntu
```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install nginx
```
### Set up your Streamlit app
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Install Nginx and Basic Configuration

```bash
sudo apt-get install nginx
sudo service nginx start
sudo service nginx status
sudo systemctl enable nginx
```
Open port 80 and port 443 on your system's firewall for HTTP and HTTPS traffic
```bash
sudo iptables -I INPUT -p tcp -m tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT -p tcp -m tcp --dport 443 -j ACCEPT
```
### Create the SSL Certificate 

```bash
sudo mkdir /etc/ssl/private
sudo chmod 700 /etc/ssl/private
 ```
 Generate a self-signed SSL certificate and private key
 ```bash
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /etc/ssl/private/nginx-selfsigned.key -out /etc/ssl/certs/nginx-selfsigned.crt
  ```
 Ensure that the SSL/TLS connection is secure and provides forward secrecy
  ```bash
sudo openssl dhparam -out /etc/ssl/certs/dhparam.pem 2048
   ```
Open ssl.conf file 
  ```bash
sudo nano /etc/nginx/conf.d/ssl.conf
   ```
Add the following configuration 
  ```bash
server {
	listen 443 http2 ssl;
	listen [::]:443 http2 ssl;
	ssl_certificate /etc/ssl/certs/nginx-selfsigned.crt;
	ssl_certificate_key /etc/ssl/private/nginx-selfsigned.key;
	ssl_dhparam /etc/ssl/certs/dhparam.pem;
 
 
	root /usr/share/nginx/html;
	location / {
  	proxy_set_header  	  Host $host;
  	proxy_set_header    	X-Real-IP $remote_addr;
  	proxy_set_header    	X-Forwarded-For $proxy_add_x_forwarded_for;
  	proxy_set_header    	X-Forwarded-Proto $scheme;
 
 
  	# Fix the “It appears that your reverse proxy set up is broken" error.
  	proxy_pass      	http://localhost:8501;
 
   proxy_read_timeout  90;
 
  	# WebSocket support
  	proxy_http_version 1.1;
  	proxy_set_header Upgrade $http_upgrade;
  	proxy_set_header Connection "upgrade";
 
}
	error_page 404 /404.html;
	location = /404.html {
	}
 
	error_page 500 502 503 504 /50x.html;
	location = /50x.html {
	}
}
```
### Configure Nginx to redirect all HTTP to HTTPS

(Create defaut.d if the file doesn’t exist)
```bash
sudo nano /etc/nginx/default.d/ssl-redirect.conf 
```
Add the following configuration 
```bash
return 301 https://$host$request_uri/;
```
### Create Trusted Certificates
```bash
sudo certbot certonly --webroot --webroot-path=/var/www/html --email manelreghima01@gmail.com --agree-tos --no-eff-email --staging -d landex.com -d www.landex.com
```
Test the Nginx configuration and restart the Nginx service
```bash
sudo nginx –t
sudo service nginx restart
```
### Create a systemd service for the Streamlit app
```bash
sudo nano /etc/systemd/system/streamlit-app.service
```
Add the following configuration:
```bash
[Unit]
Description=Streamlit App

[Service]
Type=simple
User=ubuntu
ExecStart=/home/ubuntu/venv/bin/streamlit run /home/ubuntu/Landex/Home.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl start streamlit-app
sudo systemctl status streamlit-app
sudo systemctl enable streamlit-app
```
Now, we can acces the HTTPS address
  
