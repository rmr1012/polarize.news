!# /bin/sh

ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no root@ssh.dennisren.com -p 1222 << EOF
  su
  git config --global user.email "rmr1012@gmail.com"
  cd /var/www/dennisren_web/
  ./deploy.sh
EOF
