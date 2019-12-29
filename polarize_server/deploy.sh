#! /bin/sh
git fetch --all
git reset --hard origin/master

chown -R www-data:www-data /var/www/polarize_server/
sed -i -e 's/DEBUG=True/DEBUG=False/g' polarize_server/settings.py
githash=$(git rev-parse HEAD| cut -c 1-5)
sed -i -e "s/MYVERSION=random.randint(1,100)/MYVERSION=\'${githash}\'/g" polarize_server/settings.py


