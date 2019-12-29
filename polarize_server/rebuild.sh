#!/bin/bash

apachectl restart
tail -f /var/www/error.log
