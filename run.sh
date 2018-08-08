#!/bin/sh

while true
do
    /home/iis/anaconda3/envs/py3.5/bin/python /home/iis/wuling31715/captcha_generator/test.py
    pkill -9 python
    sleep 5
done
