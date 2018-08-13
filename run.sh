#!/bin/sh
while true
do
    pkill -9 python
    /home/iis/anaconda3/envs/py3.6/bin/python /home/iis/wuling31715/captcha_generator/transform.py    
done
