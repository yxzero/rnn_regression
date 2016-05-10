#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: basic_onehours.py
#Author: yuxuan
#Created Time: 2016-04-24 19:00:28
############################
from draw_data import draw_data
import logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.INFO)

load_data = draw_data()
title_list = load_data.get_title_data('2015-09-25', '2015-12-25', 100)
logging.info("总共: " + str(title_list.count()))
all_file = open('all200.txt', 'w')
svm_train_file = open('basic_train.txt', 'w')
k = 0
for i in title_list:
    logging.info("process: " + str(k))
    k += 1
    titleid = i['_id']
    hours_count = load_data.one_hours_count(titleid, i['title_time']).count()
    all_count = load_data.title_comment(titleid).count()
    all_file.write(titleid + '\t' + str(hours_count) + '\t' +\
            str(all_count) + '\n')
    svm_train_file.write(str(all_count) + ' ' + '1:' + str(hours_count) + ' \n')

all_file.close()
svm_train_file.close()
