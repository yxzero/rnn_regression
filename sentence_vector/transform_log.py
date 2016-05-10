#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: transform_log.py
#Author: yuxuan
#Created Time: 2016-05-03 10:06:12
############################
import numpy as np
'''
f = open("svm_test.txt", 'r')
fw = open("svm_test_log.txt", 'w')
for i in f.readlines():
    line = i.strip().split()
    xline = line[1].split(':')
    fw.write(str(np.log10(float(line[0]))) + ' ' + xline[0] + ':' + str(np.log10(float(xline[1]))) + '\n')
f.close()
fw.close()
'''

import matplotlib.pyplot as plt
f = open("svm_test.txt", 'r')
test_x = []
for i in f.readlines():
    line = i.strip().split()
    test_x.append(float(line[0]))
f.close()
fre = open("result.txt", 'r')
target = []
for i in fre.readlines():
    target.append(float(i.strip()))
fre.close()
rmse = np.sqrt(((np.array(test_x) - np.array(target)) ** 2).mean())
print(rmse)
plt.plot(test_x, target, '.')
plt.show()


'''
#划分
import matplotlib.pyplot as plt
X_train = []
#f = open("all200.txt", 'r')
f = open("2.5hour.txt", 'r')
for i in f.readlines():
    line = i.strip().split('\t')
    X_train.append([float(line[1]), float(line[2]), float(line[3]),
        float(line[4])])
f.close()
#plt.plot(np.array(X_train)[:,0], np.array(X_train)[:,1], '.')
#plt.show()
valid_portion = 0.7
n_samples = 2079
sidx = np.random.permutation(n_samples)
n_train = int(np.round(n_samples * (1. - valid_portion)))
train_set = [X_train[s] for s in sidx[n_train:]]
test_set = [X_train[s] for s in sidx[:n_train]]

fw = open("svm_train.txt", 'w')
for i in train_set:
    fw.write(str(i[3]) + ' 1:' + str(i[0]) +' 2:' + str(i[1]) + ' 3:' + str(i[2]) + ' \n')
fw.close()

fw1 = open("svm_test.txt", 'w')
for i in test_set:
    fw1.write(str(i[3]) + ' 1:' + str(i[0]) +' 2:' + str(i[1]) + ' 3:' + str(i[2]) + ' \n')
fw1.close()
'''
