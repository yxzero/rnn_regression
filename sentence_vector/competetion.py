# -*-coding:utf-8-*-
'''
Created on 2016年4月16日

@author: yx
'''
from draw_data import draw_data
from draw_data import time_between
import logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO)

def gradp(x_time):
    a = 1
    for i in range(len(x_time)-1):
        if(x_time[i+1]>x_time[i]):
            a = (a<<1)|1
        else:
            a = a<<1
    return a

def compare_two(x_time, y_time):
    begin_i = 0
    end_i = min(len(x_time), len(y_time))
    for i in range(end_i):
        if ((x_time[i]>0) and (y_time[i]>0)):
            begin_i = i
            break
    if begin_i == 0:
        return ([],[])
    x_time = x_time[begin_i:end_i]
    y_time = y_time[begin_i:end_i]
    x_pos = gradp(x_time)
    y_pos = gradp(y_time)
    upup = x_pos & y_pos
    uplist = []
    xtemp = []
    ytemp = []
    for i in range(min(len(x_time), len(y_time))):
        if(upup&(1<<i)):
            xtemp.append(x_time[i])
            ytemp.append(y_time[i])
        else:
            if(len(xtemp) > 2) and (len(ytemp) > 2):
                uplist.append((xtemp, ytemp))
            xtemp = []
            ytemp = []
    oror = x_pos ^ y_pos
    orlist = []
    xtemp = []
    ytemp = []
    for i in range(min(len(x_time), len(y_time))):
        if(oror&(1<<i)):
            xtemp.append(x_time[i])
            ytemp.append(y_time[i])
        else:
            if(len(xtemp) > 2) and (len(ytemp) > 2):
                orlist.append((xtemp, ytemp))
            xtemp = []
            ytemp = []
    return (uplist, orlist)

def dataword(temp_k, begin_time=u"2015-09-25", last_time = u"2015-09-27"):
    import numpy as np
    getdate = draw_data()
    title_item = getdate.title_find_day(begin_time, last_time, 1000)
    news_list = list()
    tid = []
    for ti in title_item:
        x_time = [0 for i in range(350)]
        comment_item = getdate.db.CommentItem.find(
            {"title_id": ti["_id"]}
        )
        for ci in comment_item:
            tempk = time_between(begin_time, ci["time"], temp_k)
            if tempk < 350:
                x_time[tempk] += 1
        news_list.append(x_time[0:np.argmax(x_time)])
        tid.append((ti["_id"], comment_item.count()))
    compare_list = []
    for i in range(len(news_list)):
        for j in range(len(news_list)):
            if(i != j):
                uplist,orlist = compare_two(news_list[i],news_list[j])
                if len(uplist) or len(orlist):
                     compare_list.append([(tid[i],tid[j]),uplist,orlist])
    try:
        f = open('compare_list.txt', 'a')
        f.write(str(compare_list)+'\n')
        f.close()
    except Exception,e:
        logging.info(str(e))
    return compare_list

if __name__ == "__main__":
    import datetime
    mydraw = draw_data()
    begin_time = datetime.datetime.strptime('2015-09-25', '%Y-%m-%d')
    end_time = datetime.datetime.strptime('2015-09-26', '%Y-%m-%d')
    for i in range(60):
        print begin_time
        dataword(10, str(begin_time),str(end_time)) 
        begin_time = end_time
        end_time = end_time + datetime.timedelta(days = 1)
    #dataword(5, str(begin_time),str(end_time))
