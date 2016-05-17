#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: process7day.py
#Author: yuxuan
#Created Time: 2016-05-05 10:15:29
############################
#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: process7day.py
#Author: yuxuan
#Created Time: 2016-05-05 10:15:29
############################
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
import logging
import numpy as np
from draw_data import draw_data
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def TOPk(x_dict, k):
    import heapq
    from operator import itemgetter
    topklist = heapq.nlargest(k, x_dict.iteritems(), key=itemgetter(1))
    return topklist

def dataword(temp_k, all_days, each_item, begin_time=u"2015-09-25", last_time = u"2015-09-27"):
    import numpy as np
    getdate = draw_data()
    title_item = getdate.get_title_data(begin_time, last_time, 100)
    news_list = list()
    today_item = {}
    for i in title_item:
        today_item[i['_id']] = datetime.datetime.strptime(i['title_time'], "%Y-%m-%d %H:%M")
    for i in today_item.keys():
        onehours_time = today_item[i] + datetime.timedelta(hours=2.5)
        temp_befor = getdate.return_group(begin_time, str(onehours_time))
        temp_dict = {}
        for j in temp_befor:
            if j["_id"] in today_item:
                j_time = today_item[j["_id"]]
                minus = 1.0*(onehours_time-j_time).days*24*60 + \
                        1.0*(onehours_time-j_time).seconds/60
                temp_dict[j["_id"]] = [j["count"]/minus, j["count"]]
        topk = TOPk(temp_dict, 3)
        if len(topk) == 3:
            each_item[i] = topk
    all_days.append(today_item)
    print(len(today_item))
    print(len(each_item))

def read_sourse(each_item):
    idfile = open("paragraph_name.txt", 'r')
    id_x = []
    for i in idfile.readlines():
        id_x.append(i.strip().split('\t')[1])
    idfile.close()

    vectorfile = open("sentence_vectors.txt", 'r')
    vector_list = []
    for lines in vectorfile.readlines():
        line = lines.strip().split(' ')
        tv = []
        for i in range(1,len(line)):
            tv.append(float(line[i]))
        vector_list.append(tv)
    vectorfile.close()

    for i in range(len(id_x)):
        if id_x[i] in each_item:
            each_item[id_x[i]].append(vector_list[i])

    hours_file = open("2.5hour.txt", 'r')
    for i in hours_file.readlines():
        line = i.strip().split('\t')
        if float(line[3]) < 5:
            continue
        if line[0] in each_item:
            '''
            a1 = 0.0
            a2 = 0.0
            a3 = 0.0
            if float(line[2]) >= 1:
                a2 = np.log10(float(line[2]))
            if float(line[3]) >= 1:
                a3 = np.log10(float(line[3]))
            if float(line[1]) >= 1:
                a1 = np.log10(float(line[1]))
            if float(line[4]) < 20:
                continue
            '''
            each_item[line[0]].append([float(line[1]), float(line[2]), float(line[3]), float(line[4])])
            #each_item[line[0]].append([a1, a2, a3, np.log10(float(line[4]))])
    hours_file.close()

def find_most_simirlar(cid, find_days_id, each_item):
    from operator import itemgetter
    c_vector = np.array(each_item[cid][3])
    hash_sim = {}
    for i in find_days_id:
        #dist = np.linalg.norm(p_vector[i] - inputvec)
        #hash_sim[title_name[i]] = 1.0 / (1.0 + dist)
        if i not in each_item or len(each_item[i])!=5:
            continue
        i_vector =  np.array(each_item[i][3])
        tempc = c_vector - i_vector
        hash_sim[i] = np.dot(tempc,tempc.T) 
    if len(hash_sim) > 0:
        scores = min(hash_sim.iteritems(), key = itemgetter(1))
        return scores[1], each_item[scores[0]]
    else:
        return 10.0,[0,0,0,0,[0,0,0,0]]

def final_data(all_days, each_item):
    from sklearn import preprocessing
    X_data_lstm = []
    X_data_c = []
    X_data_s = []
    Y_data = []
    cidk = 0
    cidv = 0
    cidk0 = 0
    for i in range(6, len(all_days)):
        for cid in all_days[i]:
            #前6天相似的
            if cid not in each_item:
                cidk += 1
                continue
            if len(each_item[cid]) != 5:
                cidv += 1
                continue
            cid0 = each_item[cid][0][0]
            cid1 = each_item[cid][1][0]
            cid2 = each_item[cid][2][0]
            if (cid0 not in each_item or len(each_item[cid0]) != 5 
                    or cid1 not in each_item or len(each_item[cid1]) != 5
                    or cid2 not in each_item or len(each_item[cid2]) != 5):
                cidk0 += 1
                continue
            xmintemp = []
            xminscore = []
            #for j in range(12, 0, -1):
            for j in range(1, 7, 1):
                scores, onehours = find_most_simirlar(cid, all_days[i-j], 
                        each_item)
                xminscore.append(scores)
                xmintemp.append([onehours[4][0],onehours[4][1],onehours[4][2],onehours[4][3]])
                #xmintemp.append([onehours[4][3]])
            X_data_lstm.append(xmintemp)
            X_data_c.append([
                #each_item[cid][0][1][0],each_item[cid][0][1][1],
                #each_item[cid][1][1][0],each_item[cid][1][1][1],
                #each_item[cid][2][1][0],each_item[cid][2][1][1],
                each_item[cid][4][0], each_item[cid][4][1], each_item[cid][4][2], 1.0])
            xminscore = (1-preprocessing.normalize(xminscore)).tolist()
            X_data_s.append(xminscore)
            '''
            xmintemp = []
            xmintemp.append([each_item[cid][0][1][0],each_item[cid][0][1][1],1.0, each_item[cid][4][0]])
            xmintemp.append([each_item[cid][1][1][0],each_item[cid][1][1][1],1.0, each_item[cid][4][0]])
            xmintemp.append([each_item[cid][2][1][0],each_item[cid][2][1][1],1.0, each_item[cid][4][0]])
            xtemp.append(xmintemp)
            '''
            Y_data.append(each_item[cid][4][3])
    print(cidk)
    print(cidv)
    print(cidk0)
    return (X_data_lstm,X_data_c,X_data_s),Y_data
'''
    each_item:0,1,2是与当前一天的平均量大的;3是docment2id;4是[1小时，全部]
'''

if __name__ == "__main__":
    import datetime
    all_days = []
    each_item = {}
    '''
    begin_time = datetime.datetime.strptime('2015-09-25', '%Y-%m-%d')
    end_time = datetime.datetime.strptime('2015-09-26', '%Y-%m-%d')
    for i in range(91):
        print begin_time
        dataword(10, all_days, each_item, str(begin_time),str(end_time)) 
        begin_time = end_time
        end_time = end_time + datetime.timedelta(days = 1)
    allfile = open("allday1.list", 'w')
    allfile.write(str(all_days))
    allfile.close()
    each_file = open("each1.top", 'w')
    each_file.write(str(each_item))
    each_file.close()
    '''
    allfile = open("allday1.list", 'r')
    all_days_str = allfile.read()
    allfile.close()
    each_file = open("each1.top", 'r')
    each_item_str = each_file.read()
    each_file.close()
    all_days = eval(all_days_str)
    each_item = eval(each_item_str)
    
    read_sourse(each_item)
    allfile = open("allday2.list", 'w')
    allfile.write(str(all_days))
    allfile.close()
    each_file = open("each2.top", 'w')
    each_file.write(str(each_item))
    each_file.close()

    allfile = open("allday2.list", 'r')
    all_days_str = allfile.read()
    allfile.close()
    each_file = open("each2.top", 'r')
    each_item_str = each_file.read()
    each_file.close()
    all_days = eval(all_days_str)
    each_item = eval(each_item_str)
    print(len(each_item))
    X_data,Y_data = final_data(all_days, each_item)
    print(len(Y_data))
    '''

    allfile = open("allday.list", 'r')
    all_days_str = allfile.read()
    allfile.close()
    each_file = open("each.top", 'r')
    each_item_str = each_file.read()
    each_file.close()
    all_days = eval(all_days_str)
    each_item = eval(each_item_str)
    '''
    xfile = open("X1.data", 'w')
    xfile.write(str(X_data))
    xfile.close()
    yfile = open("Y1.data", 'w')
    yfile.write(str(Y_data))
    yfile.close()
