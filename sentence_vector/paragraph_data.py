# -*- coding:utf-8
'''
    created at 2016-03-31
'''
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
import logging
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

title_content = []
title_name = []
title_id  = []
hashname_id = {}
# 使用jieba进行分词
def use_jieba():
    import jieba
    import codecs
    import re
    k = 0
    data_file = codecs.open('train.txt', 'w', 'utf-8')
    data_paragraph_name = codecs.open('paragraph_name.txt', 'w', 'utf-8')
    logging.info("start to write paragraph...")
    jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数
    for i in title_content:
        data_file.write('_*'+str(k)+' ')
        data_paragraph_name.write(title_name[k] + '\t' + title_id[k] + '\n')
        words = " ".join(jieba.cut(
            re.sub(u'\s|！|。|，|“|”|（|）|《|》|\(|\)|：|、',' ',i),
            cut_all=False))
        data_file.write(words+u'\n')
        k += 1
    data_file.close()
    data_paragraph_name.close()

def process_data(begin='2015-09-25', end='2015-12-25', num=100):
    import re
    from draw_data import draw_data
    title = draw_data()
    title_item = title.get_title_data(begin, end, num)
    logging.info("start to get data...")
    for ti in title_item:
        title_name.append(ti['title_content'])
        title_content.append(ti['title_text'])
        title_id.append(ti['_id'])
    use_jieba()

# 得到 paragraph vector
def get_vector():
    temp_vector = []
    vector_file = open('sentence_vectors.txt', 'r')
    for lines in vector_file.readlines():
        line = lines.strip().split(' ')
        tv = []
        for i in range(1,len(line)):
            tv.append(float(line[i]))
        temp_vector.append(tv)
    vector_file.close()
    name_file = open('paragraph_name.txt', 'r')
    k = 0
    for lines in name_file.readlines():
        line = lines.strip().split('\t')
        title_name.append(line[0])
        title_id.append(line[1])
        hashname_id[line[0]] = k
        k += 1
    name_file.close()
    return np.mat(temp_vector)

def findsimirlar():
    from operator import itemgetter
    p_vector = get_vector()
    while(1):
        inputname = raw_input("输入标题:")
        inputid = hashname_id[inputname]
        inputvec = p_vector[inputid]
        hash_sim = {}
        for i in range(np.shape(p_vector)[0]):
            #dist = np.linalg.norm(p_vector[i] - inputvec)
            #hash_sim[title_name[i]] = 1.0 / (1.0 + dist)
            tempc = p_vector[i] - inputvec
            hash_sim[title_name[i]] = np.trace(np.dot(tempc,tempc.T))**0.5
        scores = sorted(hash_sim.iteritems(), key = itemgetter(1))
        for i in range(50):
            print scores[i][0] + '\t' + str(scores[i][1])
        

def cluster(methor = 'DBSCAN'):
    from sklearn.cluster import DBSCAN
    p_vector = get_vector()
    if methor == 'DBSCAN':
        db = DBSCAN(eps=2.7, min_samples=3).fit(p_vector)
        labels = db.labels_
        print labels
        cluster_num = max(labels) + 1
        label_name = [[] for i in range(cluster_num)]
        for i in range(len(labels)):
            if labels[i] >= 0:
                label_name[labels[i]].append(title_name[i])
    for i in range(len(label_name)):
        print i
        for j in label_name[i]:
            print j

if __name__ == '__main__':
    import getopt
    try:
        opts, args = getopt.getopt(sys.argv[1:], "fc", ["getdata", "cluster="])
        for op, value in opts:
            if op == "--getdata":
                process_data()
            elif op == "--cluster":
                cluster(value)
            elif op == '-c':
                logging.info("默认聚类为dbscan...")
                cluster()
            elif op == "-f":
                findsimirlar()        
    except getopt.GetoptError:
        logging.info('参数错误..')
        system.exit()
