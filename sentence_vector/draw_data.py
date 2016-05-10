# -*-coding:utf-8-*-
'''
Created on 2015年9月29日

@author: yx
'''
import pymongo
import datetime
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def time_between(s_time, later_time, time_k): #time_k 步长几分钟
    ck = 60 / time_k
    minus = datetime.datetime.strptime(later_time, "%Y-%m-%d %H:%M") -\
        datetime.datetime.strptime(s_time, "%Y-%m-%d %H:%M:%S")
        #datetime.datetime.strptime(s_time, "%Y-%m-%d")
    tempk = minus.days * 24 * ck + minus.seconds / (60 * time_k)
    return tempk

class draw_data():
    def __init__(self):
        self.mongo_uri = 'mongodb://localhost:27017/'
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client['qqcomment']

    def close_db(self):
        self.client.close()

    def return_group(self, begin_time, last_time):
        return self.db.CommentItem.aggregate([
                    {"$match": {"time": { "$gt": begin_time, "$lt": last_time } }},
                    {"$group": {"_id": "$title_id", "count":{"$sum": 1}}}
                ]) 

    def get_title_data(self, begin_time, last_time, limit_num):
        return self.db.TitleItem.find(
            {"num": { "$gt": limit_num }, "title_time": { "$gt": begin_time, "$lt": last_time } }
        ).sort([("title_time", pymongo.ASCENDING)])

    def get_comment_data(self, begin_time, last_time):
        return self.db.CommentItem.find( {"time": { "$gt": begin_time, "$lt":
            last_time } } )

    def title_comment(self, titleid):
        return self.db.CommentItem.find( {"title_id" : titleid} )

    def one_hours_count(self, titleid, title_time):
        onehours_time = datetime.datetime.strptime(title_time, "%Y-%m-%d %H:%M")
        '''
        if(onehours_time.hour < 7):
            onehours_time = onehours_time + datetime.timedelta(hours=(8.5-onehours_time.hour))
        else:
        '''
        onehours_time = onehours_time + datetime.timedelta(hours=1.5)
        return self.db.CommentItem.find({"title_id":titleid, "time": {"$lt":
            str(onehours_time)}})

    def draw_topic(self, topic_set, temp_k, begin_time, last_time):
        from matplotlib import pyplot as plt
        import numpy as np
        plt.title(begin_time + u" 到 " + last_time)
        plt.xlabel("per " + str(temp_k) + " mins")
        '''
        color_list = ['#990033','#FF66FF','#660099','#CC9909',
        '#33CCCC','#000000','#CC6699','#FF33FF','#993366','#000CC0','#333366'
        ,'#330099','#330033','#FF3300','#CC6600','#33FF33','#006600','#339900','#33CC66']
        '''
        color_list = ['b-','g-','r-','c-','m-','y-','k-','b--'
        ,'g--','r--','c--','m--','y--','k--']
        color_l = 0
        endtime = time_between(begin_time, last_time+u' 00:00' , temp_k)
        logging.info("总共： " + str(len(topic_set)))
        for topic in range(len(topic_set)):
            logging.info("process " + str(topic) + '...')
            x_time = dict()
            post_time = float('inf')
            for j in topic_set[topic]:
                ti = self.db.TitleItem.find({'_id': j})
                post_time = min(post_time, time_between(begin_time, ti[0]["title_time"],
                    temp_k))
            if (post_time > endtime) or (post_time) < 0:
                continue
            for i in range(post_time, endtime):
                x_time[i] = 0
            for j in topic_set[topic]:
                comment_item = self.db.CommentItem.find(
                    {"title_id": j}
                ).sort([("time", pymongo.ASCENDING)])
                if comment_item.count()== 0:
                    continue 
                try:
                    tempk = time_between(begin_time, comment_item[0]["time"], temp_k)
                except Exception,e:
                    continue
                b_time = datetime.datetime.strptime(begin_time, "%Y-%m-%d")
                ptime = b_time + datetime.timedelta(seconds=tempk*60)
                for ci in comment_item:
                    #tempk = time_between(begin_time, ci["time"], temp_k)
                    ctime = datetime.datetime.strptime(ci["time"], "%Y-%m-%d %H:%M")
                    while(ctime > ptime):
                        ptime = ptime + datetime.timedelta(seconds=temp_k*60)
                        tempk += 1
                    if tempk in x_time:
                        x_time[tempk] += 1
                    '''
                    if tempk in x_time:
                        x_time[tempk] += 1
                    '''
            printt = max(x_time.items(), key=lambda x:x[1])
            plt.plot(x_time.keys(), x_time.values(), color_list[color_l%14])
            plt.text(printt[0], printt[1], str(topic))
            color_l += 1
        plt.show()    

    def draw_day(self, temp_k, begin_time=u"2015-09-25", last_time = u"2015-09-27"):
        from matplotlib import pyplot as plt
        import numpy as np
        alphy = 24 * (60 / temp_k)
        plt.title(begin_time + u" 到 " + last_time)
        plt.xlabel("per " + str(temp_k) + " mins")
        color_list = ['b-','g-','r-','c-','m-','y-','k-','b--'
            ,'g--','r--','c--','m--','y--','k--'] 
        titile_item = self.db.TitleItem.find(
            {"num": { "$gt": 2000 }, "title_time": { "$gt": begin_time, "$lt": last_time } }
        ).sort([("title_time", pymongo.ASCENDING)])
        color_l = 0
        title_i = 0
        for ti in titile_item:
            title_i += 1
            print str(title_i) +' ' + ti["title_content"] + ' ' + ti["title_time"]
            post_time = time_between(begin_time, ti["title_time"], temp_k)
            x_time = {post_time: 0}
            for i in range(post_time, post_time + alphy):
                x_time[i] = 0
            comment_item = self.db.CommentItem.find(
                {"title_id": ti["_id"]}
            )
            for ci in comment_item:
                tempk = time_between(begin_time, ci["time"], temp_k)
                if tempk in x_time:
                    x_time[tempk] += 1
            #for temp_j in range(post_time+1, post_time+alphy): #求和
                #x_time[temp_j] += x_time[temp_j-1]
            plt.plot(x_time.keys(), x_time.values(), color_list[color_l%14])
            printt = max(x_time.items(), key=lambda x:x[1])
            plt.text(printt[0], printt[1], ti["title_content"])
            color_l += 1
        plt.show()
        #plt.figure(figsize=(120,60))
        #plt.savefig('./2000_every_day/' + begin_time + u" 到 " + last_time + '.png', dpi = 150)
        
if __name__ == "__main__":
    import datetime
    mydraw = draw_data()
    begin_time = datetime.datetime.strptime('2015-09-25', '%Y-%m-%d')
    end_time = datetime.datetime.strptime('2015-09-26', '%Y-%m-%d')
    for i in range(60):
        mydraw.draw_day(30, str(begin_time),str(end_time)) 
        begin_time = end_time
        end_time = end_time + datetime.timedelta(days = 1)
