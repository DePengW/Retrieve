import cv2
import pdb
import os
import faiss
import json
import numpy as np
import sys
import time
import requests
from flask import Flask, request
sys.path.append('/home/wangdepeng/xiangmuzu/project/projects_module/person')
sys.path.append('..')
from face.main import *
from person.main import *
from elasticsearch import Elasticsearch

class Information(object):
    """
        信息类：所查询的信息
    """

    def __init__(self, infoType, info):
        self.infoType = infoType    #所查找的类型：文字/人脸/行人
        self.info = info            #所查找的信息
        if(self.infoType == "face"):
            self.face_dimension = 128
        if(self.infoType == "person"):
            self.person_dimension = 256


    def face_search(self, face_feature, video_face_index_path, video_face_time_path):
        """
        对特定视频检索人脸
        :param input_face_feature: 人脸的特征
        :param video_face_index_path:数据库中某个视频索引的路径
        :param video_face_time_path:数据库中该视频有人脸的时间段
        :return: 所查找的人脸在数据库中出现的次数，及对应的视频段和时间段
        """
        if face_feature == -1:
            return -1
        else:
            face_feature = np.reshape(np.asarray(face_feature), [1, self.face_dimension]).astype(np.float32)
            face_index = faiss.read_index(video_face_index_path)
            D, I = face_index.search(face_feature, 5)
            dist_output = []
            index_output = []
            for i in range(5):
                dis_temp = D[0][i]
                if dis_temp < 0.12:
                    dist_output.append(dis_temp)
                    index_output.append(I[0][i])
            video_time_list_face = np.load(video_face_time_path, allow_pickle=True)
            output_result = []
            for j in range(len(index_output)):
                result = video_time_list_face[index_output[j]]
                for k in range(len(result['time'])):
                    per_video_dict = {}
                    if result['time'][k] == [None, None]:
                        continue
                    per_video_dict['video'] = result['video']
                    per_video_dict['time'] = result['time'][k]
                    per_video_dict['score'] = 1 / dist_output[j]
                    per_video_dict['tag'] = {'words': 0, 'face': 1, 'person': 0}
                    # if result['video']=='20171223':pdb.set_trace()
                    output_result.append(per_video_dict)
            ##删除时间间隔小于1s的时间段
            pop_list = []
            for i in range(len(output_result)):
                ms_start = time2num(output_result[i]['time'][0])
                ms_end = time2num(output_result[i]['time'][1])
                sub_cnt = ms_end - ms_start
                if sub_cnt < 3000:
                    pop_list.append(i)

            for i in range(len(pop_list)):
                del output_result[pop_list[-(i + 1)]]

            video_time_result = {"cnt": len(output_result), "result": output_result}
            return video_time_result

    def face_search_in_many_videos(self, index_path, face_time_path):
        """
        遍历文件夹，对所有视频进行检索人脸
        :param index_path: 视频人脸特征的索引存放的文件夹路径
        :param face_time_path:存储视频人脸时间戳信息的文件夹储路径
        :return: 所查找的人脸在数据库中出现的次数，及对应的视频段和时间段
        """
        if type(self.info) == str:
            face_path = './save_img/'+ self.info
            print('检索人脸为：' + face_path)
            face_feature = extractFaceFeaturefromImage(face_path)
        elif type(self.info) == list:
            face_feature = extractFaceFeaturefromImage2(np.array(self.info, dtype=np.uint8))
        else:
            return {'cnt': 0, 'result': []}
        result = {'cnt': 0, 'result': []}
        paths = os.listdir(index_path)      #索引路径
        for face_index_path in paths:
            videoname = os.path.splitext(face_index_path)[0]
            feature_type = index_path.split('/')[-2]#按文件夹分人脸和行人信息
            if feature_type!='face':continue
            video_face_index_path = index_path+face_index_path
            video_face_time_path = face_time_path + videoname + '.npy'
            out = self.face_search(face_feature, video_face_index_path, video_face_time_path)
            result['cnt'] += out['cnt']
            result['result'].extend(i for i in out['result'])
        return result

    def person_search(self, person_feature, person_index_path, video_time_list_path):
        """
       对特定视频检索行人
       :param input_face_feature: 行人的特征
       :param video_face_index_path:数据库中某个视频索引的路径
       :param video_face_time_path:数据库中该视频有行人的时间段
       :return: 所查找的行人在数据库中出现的次数，及对应的视频段和时间段
       """
        if person_feature.any() == -1:
            return -1
        else:
            person_feature = np.reshape(np.asarray(person_feature), [1, self.person_dimension]).astype(np.float32)
            person_index = faiss.read_index(person_index_path)
            D, I = person_index.search(person_feature, 5)
            dist_output = []
            index_output = []
            for i in range(5):
                dis_temp = D[0][i]
                if dis_temp > 0.5:  # 这里dis_temp相当于相似度
                    dist_output.append(dis_temp)
                    index_output.append(I[0][i])
            output_result = []
            for j in range(len(index_output)):
                video_time_list_person = np.load(video_time_list_path, allow_pickle=True)
                result = video_time_list_person[index_output[j]]
                # pdb.set_trace()
                for k in range(len(result['time'])):
                    per_video_dict = {}
                    per_video_dict['video'] = result['video']
                    per_video_dict['time'] = result['time'][k]
                    per_video_dict['score'] = dist_output[j]
                    per_video_dict['tag'] = {'words': 0, 'face': 0, 'person': 1}
                    output_result.append(per_video_dict)
            ##删除时间间隔小于1s的时间段
            pop_list = []
            for i in range(len(output_result)):
                ms_start = time2num(output_result[i]['time'][0])
                ms_end = time2num(output_result[i]['time'][1])
                sub_cnt = ms_end - ms_start
                if sub_cnt < 3000:
                    pop_list.append(i)
            for i in range(len(pop_list)):
                del output_result[pop_list[-(i + 1)]]

            video_time_result = {"cnt": len(output_result), "result": output_result}
            return video_time_result

    def person_search_in_many_videos(self, person_index_dir, person_time_dir):
        """
        遍历文件夹，对所有视频进行检索行人
        :param index_path: 视频行人特征的索引存放的文件夹路径
        :param face_time_path:存储视频行人时间戳信息的文件夹储路径
        :return: 所查找的人脸在数据库中出现的次数，及对应的视频段和时间段
        """
        if type(self.info) == str:
            input_person_path = './save_img/' + self.info
            print('检索人脸为：' + input_person_path)
            person_feature = extractPersonFeaturefromImage(input_person_path)
        elif type(self.info) == list:
            person_feature = extractPersonFeaturefromImage_needClassifyLoc(np.array(self.info, dtype=np.uint8))
        else:
            return {'cnt': 0, 'result': []}
        result = {'cnt': 0, 'result': []}
        paths = os.listdir(person_index_dir)
        paths = sorted(paths)
        for person_index_path in paths:
            videoname = os.path.splitext(person_index_path)[0]
            feature_type = person_index_dir.split('/')[-2]  # 按文件夹分人脸和行人信息
            if feature_type != 'person': continue
            video_person_index = person_index_dir + person_index_path
            video_person_time = person_time_dir + videoname + '.npy'
            out = self.person_search(person_feature, video_person_index, video_person_time)
            result['cnt'] += out['cnt']
            result['result'].extend(i for i in out['result'])
        return result

    def word_search(self):
        """
        检索关键字
        :param input_name:被检索关键字
        :return: 返回检索到的视频个数，视频名称，片段时间戳，被检索信息>类别（即"words"）
        """
        es = Elasticsearch(['localhost:9200'])
        query = {'query': {'match': {'sent': self.info}}, "size": 10000}
        allDoc = es.search(index='newsbroadcast_sub', body=query)  # 新闻联播视频的es索引
        result = []
        filter_cnt = 0
        save_cnt = 0
        pop_list = []
        for i in allDoc['hits']['hits']:
            result.append([i['_source']['time'], i['_source']['sent'], i['_source']['newsname'], i['_score']])
        original_count = len(result)
        for i in range(len(result)):
            str00 = result[i][0]
            second_start = int(str00[6:8])
            second_end = int(str00[-6:-4])
            sub_cnt = second_end - second_start
            if -1 < sub_cnt < 3 or sub_cnt < (-57):
                pop_list.append(i)
        for i in range(len(pop_list)):
            del result[pop_list[-(i + 1)]]

        cnt = len(result)
        vid_info = []
        for i in range(cnt):
            time = result[i][0].replace('-->', ',')
            time = time.split(' , ')
            vid_name = result[i][2].split('.')[0]
            vid_info.append(
                {'video': vid_name, 'time': time, 'score': result[i][3],
                 'tag': {'words': 1, 'face': 0, 'person': 0}})
        dict = {'cnt': cnt, 'result': vid_info}
        return dict


class Time(object):
    """
        对检索返回的结果进行操作
    """

    def __init__(self, all_video_inf):
        self.all_video_inf = all_video_inf

    def merge(self):
        """
        合并重叠的结果
        :return:
        """
        all_inf = []
        for res in self.all_video_inf:
            all_inf.extend(res)
        all_inf = sorted(all_inf, key=lambda all_inf: int(all_inf['video']) * 100000000 + time2num(
            all_inf['time'][0]))  # 按视频名和时间段排序
        i = 0
        new_video = []
        while i < len(all_inf) - 1:
            j = i
            while j < len(all_inf) - 1:
                if all_inf[j]['video'] != all_inf[j + 1]['video']: break
                if time2num(all_inf[j + 1]['time'][0]) - time2num(all_inf[j]['time'][1]) >= 1: break
                j += 1
            score = {'words': 0, 'face': 0, 'person': 0}
            tag = {'words': 0, 'face': 0, 'person': 0}
            for word in ['words', 'face', 'person']:
                for k in range(i, j + 1):
                    score[word] = max(score[word], all_inf[k]['tag'][word] * all_inf[k]['score'])
                    tag[word] = tag[word] | all_inf[k]['tag'][word]
            score = score['words'] + score['face'] + score['person']
            new_video.append(
                {'video': all_inf[j]['video'], 'time': [all_inf[i]['time'][0], all_inf[j]['time'][1]], 'score': score,
                 'tag': tag})
            cnt = len(all_inf)
            for k in range(j - i + 1):
                try:
                    del all_inf[j - k]
                except:
                    pdb.set_trace()
            i = j + 1
        all_inf.extend(new_video)
        all_inf = sorted(all_inf, key=lambda all_inf: int(all_inf['video']) * 100000000 + time2num(all_inf['time'][0]))
        cnt = len(all_inf)
        for i in range(cnt - 1):
            if all_inf[cnt - i - 1] == all_inf[cnt - i - 2]: del (all_inf[cnt - i - 1])
        for i in range(len(all_inf)):
            for j in all_inf[i + 1:]:
                if all_inf[i] == j:
                    pdb.set_trace()
        all_inf = sorted(all_inf, key=lambda all_inf: int(all_inf['video']) * 100000000 + time2num(all_inf['time'][0]))
        all_inf2 = []
        for i, inf in enumerate(all_inf):
            if (inf['tag']['person'] == 1) and (inf['tag']['face'] == 0) and (inf['tag']['words'] == 0):
                # if inf['video'] == '20181123':pdb.set_trace()
                if i != 0:
                    if ((all_inf[i - 1]['video'] == inf['video']) and (all_inf[i - 1]['tag']['face'] == 1) \
                            and (time2num(inf['time'][0]) - time2num(all_inf[i - 1]['time'][1]) < 1000)):
                        all_inf2.append(inf)
                        continue
                if i != len(all_inf) - 1:
                    if ((all_inf[i + 1]['video'] == inf['video']) and (all_inf[i + 1]['tag']['face'] == 1) \
                            and (time2num(all_inf[i + 1]['time'][0]) - time2num(inf['time'][1]) < 1000)):
                        all_inf2.append(inf)
            else:
                all_inf2.append(inf)
        all_inf = all_inf2
        all_inf = sorted(all_inf, key=lambda all_inf: int(all_inf['video']) * 100000000 + time2num(all_inf['time'][0]))
        cnt_num = len(all_inf)
        result = {'status': 'OK', 'cnt': cnt_num, 'inf': all_inf}
        return result

def time2num(time_str):
    """
    时间转数字
    :param time_str:
    :return:
    """
    h = int(time_str[:2])
    m = int(time_str[3:5])
    s = int(time_str[6:8])
    ms = int(time_str[9:])
    return int(h*3600000+m*60000+s*1000+ms)



"""
传输模块
"""
app = Flask(__name__)

@app.route('/register', methods=['POST'])
def register():
    all_video_inf = []
    res = json.loads(request.data)
    if ('key words' in res.keys()) and (res['key words'] != ''):
        print('输入关键字成功')
        wordInformation = Information("word", res['key words'])
        video_time_words = wordInformation.word_search()    #检索关键字
        all_video_inf.append(video_time_words['result'])
    if ('face' in res.keys()) and (res['face'] != ''):
        print('输入人脸图片成功')
        faceInformation = Information("face", res['face'])
        video_time_face = faceInformation.face_search_in_many_videos(faceFeature2IndexPath, faceFeature2TimeListPath)   #检索人脸
        all_video_inf.append(video_time_face['result'])
    if ('person' in res.keys()) and (res['person'] != ''):
        print('输入人体图片成功')
        personInformation = Information("person", res['person'])
        video_time_person = personInformation.person_search_in_many_videos(personFeature2IndexPath, personFeature2TimeListPath)     #检索行人
        all_video_inf.append(video_time_person['result'])
    else:
        return str('status:Error')

    #合并重复的检索结果
    allOfTime = Time(all_video_inf)
    result = allOfTime.merge()

    return str(result)



if __name__ == '__main__':
    faceFeature2IndexPath = '/home/pengfan/project/projects_module/task4/index/face/'   #人脸索引存储路径
    faceFeature2TimeListPath = '/home/pengfan/project/projects_module/task4/timelist/face/' #人脸时间列表存储路径
    personFeature2IndexPath = '/home/pengfan/project/projects_module/task4/index/person/'   #行人索引存储路径
    personFeature2TimeListPath = '/home/pengfan/project/projects_module/task4/timelist/person/' #行人时间列表存储路径
    app.run(port=9000, debug=True)