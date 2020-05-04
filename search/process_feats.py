import faiss
import json
import numpy as np
import sys
import time
import os

class Feature(object):
    """
    处理特征类：对所提取的特征进行一些操作
    """
    def __init__(self, featureType, featurePath):
        self.featureType = featureType      #特征的种类：人脸or行人
        self.featurePath = featurePath      #特征路径
        
		
    def index_build(self, indexfilename):
        """
        将特征构建索引库
        :param indedxfilename: 索引所保存的位置
        :return:
        """
        if self.featureType == "face": d = 128
        if self.featureType == "person": d = 256

        if os.path.exists(indexfilename):
            index_face_all = faiss.read_index(indexfilename)
            file = open(self.featurePath)
            json_data = json.load(file)
            features = []
            if d == 128:  # 表示读入的为人脸特征
                for i in range(len(json_data)):
                    features.append(json_data[i]['feature'])
            if d == 256:  # 表示读入的为行人特征，保存格式与人脸不同，遂用两段代码分别处理
                for person in json_data[0].keys():
                    features.append(json_data[0][person]['feature'])
            features = np.asarray(features).astype(np.float32)
            index_face_all.add(features)
            faiss.write_index(index_face_all, indexfilename)
        else:
            file = open(self.featurePath)
            json_data = json.load(file)
            features = []
            if d == 128:  # 表示读入的为人脸特征
                for i in range(len(json_data)):
                    features.append(json_data[i]['feature'])
                index = faiss.IndexFlatL2(d)
            if d == 256:  # 表示读入的为行人特征，保存格式与人脸不同，遂用两段代码分别处理
                for person in json_data[0].keys():
                    features.append(json_data[0][person]['feature'])
                index = faiss.IndexFlatIP(d)
            features = np.asarray(features).astype(np.float32)
            index.add(features)
            faiss.write_index(index, indexfilename)
        return

    def build_time_list(self, time_list_filename):
        """
        将特征构建时间列表
        :param time_list_filename: 时间列表所保存的位置
        :return:
        """
        if self.featureType == "face":
            if os.path.exists(time_list_filename):
                video_time_list = np.load(feature2TimeListPath, allow_pickle=True)
                video_time_list = list(video_time_list)
            else:
                video_time_list = []
            file = open(self.featurePath)
            json_data = json.load(file)
            for j in range(len(json_data)):
                tmp_dict = {}
                tmp_dict['video'] = os.path.splitext(self.featurePath.split('/')[-1])[
                    0]  # josn名：video1_person.json，json_path[0:5]:输出视频段video0
                tmp_dict['time'] = json_data[j]['time']
                video_time_list.append(tmp_dict)
            np.save(time_list_filename, video_time_list)
        elif self.featureType == "person":
            if os.path.exists(time_list_filename):
                video_time_list = np.load(time_list_filename)
            else:
                video_time_list = []
            file = open(self.featurePath)
            json_data = json.load(file)
            for person in json_data[0].keys():
                person_time = json_data[0][person]['time']
                video_time_list.append(
                    {'video': os.path.splitext(self.featurePath.split('/')[-1])[0], 'time': [person_time]})
            np.save(time_list_filename, video_time_list)
        else:
            print("Error!")

        return

            
if __name__ == '__main__':
    featureType = "face"    # 所处理特征的类型："face" or "person"
    featurePath = '/home/pengfan/project/projects_module/task4/feature_json/face/20180101.json' #所处理特征的保存路径
    indexfilename = '/home/pengfan/project/projects_module/task4/0test/test.index'      #特征构建好索引库后要保存的路径
    time_list_filename = '/home/pengfan/project/projects_module/task4/0test/test.npy'   #特征构建好时间列表后要保存的路径

    feature = Feature(featureType, featurePath)
    feature.index_build(indexfilename)        #构建特征索引库
    feature.build_time_list(time_list_filename)      #构建特征时间库


