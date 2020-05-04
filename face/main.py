# -*- coding:utf-8 -*-
from __future__ import print_function
import cv2
import numpy as np
import time,os,json
import dlib
import glob
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# global person_count
#global all_output_information
# global curent_time
person_count = 0             # 记录视频检测出的总人数
all_output_information = {}  # 记录所有人物的输出信息
current_time = ''            # 记录开始结束时间的中介

detector = dlib.get_frontal_face_detector()     # 模型加载
currentpath = os.path.dirname(os.path.abspath(__file__))
face_rec_model = dlib.face_recognition_model_v1(currentpath + '/models/dlib_face_recognition_resnet_model_v1.dat')
shape_predictor = dlib.shape_predictor(currentpath + '/models/shape_predictor_68_face_landmarks.dat')
cascade = cv2.CascadeClassifier(currentpath + "/models/haarcascade_frontalface_alt.xml")

def init_global_variable():
    person_count = 0  # 记录视频检测出的总人数
    all_output_information = {}  # 记录所有人物的输出信息
    current_time = ''  # 记录开始结束时间的中介

    # print(person_count)
    # print(all_output_information)

class information(object):
    """
    一个头像一个输出信息对象存取所有信息
    """
    def __init__(self, id, start_time, end_time, start_time_ms, end_time_ms, box, feature):
        self.id = id
        self.start_time = start_time
        self.start_time_ms = start_time_ms
        self.box = box
        self.feature = feature
        self.end_time = end_time
        self.end_time_ms = end_time_ms
        self.show_again = False
        self.time_show = []
        self.time_show_ms = []
        self.boxes = []

    def process_info(self):
        """
        储存上一次人物头像出现的开始结束时间，并且做一下存储下次存储的准备
        """
        time_interval = [self.start_time, self.end_time]
        time_interval_ms = [self.start_time_ms, self.end_time_ms]
        self.boxes.append(self.box)
        self.time_show.append(time_interval)
        self.time_show_ms.append(time_interval_ms)
        self.start_time = ''
        self.end_time = ''
        self.start_time_ms = ''
        self.end_time_ms = ''

    def print_info(self):
        print(str(self.id) + ':', end='')
        print(self.time_show_ms)
        print(self.time_show)
        # print(self.boxes)

class Video_Process(object):
    """
    视频处理类
    """
    def __init__(self, filename):
        """
        视频一些基础信息初始化
        """
        self.videos_src_path = filename
        self.time_interval = 25             # 25帧一秒，一秒提取一次，个人感觉不用每帧都提取，因为不是每帧都有人而且，一些帧人物重复

    def frame_time(self, time_ms):
        """
        输入所播放帧的ms时间，返回hh:mm:ss,ms格式的时间
        :param time_ms: 当前帧ms数
        :return: 指定格式的时间
        """
        if time_ms < 1000:
            return "00:00:00," + str(time_ms)

        if time_ms >= 1000 and time_ms < 60000:
            s = int(time_ms / 1000)
            str_s = str(s) if s >= 10 else '0'+str(s)
            str_ms = str(time_ms-1000*s)
            return "00:00:" + str_s + ',' + str_ms

        if time_ms >= 60000 and time_ms < 3600000:
            m = int(time_ms / 60000)
            str_m = str(m) if m >= 10 else '0'+ str(m)
            s = int((time_ms - 60000*m) / 1000)
            str_s = str(s) if s >= 10 else '0' + str(s)
            str_ms = str(time_ms - 60000*m - 1000*s)
            return "00:" + str_m +':'+ str_s + ',' + str_ms

    def video2frame(self):
        """
        将视频按固定间隔读取写入图片
        """
        global current_time
        global current_time_ms
        cap = cv2.VideoCapture(self.videos_src_path)
        ir = Image_Recognition()        # 调取图片处理类中人脸检测方法
        frame_index = 0
        frame_count = 0
        success = True
        if not cap.isOpened():                  # 判断是否读取成功
            success = False
        success, frame = cap.read()         # success：是否读取到帧 frame：截取到一帧图像，三维矩阵表示
        while (success):
            current_time = self.frame_time(int(cap.get(0)))     # 获取当前播放帧在视频中的时间
            current_time_ms = int(cap.get(0))                   # 获取当前帧的ms时间
            if frame_index % self.time_interval == 0:     # 隔几帧取一次，加快执行速度
                ir.face_detection(frame)
                k = cv2.waitKey(1)
            frame_index += 1
            success, frame = cap.read()
        cap.release()                           # Release everything if job is finished

class Image_Recognition(object):
    """
    人脸检测类
    """
    def __init__(self):
        self.fp = Faces_Process()

    def face_detection(self, frame):
        """
        dlib方法检测一帧图片上的所出现的人脸
        :param frame:
        :return:
        """
        faces = detector(frame,1)              # 使用detector检测器来检测图像中的人脸 ,1 表示将图片放大一倍
        if len(faces) != 0:                     # 若检测到人脸，一张脸一个信息对象，并加到all_output_information[id]
            for i, d in enumerate(faces):
                self.fp.save_face(frame, d)
                cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), color=(0, 0, 255), thickness=2)  # 框出人脸
        for key in all_output_information.keys():
            if all_output_information[key].end_time != current_time and not all_output_information[key].show_again :     # 如果记录的人物出现结束时间没有更新，说明这一帧他已经不在图片上，更改flag
                all_output_information[key].process_info()
                # all_output_information[key].print_info()
                all_output_information[key].show_again = True
        # cv2.imshow('image',frame)

class Faces_Process(object):
    """
    人脸处理类
    """
    def save_face(self, frame, face):
        """
        保存所检测到的人脸,一次保存一个人的脸
        :param:视频帧检测到的所有人脸信息
        :return:
        """
        global person_count
        descriptors = []
        images = []
        # shape = shape_predictor(frame, face)                 # 检测人脸68个特征点
        person_data1 = self.get_face_features_1(frame, face)
        if not self.exist1(frame, face):                      # 判断此人是否已经检测过，若没有，保存人物头像
            person_count += 1                                 # 检测到的人脸数目作为id
            position = [face.left(), face.right(), face.top(), face.bottom()]        # 存储输出信息
            person = information(person_count, current_time, current_time, current_time_ms, current_time_ms, position, person_data1)
            all_output_information[person_count] = person                             # end_time等到检检测不到再赋值

        else:
            for key in all_output_information.keys():
                person_data2 = all_output_information[key].feature
                if self.comparePerson(person_data1, person_data2):
                    if not all_output_information[key].show_again:
                        all_output_information[key].end_time = current_time
                        all_output_information[key].end_time_ms = current_time_ms
                        return
                    else :
                        # 第二次出现,将上次的时间间隔存储
                        all_output_information[key].show_again = False
                        all_output_information[key].start_time = current_time
                        all_output_information[key].end_time = current_time
                        all_output_information[key].start_time_ms = current_time_ms
                        all_output_information[key].end_time_ms = current_time_ms
                        return

    def exist1(self, frame, face):
        """
        判断所存信息中是否有这个特征的人
        :param frame: 一帧图片的RGB值
        :param face:人脸位置，矩形对角坐标表示
        :return: 若已存在此人脸特征，返回True，否则False
        """
        person_data1 = self.get_face_features_1(frame, face)
        for key in all_output_information.keys():
            person_data2 = all_output_information[key].feature
            if self.comparePerson(person_data1, person_data2):
                return True
        return False

    def get_face_features_1(self, frame, face):
        """
        获取人物头像128维特征信息作为输出
        :param frame: 一帧图片的RGB值
        :param detector: 人脸位置，矩形对角坐标表示
        :return: 人脸的128维特征向量
        """
        shape = shape_predictor(frame, face)
        face_feature = list(face_rec_model.compute_face_descriptor(frame, shape, 10))
        return face_feature

    def comparePerson(self, data1, data2):
        """
        计算欧式距离,用于比较2张头像是否为同一人
        :param data1:人脸1*128维向量
        :param data2:人脸1*128维向量
        :return:若是同一人，返回True，否则False
        """
        data1 = np.array(data1).reshape(128, 1)
        data2 = np.array(data2).reshape(128, 1)
        difference = np.linalg.norm(data1-data2)    # 二范数
        if (difference < 0.4):
            return True
        else:
            return False

def clear_invalid_data():
    """
    删除检测到的无效人脸信息，通过记录时间来判断
    主要清除start_time=end_time和间隔时间小于1s的时间戳
    """
    for key in list(all_output_information.keys()):
        time_show = all_output_information[key].time_show
        time_show_ms = all_output_information[key].time_show_ms
        boxes = all_output_information[key].boxes
        for i in reversed(range(len(time_show))):
            if time_show_ms[i][0] == time_show_ms[i][1] or time_show_ms[i][1] - time_show_ms[i][0] < 1000:
                time_show.pop(i)
                boxes.pop(i)
        if len(time_show) == 0 :
            del all_output_information[key]
            continue

def extractFaceFeaturefromVideo(videoname, featurefilename):
    """
    从视频中提取人脸特征，时间，位置信息，结果保存在json文件中
    :param videoname: 视频路径文件名
    :param featurefilename: 结果信息保存位置
    """
    # person_count = 0  # 记录视频检测出的总人数
    global all_output_information
    all_output_information = {}  # 记录所有人物的输出信息
    # current_time = ''  # 记录开始结束时间的中介
    init_global_variable()
    vp = Video_Process(videoname)
    vp.video2frame()
    clear_invalid_data()
    output = []
    for key in all_output_information.keys():
        temp = {}
        temp['time'] = all_output_information[key].time_show
        temp['box'] = all_output_information[key].boxes
        temp['feature'] = all_output_information[key].feature
        output.append(temp)
    with open(featurefilename, 'w') as f:
        json.dump(output, f, indent=4)


def extractFaceFeaturefromImage(imagename):
    """
    输入一张图片路径，返回此图片中人脸特征
    :param imagename: 图片路径+名称
    :return: 若检测到图片中人脸，返回128维人脸特征，否则返回None
    """
    #pdb.set_trace()
    img_bgr = cv2.imread(imagename)
    #img_bgr = cv2.imread('./save_img/韩正.jpg')
    b, g, r = cv2.split(img_bgr)
    img_rgb = cv2.merge([r, g, b])
    face = detector(img_rgb, 1)
    if len(face):
        for index, face in enumerate(face):
            shape = shape_predictor(img_rgb, face)
            face_feature = face_rec_model.compute_face_descriptor(img_rgb, shape)
            return face_feature
    else:
        print("未检出图片中人脸")
        return None

def extractFaceFeaturefromImage2(Face):
    """
    输入一张保存图片信息的数组（由cv2.imread()提取而来）
    :param imagename: 图片路径+名称
    :return: 若检测到图片中人脸，返回128维人脸特征，否则返回None
    """
    #pdb.set_trace()
    b, g, r = cv2.split(Face)
    img_rgb = cv2.merge([r, g, b])
    face = detector(img_rgb, 1)
    if len(face):
        for index, face in enumerate(face):
            shape = shape_predictor(img_rgb, face)
            face_feature = face_rec_model.compute_face_descriptor(img_rgb, shape)
            return face_feature
    else:
        print("未检出图片中人脸")
        return None

if __name__ == '__main__':
    extractFaceFeaturefromVideo('/home/pengfan/616code/projects_module/share/video/20180102.mp4', \
                                '/home/pengfan/project/projects_module/task4/feature_json/test/20180102.json')
    # extractFaceFeaturefromVideo('/home/wangdepeng/project/projects_module/task4/video/zanmen-01.mp4','/home/wangdepeng/project/projects_module/task4/json/zanmen.json')
    # pdb.set_trace()
    # face_feature = extractFaceFeaturefromImage("/home/wangdepeng/xiangmuzu/project/projects_module/task4/save_img/face.jpg")
    # print(face_feature)