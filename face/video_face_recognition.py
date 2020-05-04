# -*- coding:utf-8 -*-
import face_recognition
import numpy as np
import time
import os
import dlib     # pip install dlib
import cv2      # pip install opencv-contrib-python

detector = dlib.get_frontal_face_detector()     # 模型加载
currentpath = os.path.dirname(os.path.abspath(__file__))
face_rec_model = dlib.face_recognition_model_v1(currentpath + '/models/dlib_face_recognition_resnet_model_v1.dat')
shape_predictor = dlib.shape_predictor(currentpath + '/models/shape_predictor_68_face_landmarks.dat')
cascade = cv2.CascadeClassifier(currentpath + "/models/haarcascade_frontalface_alt.xml")


def extract_face_feature(imagename):
    """
    输入一张图片路径，返回此图片中人脸特征
    :param imagename: 图片路径+名称
    :return: 若检测到图片中人脸，返回128维人脸特征，否则返回None
    """
    img_bgr = cv2.imread(imagename)
    b, g, r = cv2.split(img_bgr)
    img_rgb = cv2.merge([r, g, b])
    face = detector(img_rgb, 1)
    if len(face):
        for index, face in enumerate(face):
            shape = shape_predictor(img_rgb, face)
            face_feature = face_rec_model.compute_face_descriptor(img_rgb, shape)
            return face_feature
    else:
        return None


def get_face_features(frame, faceposition):
    """
    获取视频中某一帧人脸128维特征向量作为输出
    :param frame: 一帧图片的RGB值
    :param detector: 人脸位置，矩形对角坐标表示
    :return: 人脸的128维特征向量
    """
    shape = shape_predictor(frame, faceposition)
    face_feature = face_rec_model.compute_face_descriptor(frame, shape)
    return face_feature


def comparePerson(feature1, feature2):
    """
    计算欧式距离,用于比较2张头像是否为同一人
    :param feature1:人脸1*128维向量
    :param feature2:人脸1*128维向量
    :return:若是同一人，返回True，否则False
    """
    data1 = np.array(feature1).reshape(128, 1)
    data2 = np.array(feature2).reshape(128, 1)
    difference = np.linalg.norm(data1 - data2)  # 二范数
    if difference < 0.54:  # 阈值设置为0.6
        return True
    else:
        return False


def total_time(func):
    """
    Calculate the running time of the function
    :param func: the function need to been calculated
    :return:
    """
    def call_fun(*args, **kwargs):
        start_time = time.time()
        f = func(*args, **kwargs)
        end_time = time.time()
        print('%s() run time：%s s' % (func.__name__, int(end_time - start_time)))
        return f
    return call_fun


def frame_time(time_ms):
    """
    输入所播放帧的ms时间，返回hh:mm:ss,ms格式的时间
    :param time_ms: 当前帧ms数
    :return: 指定格式的时间
    """
    if time_ms < 1000:
        return "00:00:00," + str(time_ms)

    elif time_ms < 60000:
        s = int(time_ms / 1000)
        str_s = str(s) if s >= 10 else '0'+str(s)
        str_ms = str(time_ms-1000*s)
        return "00:00:" + str_s + ',' + str_ms

    elif time_ms < 3600000:
        m = int(time_ms / 60000)
        str_m = str(m) if m >= 10 else '0'+ str(m)
        s = int((time_ms - 60000*m) / 1000)
        str_s = str(s) if s >= 10 else '0' + str(s)
        str_ms = str(time_ms - 60000*m - 1000*s)
        return "00:" + str_m + ':' + str_s + ',' + str_ms


class Information(object):
    """  用于记录时间戳 """
    def __init__(self):
        self.start_time = ''
        self.end_time = ''
        self.show_flag = False
        self.time_show = []

    def process_info(self):
        """  储存上一次人物头像出现的开始结束时间，并且做一下次存储的准备  """
        time_interval = [self.start_time, self.end_time]
        self.time_show.append(time_interval)
        self.start_time = ''
        self.end_time = ''
        self.show_flag = False

    def clear_invalid_data(self):
        """
        删除检测到的无效信息，通过记录时间来判断,主要清除start_time=end_time的时间戳
        """
        for time_intervel in self.time_show[::-1]:
            if time_intervel[0] == time_intervel[1]:
                self.time_show.remove(time_intervel)
        return self.time_show


class Facedetection(object):
    def __init__(self, imagename):
        self.current_time = 0
        self.frame_interval = 1
        self.feature = extract_face_feature(imagename)
        self.timestamp = Information()

    def face_detected(self, frame):
        """
        检测一帧图片上的所出现的人脸, 并用矩形框框出,同时记录下此人脸出现的时间
        :param frame:视频中的一帧
        :return:
        """
        faces = detector(frame, 1)                  # 使用detector检测器来检测图像中的人脸 ,1 表示将图片放大一倍
        if len(faces) != 0:                         # 若检测到人脸，将人脸框出来
            for i, d in enumerate(faces):
                feature1 = get_face_features(frame, d)
                if comparePerson(self.feature, feature1):
                    if not self.timestamp.show_flag:            # 说明此人脸是第一次出现
                        self.timestamp.start_time = self.current_time
                        self.timestamp.end_time = self.current_time
                        self.timestamp.show_flag = True         # 下次出现只更新end_time
                    else:
                        self.timestamp.end_time = self.current_time
        # 说明这帧已经不存在这个人脸了，保存时间戳到time_intervel
        if self.timestamp.show_flag and self.timestamp.end_time != self.current_time:
            self.timestamp.process_info()

    @total_time
    def start(self, vidoepath):
        """ 将视频按固定间隔读取检测图片 """
        self.current_time = 0
        frame_index = 0
        cap = cv2.VideoCapture(vidoepath)
        count = 5
        while count and not cap.isOpened():
            cap = cv2.VideoCapture(vidoepath)
            count -= 1  # 读五次
            if not count and not cap.isOpened():
                print('Read error !')
                return
        success, frame = cap.read()  # success：是否读取到帧 frame：截取到一帧图像，三维矩阵表示
        while success:
            self.current_time = frame_time(int(cap.get(0)))  # 获取当前播放帧在视频中的时间
            if frame_index % self.frame_interval == 0:  # 隔几帧取一次，加快执行速度
                self.face_detected(frame)
            frame_index += 1
            success, frame = cap.read()
        cap.release()  # Release everything if job is finished
        return self.timestamp.clear_invalid_data()


class Facerecognition(object):
    def __init__(self, imagename):
        self.current_time = 0
        self.frame_interval = 5
        self.feature_encodeing = face_recognition.face_encodings(face_recognition.load_image_file(imagename))[0]
        self.known_faces = [self.feature_encodeing]
        self.timestamp = Information()

    def face_detected(self, frame):
        """
         检测一帧图片上的所出现的人脸, 并用矩形框框出,同时记录下此人脸出现的时间
         :param frame:视频中的一帧
         :return:
         """
        rgb_frame = frame[:, :, ::-1]
        frame_face_locations = face_recognition.face_locations(rgb_frame)
        frame_face_encodings = face_recognition.face_encodings(rgb_frame, frame_face_locations)
        for frame_face_encoding in frame_face_encodings:
            match = face_recognition.compare_faces(self.known_faces, frame_face_encoding, tolerance=0.50)
            if match[0]:
                if not self.timestamp.show_flag:  # 说明此人脸是第一次出现
                    self.timestamp.start_time = self.current_time
                    self.timestamp.end_time = self.current_time
                    self.timestamp.show_flag = True  # 下次出现只更新end_time
                else:
                    self.timestamp.end_time = self.current_time
        if self.timestamp.show_flag and self.timestamp.end_time != self.current_time:
            self.timestamp.process_info()

    @total_time
    def start(self, vidoepath):
        self.current_time = 0
        frame_index = 0
        cap = cv2.VideoCapture(vidoepath)
        count = 5
        while count and not cap.isOpened():
            cap = cv2.VideoCapture(vidoepath)
            count -= 1  # 读五次
            if not count and not cap.isOpened():
                print('Read error !')
                return
        success, frame = cap.read()
        while success:
            self.current_time = frame_time(int(cap.get(0)))  # 获取当前播放帧在视频中的时间
            if frame_index % self.frame_interval == 0:  # 隔几帧取一次，加快执行速度
                self.face_detected(frame)
            frame_index += 1
            success, frame = cap.read()
        cap.release()  # Release everything if job is finished
        return self.timestamp.clear_invalid_data()


if __name__ == '__main__':
    fd = Facedetection(imagename='./images/face.jpg')
    result = fd.start('./test_video.mp4')
    print(result)
    # fd = Facerecognition(imagename='./images/test.jpg')
    #     # result = fd.start('./videos/CCTV_News11.ts')
    #     # print(result)





