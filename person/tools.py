# -*- coding:utf-8 -*-
# -----------------------------------------------------
# Train Spatial Invariant Person Search Network
#
# Author: Xiankun Pei
# Creating Date: Mar 9, 2018
# Latest rectified: Mar 9, 2018
# -----------------------------------------------------
import numpy as np
import json
import os
import re
import yaml
import cv2

from tqdm import tqdm
from subprocess import call
from collections import OrderedDict


def clearShortTimeSegments(persondict):
	# ipdb.set_trace()
	newPersonDict = OrderedDict()
	count = 0
	for key in persondict.keys():
		tempTime = []
		timeList = persondict[key]['person_time']
		for timeSlice in timeList:
			if timeSlice[0] != timeSlice[1]:
				tempTime.append(timeSlice)
		# currentPersonId = int(re.findall("\d+",key)[0])
		if len(tempTime)==0:
			continue
		else:
			count +=1
			persondict[key]['person_time'] = tempTime
			newPersonDict['person{}'.format(count)] = persondict[key]
	return newPersonDict

# def video2frame(self):
# 	"""
# 	将视频按固定间隔读取写入图片
# 	"""
# 	# global current_time
# 	cap = cv2.VideoCapture(self.videos_src_path)


# 	# ir = Image_Recognition()        # 调取图片处理类中人脸检测方法
# 	frame_index = 0
# 	# frame_count = 0
# 	success = True
# 	if not cap.isOpened():                  # 判断是否读取成功
# 		success = False
# 	success, frame = cap.read()         # success：是否读取到帧 frame：截取到一帧图像，三维矩阵表示
# 	while (success):
# 		current_time = self.frame_time(int(cap.get(0)))     # 获取当前播放帧在视频中的时间
# 		if frame_index % self.time_interval == 0:     # 隔几帧取一次，加快执行速度
# 			ir.face_detection(frame)
# 			k = cv2.waitKey(1)
# 		frame_index += 1
# 	success, frame = cap.read()
# 	cap.release()                           # Release everything if job is finished
# 	return frame



# def pre_process_image(im_path, flipped=0, copy=False):
def pre_process_frame(frame, flipped=0, copy=False):
	"""Pre-process the image"""
	with open('config.yml', 'r') as f:
		config = yaml.load(f)
	target_size = config['target_size']
	max_size = config['max_size']
	pixel_means = np.array([[config['pixel_means']]])
	# ipdb.set_trace()
	im = frame
	orig_shape = im.shape
	if flipped == 1:
		im = im[:, ::-1, :]
	im = im.astype(np.float32, copy=copy)
	im -= pixel_means
	im_shape = im.shape
	im_size_min = np.min(im_shape[0:2])
	im_size_max = np.max(im_shape[0:2])
	im_scale = float(target_size) / float(im_size_min)
	# Prevent the biggest axis from being more than MAX_SIZE
	if np.round(im_scale * im_size_max) > max_size:
		im_scale = float(max_size) / float(im_size_max)
	im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, 
		interpolation=cv2.INTER_LINEAR)
	im = im[np.newaxis, :]  # add batch dimension

	return im, im_scale, orig_shape


def person_comparation(person1,person2):
	# ipdb.set_trace()
	person1 = np.array(person1).reshape(256, 1)
	person2 = np.array(person2).reshape(256, 1)
	difference = np.linalg.norm(person1-person2)    # 二范数
	return difference

def new_person_comparation(person1,person2):
	# ipdb.set_trace()
	person1 = np.array(person1).reshape(256, 1)
	person2 = np.array(person2).reshape(256, 1)
	difference = np.linalg.norm(person1-person2)    # 二范数
	return difference


def cvframe_time(time_ms):
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


def frame_time(im_name):
	"""
	输入所播放帧的ms时间，返回hh:mm:ss,ms格式的时间
	:param time_ms:
	:return:
	"""
	# ipdb.set_trace()
	frame_sequence = int(im_name[5:10])
	if frame_sequence < 60:
	    return "00:00:" + str(frame_sequence)

	if frame_sequence >= 60 and frame_sequence < 3600:
		minutes = int(frame_sequence / 60)
		str_minutes = str(minutes) if minutes >= 10 else '0'+str(minutes)
		str_s = str(frame_sequence-60*minutes)
	return "00:" + str_minutes + ':' + str_s

def differenceEveryperson(persondict):
	count = 0
	for key in persondict.keys():
		
		count +=1
		if count >=2:
			difference = new_person_comparation(persondict[key]['person_feature'],persondict[lastKey]['person_feature'])
			# print 'The difference between adjacent person :', difference
			print('The difference between adjacent person :',difference)
		lastKey = key

class User(object):
	def __init__(self,name):
		self.name = name

class UserEncoder(json.JSONEncoder):
	def default(self,obj):
		if isinstance(obj,float):
			return float(obj)

		return json.JSONEncoder.default(self,obj)

def jsonwriter(persondict,savedPath):
	items = persondict.items()
	# ipdb.set_trace()
	items.sort()
	# # for i,value in items:
	# # 	print 'difference: ',i,'time: ', value
	# for i,value in items:
	# 	persondict[i]['person_box'] = persondict[i]['person_box'].tolist()
	# 	persondict[i]['person_score'] = persondict[i]['person_score'].tolist()
	# 	persondict[i]['person_feature'] = persondict[i]['person_feature'].tolist()

	for key in persondict.keys():
		persondict[key]['person_box'] = persondict[key]['person_box'].tolist()
		persondict[key]['person_score'] = persondict[key]['person_score'].tolist()
		persondict[key]['person_feature'] = persondict[key]['person_feature'].tolist()
		# persondict[key]['person_time'] = persondict[key]['person_time'].tolist()

		# persondict[key]={'person_box':persondict[key]['person_box'].tolist(),'person_score':persondict[key]['person_score'].tolist(), 
  #                       'person_feature':persondict[key]['person_feature'].tolist(),'person_time':persondict[key]['person_time'].tolist()}
	test_dict=[
	# 'version':'1.0',
	persondict
	# 'explain':{
	# 'used':True,
	# 'details':'this is person feature',
		# }
	]
	json_str = json.dumps(test_dict,cls=UserEncoder, indent=4)
	with open(os.path.join(savedPath, 'person_data.json'), 'w') as json_file:
	# with open('person_data.json','w') as json_file:
		json_file.write(json_str)

def extract_frames(src_path,target_path,fps):
	# ipdb.set_trace()
	new_path = target_path
	for video_name in tqdm(os.listdir(src_path)):
		# ipdb.set_trace()
		filename = src_path + '/' + video_name
		cur_new_path = new_path + '/' + video_name.split('.')[0]
		if not os.path.exists(cur_new_path):
			os.mkdir(cur_new_path)
		dest = cur_new_path + '/' +video_name.split('.')[0] +'_%05d.jpg'
		call(["ffmpeg","-i", filename,"-r","{}".format(fps),dest])# fps = 5


