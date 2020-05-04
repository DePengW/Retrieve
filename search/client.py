# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 19:27:26 2019

@author: Administrator
"""

import requests
#import matplotlib.pyplot as cv2
import cv2
import json
import numpy as np
import pdb

'''
使用实验室的http代理，验证成功即可访问服务器内网
'''
# proxies = {"http": "http://proxy@:Inooghai7Ohr1yooyo7uDip8eifuShoQuaighiel0aepu6oa1ain2quoo8gath9A@222.195.7.214:13128/",
#            "https": "https://proxy@:Inooghai7Ohr1yooyo7uDip8eifuShoQuaighiel0aepu6oa1ain2quoo8gath9A@222.195.7.214:13128/"}
'''
输入查询数据（文本、人脸、行人）
'''
face_name = '王沪宁.jpg'
person_name = '王沪宁person2.jpg'
test_data = [1,2,3,4,5]

inquire_data = {'key words':'韩正',
                'face':(cv2.imread('./save_img/'+face_name)).tolist(),
                'person':(cv2.imread('./save_img/'+person_name)).tolist()
                }

# requests发送数据给服务器端并接收返回数据
response = requests.post("http://127.0.0.1:9000/register", data=json.dumps(inquire_data))
output = eval(response.text)
print(output)









