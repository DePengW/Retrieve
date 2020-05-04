# -*- coding:utf-8 -*-
"""
配置任务三环境：
opencv: pip install opencv-python==4.1.0.25
pytorch: conda install pytorch=0.4.1 cuda80 -c pytorch
yaml: pip install pyyaml
torchvision: pip install torchvision==0.2.1
tqdm: pip install tqdm
"""

import os
import torch
import glob
import cv2
import numpy as np
import yaml
from collections import OrderedDict
import pdb

from models.model import SIPN
from nms.gpu_nms import gpu_nms as nms
from tools import pre_process_frame
from dataset.dataset import pre_process_image
from dataset.dataset import pre_process_image2

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def loadNetwork():
    """
    加载网络模型用于提取图片特征
    :return: 返回测试网络模型
    """
    net = SIPN('res50', 'sysu')
    currentpath = os.path.dirname(os.path.realpath(__file__))
    trained_model_dir = currentpath + '/output/sysu/sipn_res50_20.tar'
    checkpoint = torch.load(trained_model_dir)
    net.load_trained_model(checkpoint['model_state_dict'])
    net.eval()
    return net

def bbox_transform_inv(boxes, deltas):
    # Input should be both tensor or both Variable and on the same device
    if len(boxes) == 0:
        return deltas.detach() * 0

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    pred_boxes = torch.cat(
        [_.unsqueeze(2) for _ in [pred_ctr_x - 0.5 * pred_w,
                                  pred_ctr_y - 0.5 * pred_h,
                                  pred_ctr_x + 0.5 * pred_w,
                                  pred_ctr_y + 0.5 * pred_h]], 2).view(
        len(boxes), -1)

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes

def time2num(time_str):
    h = int(time_str[:2])
    m = int(time_str[3:5])
    s = int(time_str[6:8])
    ms = int(time_str[9:])
    return int(h*3600000+m*60000+s*1000+ms)

def extractPersonFeaturefromVideo(videoname, featurefilename):
    """
    从指定路径的视频中提取行人特征，并保存在featurefilename中
    :param videoname: 视频文件的绝对路径
    :param featurefilename: 存储对应行人特征信息的json文件的绝对路径
    :return: None
    """
    net = loadNetwork()
    persondict = OrderedDict()
    persondict['person_num'] = 0
    videonamepath = '/home/wangdepeng/task3/video/{}'.format(videoname)
    cap = cv2.VideoCapture(videoname)
    with open('config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    flag = 0

    frame_index = 0
    success = True
    time_interval = 5
    thresh = 0.9
    if not cap.isOpened():
        success = False
    success, frame = cap.read()
    frame_information = {}
    thisFramePerson = []
    lastFramePerson = thisFramePerson
    while(success):
        #frame_information[frame_index] = []
        current_time = cvframe_time(int(cap.get(0)))
        if frame_index % time_interval == 0:
            use_cuda = True
            #frame = cv2.copyMakeBorder(frame,30,30,30,30,cv2.BORDER_CONSTANT,value=0)
            im, im_scale, orig_shape = pre_process_frame(frame)
            im_info = np.array([im.shape[1], im.shape[2], im_scale], dtype=np.float32)
            im = im.transpose([0, 3, 1, 2])
            if use_cuda:
                net.cuda()
                im = torch.from_numpy(im).cuda()
            else:
                im = torch.from_numpy(im)
            scores, bbox_pred, rois, features = net.forward(im, None, im_info)
            boxes = rois[:, 1:5] / im_info[2]
            scores = np.reshape(scores, [scores.shape[0], -1])
            bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
            if config['test_bbox_reg']:
                box_deltas = bbox_pred
                pred_boxes = bbox_transform_inv(
                    torch.from_numpy(boxes),
                    torch.from_numpy(box_deltas)).numpy()
                pred_boxes = clip_boxes(pred_boxes, orig_shape)
            else:
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))
            boxes = pred_boxes
            j = 1
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            features = features[inds]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack(
                (cls_boxes, cls_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            #keep = nms(torch.from_numpy(cls_dets), config['test_nms']).numpy() if cls_dets.size > 0 else []
            keep = np.array(nms(cls_dets, config['test_nms'])) if cls_dets.size > 0 else []
            cls_dets = cls_dets[keep, :]
            features = features[keep, :]
            if len(cls_dets) == 0:
                #print('There are no persons found at time of {}'.format(current_time))
                frame_index += 1
                success, frame = cap.read()
                thisFramePerson = []
                continue
            lastFramePerson = thisFramePerson
            thisFramePerson = []
            draw1 = frame
            for i in range(len(cls_dets)):
                x1, y1, x2, y2, score = cls_dets[i]
                draw1 = cv2.rectangle(draw1, (x1, y1), (x2, y2), (100, 200, 50), 2)
                person_box = cls_dets[i, :4]
                lastFrameHaveThisPerson = False#从这里开始改的
                sim = 0
                max_sim_id = -1
                for j in range(len(lastFramePerson)):
                    simij = person_comparation(lastFramePerson[j]['feature'],features[i])
                    if (simij > 0.6)and(simij > sim)and(lastFramePerson[j]['flag'] == 0):
                        sim = simij
                        max_sim_id = j
                if max_sim_id != -1:
                    #try:
                    if time2num(current_time)-time2num(persondict[lastFramePerson[max_sim_id]['id']]['time'][1])!=40*time_interval:
                        pdb.set_trace()
                    #except:
                    #    pdb.set_trace()
                    persondict[lastFramePerson[max_sim_id]['id']]['time'][1] = current_time
                    if score > persondict[lastFramePerson[max_sim_id]['id']]['score']:
                        persondict[lastFramePerson[max_sim_id]['id']]['score'] = score
                        persondict[lastFramePerson[max_sim_id]['id']]['feature'] = features[i]
                        persondict[persondict['person_num']]['frame'] = frame_index
                    thisFramePerson.append({'id': lastFramePerson[max_sim_id]['id'],
                                            'score': persondict[lastFramePerson[max_sim_id]['id']]['score'],
                                            'feature': persondict[lastFramePerson[max_sim_id]['id']]['feature'],
                                            'flag': 0}
                                           )
                    persondict[lastFramePerson[max_sim_id]['id']]['frame_num'] += 1
                    lastFramePerson[max_sim_id]['flag'] = 1
                else:
                    persondict['person_num'] += 1
                    persondict[persondict['person_num']] = {}
                    persondict[persondict['person_num']]['time'] = [current_time, current_time]
                    persondict[persondict['person_num']]['score'] = score
                    persondict[persondict['person_num']]['feature'] = features[i]
                    persondict[persondict['person_num']]['frame'] = frame_index
                    persondict[persondict['person_num']]['frame_num'] = 1
                    thisFramePerson.append({'id': persondict['person_num'], 'score': score, 'feature': features[i], 'flag': 0})
                    #pdb.set_trace()
            #一直改到这
        frame_index += 1
        success, frame = cap.read()

    persondict.pop('person_num')
    persondict = clearShortTimeSegments(persondict)
    jsonwriter(persondict, featurefilename)
    return persondict

def extractPersonFeaturefromImage(imagename):
    """
    :param imagename:行人图片名
    :return: 从该行人图片中提取到的256维特征向量
    """
    net = loadNetwork()
    img = cv2.imread(imagename)
    height, width, _ = img.shape
    q_roi = [0,0,height,width]
    x1, y1, h, w = q_roi
    q_im, q_scale, _ = pre_process_image(imagename)
    q_roi = np.array(q_roi) * q_scale
    q_info = np.array([q_im.shape[1], q_im.shape[2], q_scale], dtype=np.float32)
    q_im = q_im.transpose([0, 3, 1, 2])
    q_roi = np.hstack(([[0]], q_roi.reshape(1, 4)))

    use_cuda = True
    net.cuda()
    with torch.no_grad():
        if use_cuda:
            q_im = torch.from_numpy(q_im).cuda()
            q_roi = torch.from_numpy(q_roi).float().cuda()
        else:
            q_im = torch.from_numpy(q_im)
            q_roi = torch.from_numpy(q_roi).float()
        q_feat = net.forward(q_im, q_roi, q_info, 'query')[0]
    return q_feat

def extractPersonFeaturefromImage2(img):       # extract feature directly from numpy array(from cv2.imread())
    """
    从指定图片中提取行人特征
    :param img: 用opencv(cv2.imread())提取的图片RGB信息，为H*W*C numpy数组
    :return: 该图片的256维行人特征向量
    """
    net = loadNetwork()
    height, width, _ = img.shape
    q_roi = [0,0,height,width]
    x1, y1, h, w = q_roi
    q_im, q_scale, _ = pre_process_image2(img)
    q_roi = np.array(q_roi) * q_scale
    q_info = np.array([q_im.shape[1], q_im.shape[2], q_scale], dtype=np.float32)
    q_im = q_im.transpose([0, 3, 1, 2])
    q_roi = np.hstack(([[0]], q_roi.reshape(1, 4)))
    use_cuda = True
    net.cuda()
    with torch.no_grad():
        if use_cuda:
            q_im = torch.from_numpy(q_im).cuda()
            q_roi = torch.from_numpy(q_roi).float().cuda()
        else:
            q_im = torch.from_numpy(q_im)
            q_roi = torch.from_numpy(q_roi).float()
        q_feat = net.forward(q_im, q_roi, q_info, 'query')[0]
    return q_feat


def extractPersonFeaturefromImage_needClassifyLoc(img): # extract feature directly from numpy array(from cv2.imread())
    """
    在指定图片中定位主要人物，并提取其行人特征
    :param img: 用opencv(cv2.imread())提取的图片RGB信息，为H*W*C numpy数组
    :return: 该图片中最主要人物的256维行人特征向量
    """
    net = loadNetwork()
    with open('config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    thresh = 0.7
    use_cuda = True
    #frame = cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=0)
    im, im_scale, orig_shape = pre_process_frame(img)
    im_info = np.array([im.shape[1], im.shape[2], im_scale], dtype=np.float32)
    im = im.transpose([0, 3, 1, 2])
    if use_cuda:
        net.cuda()
        im = torch.from_numpy(im).cuda()
    else:
        im = torch.from_numpy(im)
    scores, bbox_pred, rois, features = net.forward(im, None, im_info)
    boxes = rois[:, 1:5] / im_info[2]
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    if config['test_bbox_reg']:
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(
            torch.from_numpy(boxes),
            torch.from_numpy(box_deltas)).numpy()
        pred_boxes = clip_boxes(pred_boxes, orig_shape)
    else:
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))
    boxes = pred_boxes
    j = 1
    inds = np.where(scores[:, j] > thresh)[0]
    cls_scores = scores[inds, j]
    features = features[inds]
    cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
    cls_dets = np.hstack(
        (cls_boxes, cls_scores[:, np.newaxis])).astype(
        np.float32, copy=False)


    keep = np.array(nms(cls_dets, config['test_nms'])) if cls_dets.size > 0 else []
    cls_dets = cls_dets[keep, :]
    features = features[keep, :]
    maxscore = 0
    maxscore_i = -1
    for i in range(len(cls_dets)):
        if cls_dets[i][-1] > maxscore:
            maxscore = cls_dets[i][-1]
            maxscore_i = i
    if maxscore_i != -1:
        return features[maxscore_i]
    else:
        return extractPersonFeaturefromImage2(img)#如果局部未感知到行人，则输入图片经过了裁剪，直接对整张图片做特征提取



if __name__ == '__main__':
    q_feat = extractPersonFeaturefromImage(imagename='/home/wangdepeng/project/call_module_test/images/task3_test.jpg')
    print(q_feat)
    #persondict = extractPersonFeaturefromVideo(videoname, featurefilename) #videoname:要提取特征的视频路径， featurefilename：提取特征的保存路径

