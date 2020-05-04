# -----------------------------------------------------
# Spatial Invariant Person Search Network
#
# Author: Liangqi Li and Xinlei Chen
# Creating Date: Apr 1, 2018
# Latest rectified: Nov 5, 2018
# -----------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as func
import yaml

from .vgg16 import Vgg16
from .resnet import MyResNet
from .densenet import DenseNet
from .strpn import STRPN
from utils.losses import smooth_l1_loss, TripletLoss


class SIPN(nn.Module):
    # def __init__(self, block, layers, num_classes=1000):
    #     self.inplanes = 64
    #     super(ResNet, self).__init__()
    def __init__(self, net_name, dataset_name, pre_model=''):
        super(SIPN, self).__init__()
        # super(SIPN,self).__init__()
        self.net_name = net_name
        self.dataset_name = dataset_name
        self.pre_model = pre_model

        if dataset_name == 'sysu':
            self.num_pid = 5532
            self.queue_size = 5000
        elif dataset_name == 'prw':
            self.num_pid = 483
            self.queue_size = 500
        else:
            raise KeyError(dataset_name)
        self.lut_momentum = 0.5
        self.reid_feat_dim = 256

        self.register_buffer('lut', torch.zeros(
            self.num_pid, self.reid_feat_dim).cuda())
        self.register_buffer('queue', torch.zeros(
            self.queue_size, self.reid_feat_dim).cuda())

        if self.net_name == 'vgg16':
            self.net = Vgg16(pre_model)
        elif self.net_name == 'res34':
            self.net = MyResNet(34, pre_model)
        elif self.net_name == 'res50':
            self.net = MyResNet(50, pre_model)
        elif self.net_name == 'dense121':
            self.net = DenseNet(121, pre_model)
        elif self.net_name == 'dense161':
            self.net = DenseNet(161, pre_model)
        else:
            raise KeyError(self.net_name)

        self.fc7_channels = self.net.fc7_channels

        # SPIN consists of three main parts
        self.head = self.net.head
        self.strpn = STRPN(self.net.net_conv_channels, self.num_pid)
        self.tail = self.net.tail

        self.cls_score_net = nn.Linear(self.fc7_channels, 2)
        self.bbox_pred_net = nn.Linear(self.fc7_channels, 8)
        self.reid_feat_net = nn.Linear(self.fc7_channels, self.reid_feat_dim)
        self.init_linear_weight(False)

    def forward(self, im_data, gt_boxes, im_info, mode='gallery'):
        if self.training:
            # ipdb.set_trace()
            if mode == 'query':
                assert gt_boxes.size(0) == 1
                # im_h = im_data.size(2)
                # im_w = im_data.size(3)
                # x1, y1, x2, y2 = gt_boxes[0]
                # w = x2 - x1
                # h = y2 - y1
                #
                # sw = w / im_w
                # sh = h / im_h
                # tx = (x1 + x2) / im_w - 1
                # ty = (y1 + y2) / im_h - 1
                #
                # theta = torch.Tensor([[sw, 0, tx, 0, sh, ty]]).view(-1, 2, 3)
                # im_crop = func.grid_sample(im_data, func.affine_grid(
                #     theta, torch.Size([1, 3, h, w])))

                net_conv = self.head(im_data)
                # # TODO: move pooling layer from strpn to SIPN
                # pooled_feat = self.strpn(net_conv, gt_boxes, im_info, mode)
                if self.net_name == 'vgg16':
                    fc7 = self.tail(net_conv)
                else:
                    fc7 = self.tail(net_conv).mean(3).mean(2)
                reid_feat = self.reid_feat_net(fc7)
                return reid_feat

            net_conv = self.head(im_data)
            # Returned parameters contain 3 tuples here
            pooled_feat, trans_feat, rpn_loss, label, bbox_info = self.strpn(
                net_conv, gt_boxes, im_info)
            if self.net_name == 'vgg16':
                pooled_feat = pooled_feat.view(pooled_feat.size(0), -1)
                fc7 = self.tail(pooled_feat)
            else:
                fc7 = self.tail(pooled_feat).mean(3).mean(2)
            cls_score = self.cls_score_net(fc7)
            bbox_pred = self.bbox_pred_net(fc7)

            # reid_fc7 = self.tail(trans_feat).mean(3).mean(2)
            # reid_feat = F.normalize(self.reid_feat_net(reid_fc7))
            reid_feat = self.reid_feat_net(fc7)

            det_label, pid_label = label
            det_label = det_label.view(-1)
            cls_loss = func.cross_entropy(cls_score.view(-1, 2), det_label)
            bbox_loss = smooth_l1_loss(bbox_pred, bbox_info)
            rpn_cls_loss, rpn_box_loss = rpn_loss
            det_loss = rpn_cls_loss, rpn_box_loss, cls_loss, bbox_loss

            return det_loss, reid_feat, pid_label ###the true module is not used in test ,the training  is not sure.

        else:
            if mode == 'gallery':
                # ipdb.set_trace()
                net_conv = self.head(im_data)
                # ipdb.set_trace()
                rois, pooled_feat, trans_feat = self.strpn(
                    net_conv, gt_boxes, im_info)
                if self.net_name == 'vgg16':
                    pooled_feat = pooled_feat.view(pooled_feat.size(0), -1)
                    fc7 = self.tail(pooled_feat)
                else:
                    fc7 = self.tail(pooled_feat).mean(3).mean(2) ##tail(pooled_feat) pooled_feat [300,1024,7,7] -->[300,2048,7,7]
                                                                 ## .mean(3).mean(2) -->[300,2048]
                cls_score = self.cls_score_net(fc7)              ##self.cls_score_net = nn.Linear(self.fc7_channels, 2)
                bbox_pred = self.bbox_pred_net(fc7)              ##self.bbox_pred_net = nn.Linear(self.fc7_channels, 8)

                # reid_fc7 = self.tail(trans_feat).mean(3).mean(2)
                # reid_feat = F.normalize(self.reid_feat_net(reid_fc7))
                reid_feat = func.normalize(self.reid_feat_net(fc7))###self.reid_feat_net = nn.Linear(self.fc7_channels, self.reid_feat_dim) 
                                                                   ##[300,2048]-->[300,256] normalize zhengzehua
                cls_prob = func.softmax(cls_score, 1) ## 1 maybe the pattern
                # ipdb.set_trace()

                with open('config.yml', 'r') as f:
                    config = yaml.load(f)
                mean = config['train_bbox_normalize_means']
                std = config['train_bbox_normalize_stds']
                means = bbox_pred.new(mean).repeat(2).unsqueeze(0).expand_as(
                    bbox_pred)
                stds = bbox_pred.new(std).repeat(2).unsqueeze(0).expand_as(
                    bbox_pred)
                bbox_pred = bbox_pred.mul(stds).add(means) ## * std + mean ##(chengyi fangcha + junzhi) 

                # ipdb.set_trace()
                # cls_prob = cls_prob.cpu()
                # bbox_pred = bbox_pred.cpu()
                # rois = rois.cpu()
                # reid_feat = reid_feat.cpu()

                # cls_prob = cls_prob.detach().numpy()
                # bbox_pred = bbox_pred.detach().numpy()
                # rois = rois.detach().numpy()
                # reid_feat = reid_feat.detach().numpy()


                cls_prob = cls_prob.cpu().data.numpy()
                bbox_pred = bbox_pred.cpu().data.numpy()
                rois = rois.cpu().data.numpy()
                reid_feat = reid_feat.cpu().data.numpy()   ##tensor on the gpu to transformed numpy .cpu().data.numpy(),  .cpu().detach().numpy

                return cls_prob, bbox_pred, rois, reid_feat

            elif mode == 'query':
                net_conv = self.head(im_data)
                # TODO: move pooling layer from strpn to SIPN
                pooled_feat = self.strpn(net_conv, gt_boxes, im_info, mode)
                if self.net_name == 'vgg16':
                    pooled_feat = pooled_feat.view(pooled_feat.size(0), -1)
                    fc7 = self.tail(pooled_feat)
                else:
                    fc7 = self.tail(pooled_feat).mean(3).mean(2)
                reid_feat = func.normalize(self.reid_feat_net(fc7))

                return reid_feat.data.cpu().numpy()

            else:
                raise KeyError(mode)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        self.net.train(mode)

    def init_linear_weight(self, trun):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initializer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

        normal_init(self.cls_score_net, 0, 0.01, trun)
        normal_init(self.bbox_pred_net, 0, 0.001, trun)
        # TODO: change 0.01 for reid_feat_net
        normal_init(self.reid_feat_net, 0, 0.01, trun)
    
    def load_trained_model(self, state_dict):
        # import pdb
        # pdb.set_trace()
        nn.Module.load_state_dict(
            self, {k: state_dict[k] for k in list(self.state_dict()) if k[-len('batches_tracked'):]!='batches_tracked'})

    ##解决pytorch 1.1问题 但是在这里就会造成 结果全部是1，没有任何意义，
    # def load_trained_model(self, state_dict):
    #     nn.Module.load_state_dict(
    #         self, {k.replace('module.',''):v for k,v in self.state_dict().items()})