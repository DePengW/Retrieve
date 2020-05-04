# Retrieve
项目包含：语音转文字；人脸识别；行人识别；检索四个模块
项目内容介绍：[参考](https://blog.csdn.net/weixin_42028608/article/details/104376282)

## 环境配置
参考：requirement.sh

## 代码结构
```
├── face       # 人脸识别模块
├── person     # 行人识别模块
├── search     # 搜索模块
└── word       # 语音转文字模块 
```

## 使用
1. 给定视频，使用face、person、word模块写好的api去提取视频的特定属性的特征
2. 使用search模块的process_feats.py去处理特征，并构建数据库
3. 使用search模块中的serve.py、client.py实现前后端交互
