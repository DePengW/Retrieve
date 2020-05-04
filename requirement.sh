#!/bin/sh

# 配置任务1,2,3，4的python环境脚本
# 使用方法
# 1. 创建python环境 conda create -n project python==3.6.8
# 2. 切换到project，然后sh requirement.sh完成配置环境脚本
# 3. 另需手动安装es，版本elasticsearch-5.5.1.zip，网上下载解压就行

# task1
pip install google-api-python-client==1.4.2
pip install requests==2.3.0
pip install pysrt==1.0.1
pip install progressbar2
pip install six==1.11.0

# task2
pip install opencv-python==4.1.0.25
pip install dlib==19.17

# task3
conda install pytorch=0.4.1 cuda100 -c pytorch
pip install torchvision==0.2.1
conda install cudatoolkit==10.0.130
pip install tqdm
pip install pyyaml

# task4
conda install faiss-gpu cudatoolkit=10.0 -c pytorch
pip install elasticsearch==7.0.4
pip install flask==1.0.2
pip install pillow==6.1.0

