# TDCN
This repository is an PyTorch implementation of the paper **Tree-structured Dilated Convolutional Networks for Image Compressed Sensing**
You can find the original code and more information from [here](https://github.com/UHADS/TDCN). 

The paper: [link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9889727) 

# The version of the python package we use is：
python 3.9

torch 1.10.0 + cuda 10.2

torchvision 0.11.0

scipy 1.7.1

glob、pandas and so on


# Training
We used [DIV2K](http://www.vision.ee.ethz.ch/%7Etimofter/publications/Agustsson-CVPRW-2017.pdf) dataset to train our model. Please download it from [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or 
[百度网盘](https://pan.baidu.com/s/1IdFe83rPXEjquLb_1Kqf4g) 提取码：wj46.  

Unpack the tar file to any place you want. Then, change the ```folder``` argument in ```generate_train.m``` to the place where DIV2K images are located. Use ```train.py``` for training.

# Testing
You can use the model we provide for testing. Use ```test.py``` for testing TDCN.

# Citation
If you find TDCN useful in your research, please consider citing:

R. Lu and K. Ye, "Tree-Structured Dilated Convolutional Networks for Image Compressed Sensing," in IEEE Access, vol. 10, pp. 98374-98383, 2022, doi: 10.1109/ACCESS.2022.3206016.

If you have any questions, you are welcome to contact me. My email is: 496315130@qq.com
