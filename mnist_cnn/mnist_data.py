# TODO:下载minist数据集 并且已经分类好了
import os 
import numpy as np
import matplotlib.pyplot as plt 
from keras.datasets import mnist 
from matplotlib.image import imsave 
import itertools 

(X_train, y_train), (X_test, y_test) = mnist.load_data() 
print("X_train original shape:", X_train.shape)
print("y_train original shape:", y_train) 
print("X_test original shape:", X_test.shape)
print("y_test original shape:", y_test.shape)
print("x[0] shape:", X_train[0].shape)

#创建文件夹函数
import os
import shutil
def mkdir(path):
    if os.path.exists(path):
        pass
#         print("{} 已经被创建了".format(path))
    else:
        os.makedirs(path)
#         print("成功创建了 {}".format(path))
def rmdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
#         print("{} 已经被成功删除".format(path))
    else:
        pass
#         print("{} 该目录不存在".format(path))

# mkdir('./img/1')

train_path = './mnist/train/'
test_path = './mnist/test/'
image_counter = itertools.count(0)
for image, label in zip(X_train, y_train):
    image_name = next(image_counter)
    tree_path = train_path + str(label)
    mkdir(tree_path)
    image_path = train_path + str(label) + '/' + str(image_name) + '.png'
    imsave(image_path, image)
for image, label in zip(X_test, y_test):
    image_name = next(image_counter)
    tree_path = test_path + str(label)
    mkdir(tree_path)
    image_path = test_path + str(label) + '/' + str(image_name) + '.png'
    imsave(image_path, image)

#todo: count image 
num_image_train = 0
for i in os.listdir(train_path):
    image_path = train_path + i
    num_image_train += len(os.listdir(image_path))
print("num of train images:", num_image_train)
num_image_test = 0
for i in os.listdir(test_path):
    image_path = test_path + i
    num_image_test += len(os.listdir(image_path))
print("num of test images:", num_image_test)