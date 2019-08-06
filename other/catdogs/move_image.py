import os
import shutil
import sys
train_data_file = os.listdir('../../Original_data/dogs_vs_cats/train')
print("train文件的个数:",len(train_data_file))
test_data_file = os.listdir('../../Original_data/dogs_vs_cats/test')
print("test文件个数:",len(test_data_file))
'''
移动两个文件夹分别是test1 和 test2 的文件移到test文件中
'''
test1_file = os.listdir('./test1')
print(len(test1_file))
print(test1_file[1])
print("当前目录:",os.getcwd())
print("当前目录:",os.path.abspath(os.path.dirname(__file__)))
test2_file = os.listdir('./test2')
for i in test1_file:
    test1_path = './test1/' + i
    target_folder = os.getcwd() + '/test/' + i 
    shutil.copyfile(test1_path, target_folder)
for i in test2_file:
    test2_path = './test2/' + i
    target_folder = os.getcwd() + '/test/' + i
    shutil.copyfile(test2_path, target_folder)
# shutil.copyfile('./test1/1.txt', '/home/dutlzn/Deep-Learning-with-PyTorch/catdogs/test/1.txt')
# shutil.copyfile('./test1/1.txt', '/home/dutlzn/Deep-Learning-with-PyTorch/catdogs/test/1.txt')
'''
利用python创建目录
'''
def mkdir(path):
    path = path.strip() #去除首位空格
    path = path.rstrip('/')#去除尾部的/
    #判断路径存不存在
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path,"目录创建成功")
        return True
    else:
        print(path,"目录已经存在")
        return False
mkdir(' ./testtt/')


'''
合并图片
'''
train_data_file = os.listdir('../../Original_data/dogs_vs_cats/train')
test_data_file = os.listdir('../../Original_data/dogs_vs_cats/test')
mkdir('/home/dutlzn/data/catdogs/datasets')
for i in train_data_file:
    image_path = '../../Original_data/dogs_vs_cats/train/' + i
    target_folder = '/home/dutlzn/data/catdogs/datasets/' + i
    shutil.copyfile(image_path, target_folder)
for i in test_data_file:
    image_path = '../../Original_data/dogs_vs_cats/test/' + i
    target_folder = '/home/dutlzn/data/catdogs/datasets/' + i
    shutil.copyfile(image_path, target_folder)


datasets_file = os.listdir('/home/dutlzn/data/catdogs/datasets')
print(len(datasets_file))
print(len(train_data_file)+len(test_data_file))
#查看最后合并之后图片的数量是不是原来两个文件夹的图片数量加起来是一样的 如果是一样的就说明


# shutil.rmtree('/home/dutlzn/data/catdogs/datasets') #删除文件夹 