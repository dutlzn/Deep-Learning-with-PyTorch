'''
用于程序编写过程中，一些程序的测试
'''
import os 
from models import BasicModule
root = "D:\data\dogcat"
print(os.path.join(root, 'train'))
#打印结果是 D:\data\dogcat\train

x = 'dog.2.jpg'
# x = '14.jpg'
print(x.split('.'))
net = BasicModule.BasicModule()
print(net.state_dict())