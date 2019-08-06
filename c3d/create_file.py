#生成 处理后文件夹路径的python文件
import os
import shutil 
from mypath import Path
# 如果处理数据的方式发生变化 需要先删除文件夹中的东西
def rmrf(path):
    for i in os.listdir(path):
        new_path = path+'/'+str(i)
        if os.path.exists(new_path):
            pass
        else:
            shutil.rmtree(new_path)
    print("rmrf处理完毕")
    # if os.path.exists(self):
    #     pass 
    # else:
    #     shutil.rmtree(self)
#再生成文件夹
def mkdir(path, split):
    for i in split:
        new_path = path+'/'+i 
        os.mkdir(new_path)
    print("mkdir处理完毕")



if __name__ == '__main__':
    '''
    root_dir = '/share2/lzn_skating'
    output_dir = '/share2/lzn_skating_images'
    '''
    dataset = 'skating'
    root_dir, output_dir = Path.db_dir(dataset)
    #前者是原始数据文件夹 后者是生成数据存放文件夹
    split = ['train', 'test', 'val']
    '''
    生成文件夹
    '''
    rmrf(output_dir)
    mkdir(output_dir, split)


