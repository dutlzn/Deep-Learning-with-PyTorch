import os
import shutil

def redistribution():
    data_file = os.listdir('../../data/catdogs/datasets') 
    # for data in data_file:
    #     print(data)
    #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序
    dogs_file = list(filter(lambda x:x[:3] == 'dog', data_file))
    cats_file = list(filter(lambda x:x[:3] == 'cat', data_file))

    data_root = '../../data/catdogs/'
    train_root = '../../data/catdogs/train'
    val_root = '../../data/catdogs/val'
    
    for i in range(len(cats_file)):
        image_path = data_root + 'datasets/' +cats_file[i]
        if i<len(cats_file)*0.9:
            new_path = train_root + '/cat/' + cats_file[i]
        else:
            new_path = val_root + '/cat/' + cats_file[i]
        shutil.move(image_path, new_path)

    for i in range(len(dogs_file)):
        image_path = data_root + 'datasets/' + dogs_file[i]
        if i < len(dogs_file) * 0.9:
            new_path = train_root + '/dog/' + dogs_file[i]
        else:
            new_path = val_root + '/dog/' + dogs_file[i]
        shutil.move(image_path, new_path)




if __name__ == '__main__':
    redistribution()