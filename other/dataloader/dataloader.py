import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import numpy as np

Data = np.asarray([[1,2],[3,4],[5,6],[7,8]])
Label = np.asarray([[0],[1],[0],[2]])

class SubDataset(Dataset.Dataset):
    #初始化 定义数据内容和标签
    def __init__(self, Data, Label):
        self.data = Data
        self.label = Label
        self.data.dtype = 'int64'
        self.label.dtype = 'int64'
        print(self.data.dtype)
        print(self.label.dtype)
    #返回数据集大小
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data = torch.Tensor(self.data[index])
        label = torch.IntTensor(self.label[index])
        return data, label 

if __name__ == "__main__":
    dataset = SubDataset(Data, Label)
    print(dataset)
    print("dataset的大小是:",dataset.__len__())
    print(len(dataset))
    print(dataset.__getitem__(0))
    print(dataset[0])


    print("创建DataLoader迭代器")
    dataloader = DataLoader.DataLoader(dataset,batch_size= 2, shuffle = False, num_workers= 4)
    '''
    对象数*batch_size就是数据集的大小__len__
    
    '''
    for i, item in enumerate(dataloader):
        print('i:', i)
        data, label = item
        print('data:', data)
        print('label:', label)
