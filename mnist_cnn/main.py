import torch 
import os 
import torch.nn as nn
import torch.optim as optim 
from torch.autograd import Variable 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
root = './mnist/'
use_gpu = False
def write_train_file(path):
    f = open(path+'train.txt','w')
    data_path = path + 'train/'
    for i in os.listdir(data_path):
        for j in os.listdir(data_path + str(i)):
            img_path = data_path + str(i) + '/' + str(j)
            label = i
            f.write(img_path + ' ' + str(label) + '\n')
    f.close()

def write_test_file(path):
    f = open(path+'test.txt', 'w') 
    data_path = path + 'test/'
    for i in os.listdir(data_path):
        for j in os.listdir(data_path + str(i)):
            img_path = data_path + str(i) + '/' + str(j) 
            label = i
            f.write(img_path + ' ' + str(label) + '\n')
    f.close()   
# 生成txt文件
write_train_file(root)
write_test_file(root)

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)
    

#create the net and training 
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 3 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out



if __name__ == '__main__':
    train_data=MyDataset(txt=root+'train.txt', transform=transforms.ToTensor())
    test_data=MyDataset(txt=root+'test.txt', transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True, num_workers=6)
    test_loader = DataLoader(dataset=test_data, batch_size=16,num_workers=6)

    if  use_gpu:
        model = Net().cuda()
    else:
        model = Net()

    print(model)

    optimizer = optim.Adam(model.parameters()) 
    loss_func = nn.CrossEntropyLoss() 

    for epoch in range(10):
        print('epoch {} '.format(epoch+1))

        train_loss = 0 
        train_acc = 0 
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = Variable(batch_x),  Variable(batch_y)
            if use_gpu:           
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda() 
            out = model(batch_x)
            loss = loss_func(out, batch_y) 
            train_loss += loss.item() 
            pred = torch.max(out, 1)[1] 
            train_correct = (pred==batch_y).sum() 
            train_acc += train_correct.item() 
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data)), train_acc / (len(train_data))))
        # evaluation--------------------------------
        model.eval()
        '''
        model.train() ：启用 BatchNormalization 和 Dropout

        model.eval() ：不启用 BatchNormalization 和 Dropout
        '''
        eval_loss = 0.
        eval_acc = 0.
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = Variable(batch_x, requires_grad=False), Variable(batch_y, requires_grad=False)
            if use_gpu:           
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda() 
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.item()
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data)), eval_acc / (len(test_data))))







        