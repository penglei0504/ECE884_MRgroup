import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import make_dataset

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, (3,3), padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d((3,3), stride=1, padding=1)
    conv6 = nn.Linear(512,4096)
    conv7 = nn.Linear(4096,4096)
    conv8 = nn.Linear(4096,1000)
    layers += [pool5, conv6,nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True), conv8, nn.ReLU(inplace=True)]
    return layers
base =  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,  'M',512, 512, 512]

'''Define hyperparameters '''
batch_size = 1          # batch size
learning_rate = 1e-5    # learning rate
num_epoches = 10        # epoch times

data_dir="E:\\ECE884-489286\\ECE884PL\\MRdata"

# abnormal acl meniscus
injury_type='meniscus'

# axial coronal sagittal
plane='sagittal'
train_dataset = make_dataset(data_dir, 'train', injury_type, plane, device='cuda')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_dataset = make_dataset(data_dir, 'valid', injury_type, plane, device='cuda')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

'''Build network based on VGG16 network architecture'''
class VGG16(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            #1
            nn.Conv2d(3,64,(3,3),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #2
            nn.Conv2d(64,64,(3,3),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #3
            nn.Conv2d(64,128,(3,3),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            #4
            nn.Conv2d(128,128,(3,3),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #5
            nn.Conv2d(128,256,(3,3),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #6
            nn.Conv2d(256,256,(3,3),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #7
            nn.Conv2d(256,256,(3,3),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #8
            nn.Conv2d(256,512,(3,3),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #9
            nn.Conv2d(512,512,(3,3),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #10
            nn.Conv2d(512,512,(3,3),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #11
            nn.Conv2d(512,512,(3,3),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #12
            nn.Conv2d(512,512,(3,3),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #13
            nn.Conv2d(512,512,(3,3),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.AvgPool2d(kernel_size=1,stride=1),
            )
        self.classifier = nn.Sequential(
            #14
            nn.Linear(512,4096),
            nn.ReLU(True),
            nn.Dropout(),
            #15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            #16
            nn.Linear(4096,num_classes),
            )
        #self.classifier = nn.Linear(512, 10)
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=None, padding=0)

    def forward(self, x):
        x=x.squeeze()
        outseries = torch.tensor([]).to(x.device)
        for series in x:
            with torch.no_grad():
                outsa=self.features(series.unsqueeze(0))
            outss=self.avg_pool(outsa)
            outseries = torch.cat((outseries, outss), 0)
            outseries = outseries.max(dim=0, keepdim=True)[0]

        outseries = outseries.max(dim=0, keepdim=True)[0].squeeze()
        #        print(out.shape)
        outseries = outseries.view(outseries.size(0), -1)
        #        print(out.shape)
        outseries=outseries.transpose(0,1)
        outseries = self.classifier(outseries)
        #        print(out.shape)
        return outseries

'''Create VGG16 network model, detect if having GPUs or not'''
model = VGG16()
use_gpu = torch.cuda.is_available()  # If GPUs or not
if use_gpu:
    model = model.cuda()

'''Define loss and optimizer'''
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

'''Train model'''
for epoch in range(num_epoches):
    print('*' * 25, 'epoch {}'.format(epoch + 1), '*' * 25)
    running_loss = 0.0
    running_acc = 0.0

    for i, data in tqdm(enumerate(train_loader, 1)):
        img, label = data
        # cuda
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        img = Variable(img)
        label = Variable(label)
        # Forward propagation
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)  # predict result
        if (pred == label[0][0]):
            num_correct=0
        else:
            num_correct=1
        # num_correct = (pred == label[0]).sum()
        accuracy = (pred == label[0][0]).float().mean()
        running_acc += num_correct
        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))))

    torch.save(model, 'ECE884.pkl')
    model.eval()  # 模型评估
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:  # 测试模型
        img, label = data
        if use_gpu:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        if (pred == label[0][0]):
            num_correct=0
        else:
            num_correct=1
        # num_correct = (pred == label).sum()
        eval_acc += num_correct
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_dataset)), eval_acc / (len(test_dataset))))
    print()

