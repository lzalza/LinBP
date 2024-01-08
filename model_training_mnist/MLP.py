import torch
import torchvision
from torch.autograd import Variable
import torch.utils.data.dataloader as Data
import numpy as np
import torch.nn.functional as F
import random
import argparse


parser = argparse.ArgumentParser(description='MNIST Training')
parser.add_argument('--lr', '-lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--epoch', '-e', default=50, type=int, help='learning rate')
parser.add_argument('--linbp', '-lin', action='store_true',
                    help='use LinBP or not')
args = parser.parse_args()


seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

input = torch.randn(3,3)
loss = torch.nn.NLLLoss()
target = torch.tensor([0,1,2])
print(loss(input,target))
loss = torch.nn.CrossEntropyLoss()
print(loss(input,target))

train_data = torchvision.datasets.MNIST(
    './mnist', train=True, transform=torchvision.transforms.ToTensor(), download=True
)
test_data = torchvision.datasets.MNIST(
    './mnist', train=False, transform=torchvision.transforms.ToTensor()
)
'''
print("train_data:", train_data.data.size())
print("train_labels:", train_data.targets.size())
print("test_data:", test_data.data.size())
print("test_labels:", test_data.targets.size())
'''
train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=64)



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(784, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10))
        

    def forward(self, x,LinBP=False):
        cnt = 0
        if(LinBP):
            flag = [1,0,0]
        else:
            flag = [1,1,1]
        for i, module_name in enumerate(self.layer1):
            
            if(isinstance(module_name, torch.nn.ReLU) and flag[cnt] == 0):
                x_p = F.relu(-x)
                x = x + x_p.data
            else:
                x = self.layer1[i](x)
            if(isinstance(module_name, torch.nn.ReLU)):
                cnt = cnt + 1
        
        return x


model = Net()
#print(model)

optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
loss_func = torch.nn.CrossEntropyLoss()

_train_loss = []
_test_loss = []
_train_acc = []
_test_acc = []

for epoch in range(args.epoch):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_loader:
        #batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        batch_x = batch_x.view(-1,28*28)
        out = model(batch_x,args.linbp)
        loss = loss_func(out, batch_y)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_data)), train_acc / (len(train_data))))
    
    if(epoch % 1 == 0):
        _train_loss.append(train_loss / (len(train_data)))
        _train_acc.append(train_acc / (len(train_data)))

    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x, requires_grad=False), Variable(batch_y, requires_grad=False)
        batch_x = batch_x.view(-1,28*28)
        out = model(batch_x,args.linbp)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_data)), eval_acc / (len(test_data))))

    if(epoch % 1 == 0):
        _test_loss.append(eval_loss / (len(test_data)))
        _test_acc.append(eval_acc / (len(test_data)))

print(_train_loss)
print(_test_loss)
print(_train_acc)
print(_test_acc)

