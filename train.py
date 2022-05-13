from dataset import StockDataset
from net import Stocknet
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dirpath = "D:\stock"
train_loader = DataLoader(StockDataset(dirpath), batch_size=10, shuffle=True)
net = Stocknet().cuda()
opt = optim.Adam(net.parameters())
loss_fun = nn.CrossEntropyLoss()
summary = SummaryWriter("logs")

for epoch in range(10000):
    sum_train_loss = 0.
    sum_acc = 0.
    for i, (data, tag) in enumerate(train_loader):
        data, tag = data.cuda(), tag.cuda()
        out = net(data)
        loss = loss_fun(out, tag)

        opt.zero_grad()
        loss.backward()
        opt.step()

        sum_train_loss += loss.item()
        acc = torch.mean(torch.eq(torch.argmax(out, dim=1), tag).float())
        sum_acc += acc

    avg_train_loss = sum_train_loss / len(train_loader)
    avg_acc = sum_acc / len(train_loader)
    summary.add_scalar("loss", avg_train_loss, epoch)
    summary.add_scalar("acc", avg_acc, epoch)
    print("loss:{} acc:{}".format(avg_train_loss, avg_acc))
