from dataset import StockDataset
from net import Stocknet, CNNStock, StockTransNet
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

dirpath = r"D:\stock"
module_path = r"module/stock.pth"
test_loader = DataLoader(StockDataset(dirpath,train=False), batch_size=1000, shuffle=True)

net = StockTransNet().cuda()
opt = optim.Adam(net.parameters())
loss_fun = nn.NLLLoss()
summary = SummaryWriter("logs")

if os.path.exists(module_path):
    net.load_state_dict(torch.load(module_path))
    print("load success")
else:
    print("no module")

init_acc = 0.
net.eval()
for epoch in range(50000000):
    sum_train_loss = 0.
    sum_acc = 0.
    for i, (data, tag) in enumerate(test_loader):
        data, tag = data.cuda(), tag.cuda()
        tgt = torch.repeat_interleave(tag[:, None], 6, dim=1)
        tgt = tgt[:, None, :].float()
        out = net(data, tgt)
        loss = loss_fun(out, tag)

        # for name, parms in net.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)

        sum_train_loss += loss.item()
        out = out.detach().cpu().numpy()
        tag = tag.detach().cpu().numpy()
        acc = np.mean(np.equal(np.argmax(out, axis=1), tag))
        sum_acc += acc

    avg_train_loss = sum_train_loss / len(test_loader)
    avg_acc = sum_acc / len(test_loader)
    summary.add_scalar("loss", avg_train_loss, epoch)
    summary.add_scalar("acc", avg_acc, epoch)
    print("epoch:{} loss:{} acc:{:.3f}%".format(epoch, avg_train_loss, avg_acc * 100))