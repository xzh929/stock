import random
import numpy as np
import torch
import math
from dataset import StockDataset
from torch.utils.data import DataLoader
from torch import nn

# dirpath = r"D:\stock"
embedding = nn.Embedding(15, 3, padding_idx=0)
# a = torch.randint(10, (2, 5))
# pe = torch.zeros(2, 1, 6)
# b = np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]
# b = torch.randn(5)
# f = torch.floor(torch.abs(a))
# c = torch.tensor(1)
# d = torch.repeat_interleave(c,6)
# d = d[None,:]
# print(d)
# print(torch.mean(torch.eq(a,b).float()))
# print(a.item(),f.item())
# c = int(math.log10(100)) + 1
# d = 10 ** c
# print(c, d)
# train_loader = DataLoader(StockDataset(dirpath), batch_size=len(StockDataset(dirpath)), shuffle=True)
# data = next(iter(train_loader))
# print(data[0].mean(), data[0].std())
batch = [['i', 'am', 'a', 'boy', '.'], ['i', 'am', 'very', 'lucky', '.'], ['how', 'are', 'you', '?']]


# 带批次的word2id
def word2id(batch_list):
    maps = {}
    for list in batch_list:
        for item in list:
            if item not in maps:
                maps[item] = len(maps) + 1
    maps['eos'] = 1
    maps['pad'] = 0
    return maps


def list_word2id(batch_list):
    maps = word2id(batch_list)
    max_length = len(max(batch_list))
    for list in batch_list:
        list_length = len(list)
        if list_length < max_length:
            for i in range(max_length - list_length):
                list.append('pad')
        for i, item in enumerate(list):
            list[i] = maps[item]
        list.append(maps['eos'])
    return batch_list


list_id_map = word2id(batch)
list2id = list_word2id(batch)
list_embed = embedding(torch.tensor(list2id))
print(list_embed.shape)
