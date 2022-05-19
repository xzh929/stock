from torch.utils.data import Dataset
import torch
import csv
import os
import numpy as np
import math


class StockDataset(Dataset):
    def __init__(self, dir_path, train=True):
        self.stock_data = []
        for dirname in os.listdir(dir_path):
            for filename in os.listdir(os.path.join(dir_path, dirname)):
                with open(os.path.join(dir_path, dirname, filename)) as f:
                    file = csv.reader(f)
                    cls = []
                    year_data = []
                    for i, line in enumerate(file):
                        if i < 1:
                            cls = line
                        elif i >= 1:
                            day_data = dict(zip(cls, line))
                            year_data.append(day_data)
                self.stock_data.append(year_data)  # stock_data[[{}]] 按每支股票存放其一年的数据并用字典存放每日数据

        self.all_data = []
        for stock in self.stock_data[:1]:  # 取股票数据
            if len(stock) < 150: continue
            if train:
                index_box = np.random.randint(0, len(stock) - 50, size=int((len(stock) - 50) / 5))
                index_box = np.unique(index_box)
            else:
                index_box = np.random.randint(len(stock) - 50, len(stock) - 5, size=(50 // 5))
                index_box = np.unique(index_box)
            for start_index in index_box:  # 每5天数据做训练
                basevec_data = []
                end_index = start_index + 5
                base_data = stock[start_index:end_index]
                tag_data = stock[end_index]
                tgt = torch.tensor(
                    [[float(tag_data['open']), float(tag_data['close']), float(tag_data['preclose']),
                      float(tag_data['high']), float(tag_data['low']), float(tag_data['volume']) / (1e+7)]])
                if float(tag_data['pctChg']) > 0:
                    tag = torch.tensor(1)
                else:
                    tag = torch.tensor(0)
                for day in base_data:
                    s_open = float(day['open'])
                    s_close = float(day['close'])
                    preclose = float(day['preclose'])
                    high = float(day['high'])
                    low = float(day['low'])
                    volume = float(day['volume']) / (1e+7)
                    v = torch.tensor([s_open, s_close, preclose, high, low, volume])
                    basevec_data.append(v)
                basevec_data = torch.stack(basevec_data)
                self.all_data.append([basevec_data, tag])
        self.all_data = self.__normlize(self.all_data)

    def __getitem__(self, item):
        place = torch.floor(self.all_data[item][0][0][0])
        place = place.item()
        if place > 1:
            place = int(math.log10(place)) + 1
            sub = 10 ** place
            return self.all_data[item][0] / sub, self.all_data[item][1]
        else:
            return self.all_data[item][0], self.all_data[item][1]

    def __normlize(self, data):
        price_data = []
        volume_data = []
        for all_data in data:
            price_data.append(all_data[0][:, :5])
            volume_data.append(all_data[0][:, -1])
        price_data = torch.stack(price_data)
        volume_data = torch.stack(volume_data)
        print(data[:][0])
        data[:][0][:, :5] = (data[0][:, :5] - price_data.mean()) / torch.std(price_data)
        data[:][0][:, -1] = (data[0][:, -1] - volume_data.mean()) / torch.std(volume_data)
        return data

    def __len__(self):
        return len(self.all_data)


if __name__ == '__main__':
    SH_path = "D:\stock"
    dataset = StockDataset(SH_path)
    data, tag = dataset[0]
    # print(data, tag)
    # print(len(dataset))
