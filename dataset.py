from torch.utils.data import Dataset
import torch
import csv
import os
import numpy as np


class StockDataset(Dataset):
    def __init__(self, dir_path):
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

        stock = self.stock_data[0]
        self.all_data = []
        index_box = np.random.randint(0, 200, size=40)
        index_box = np.unique(index_box)
        for i in index_box:  # 每5天数据做训练
            basevec_data = []
            start_index = i
            end_index = start_index + 5
            base_data = stock[start_index:end_index]
            tag_data = stock[end_index + 1]
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
                pctChg = float(day['pctChg'])
                v = torch.tensor([s_open, s_close, preclose, high, low, pctChg])
                basevec_data.append(v)
            basevec_data = torch.stack(basevec_data)
            self.all_data.append((basevec_data, tag))

    def __getitem__(self, item):
        return self.all_data[item][0] / 10, self.all_data[item][1]

    def __len__(self):
        return len(self.all_data)


if __name__ == '__main__':
    SH_path = "D:\stock"
    dataset = StockDataset(SH_path)
    data, tag = dataset[0]
    print(data, tag)
    print(len(dataset))
