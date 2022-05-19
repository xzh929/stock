from torch import nn
import torch
import math


class Stocknet(nn.Module):
    def __init__(self):
        super().__init__()
        self.map = nn.Sequential(
            nn.Conv1d(6, 16, 1)
        )
        self.pos_encoder = PositionalEncoding(6)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc_out = nn.Sequential(
            nn.Linear(5 * 16, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.map(x.permute(0, 2, 1))
        encoder_out = self.encoder(x.permute(0, 2, 1)).reshape(-1, 5 * 16)
        cls = self.fc_out(encoder_out)
        return cls


class LearnedPositionEncoding(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0), :]
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CNNStock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(6, 12, 3, 1),
            nn.BatchNorm1d(12),
            nn.Hardswish(),
            nn.Conv1d(12, 24, 3, 1),
            nn.BatchNorm1d(24),
            nn.Hardswish(),
        )
        self.fc = nn.Sequential(
            nn.Linear(24 * 1, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.layer(x)
        out = out.reshape(-1, 24 * 1)
        cls = self.fc(out)
        return cls


class StockTransNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_encoder = PositionalEncoding(6)
        self.net = nn.Transformer(d_model=6,
                                  nhead=2,
                                  dim_feedforward=512,
                                  num_encoder_layers=1,
                                  num_decoder_layers=1,
                                  batch_first=True)
        self.fc_out = nn.Sequential(
            nn.Linear(6, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, src, tgt):
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        trans_out = self.net(src, tgt)
        trans_out = trans_out[:, -1]
        return self.fc_out(trans_out)


if __name__ == '__main__':
    a = torch.randn(2, 5, 6)
    b = torch.randn(2, 1, 6)
    net = Stocknet()
    # pe = PositionalEncoding(6)
    # net2 = CNNStock()
    # net3 = StockTransNet()
    out = net(a)
    # y = pe(a)
    # out2 = net2(a)
    # out3 = net3(a, b)
    print(out.shape)
    # print(out2.shape)
    # print(out3.shape)
    # print(y.shape)
