from torch import nn
import torch


class Stocknet(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformerlayer = nn.TransformerEncoderLayer(d_model=6, nhead=2, batch_first=True)
        self.transnet = nn.TransformerEncoder(self.transformerlayer, num_layers=4)
        self.fc_out = nn.Sequential(
            nn.Linear(5 * 6, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        trans_out = self.transnet(x)
        trans_out = trans_out.reshape(-1, 5 * 6)
        cls = self.fc_out(trans_out)
        return cls


if __name__ == '__main__':
    a = torch.randn(2, 5, 6)
    net = Stocknet()
    out = net(a)
    print(out.shape)
