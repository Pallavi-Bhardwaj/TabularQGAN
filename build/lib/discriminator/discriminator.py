import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, input_length: int):
        super(Discriminator, self).__init__()
        self.dense1 = nn.Linear(int(input_length), 2 * int(input_length))
        self.dense2 = nn.Linear(2 * int(input_length), 1)

    def forward(self, x):
        h = F.leaky_relu(self.dense1(x))
        h = F.leaky_relu(self.dense2(h))
        return F.sigmoid(h)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data, gain=10)
        nn.init.constant_(m.bias.data, 1)
