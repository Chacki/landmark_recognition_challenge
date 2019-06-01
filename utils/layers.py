from torch import nn


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *inp):
        return self.func(*inp)
