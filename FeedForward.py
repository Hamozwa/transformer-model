import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_1, d_2, d_3):
        super().__init__()
        self.linear1 = nn.Linear(d_1, d_2)
        self.linear2 = nn.Linear(d_2, d_1)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x