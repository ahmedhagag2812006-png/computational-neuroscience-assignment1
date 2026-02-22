import torch
import torch.nn as nn
x = torch.tensor([0.05, 0.10], dtype=torch.float32)
class SimpleANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)   
        self.fc2 = nn.Linear(2, 1) 
        with torch.no_grad():
            self.fc1.weight.uniform_(-0.5, 0.5)
            self.fc2.weight.uniform_(-0.5, 0.5)
            self.fc1.bias.fill_(0.5)  # b1
            self.fc2.bias.fill_(0.7)  # b2

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        y = torch.tanh(self.fc2(h))
        return y
model = SimpleANN()
output = model(x)
print("Final Output:", output.item())