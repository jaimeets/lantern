import lantern.nn as nn
import numpy as np
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2,3)
        self.l2 = nn.Linear(3,4)
        self.l3 = nn.Linear(4,3)
        self.l4 = nn.Linear(3,2)
    def forward(self, X):
        out = self.l1(X)
        out = nn.relu(out)
        out = self.l2(out)
        out = nn.relu(out)
        out = self.l3(out)
        out = nn.relu(out)
        out = self.l4(out)
        return out

model = NeuralNetwork()
inp = np.array([[1,2]])
print(model(inp))
print(model(inp))
print(model(inp))
print(model(inp))
print(model(inp))
print(model(inp))
print(model(inp))


        