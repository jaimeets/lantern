import numpy as np
from abc import ABC, abstractmethod
class Linear:
     def __init__(self, din, dout):
          self.w = np.random.normal(loc=0, scale=1, size=(din, dout))
          self.b = np.random.normal(loc=0, scale=1, size=(1, dout))
     def __call__(self, X):
          return np.matmul(X,self.w) + self.b
     

class Module(ABC):
     def __init__(self):
          super().__init__()
     
     def __call__(self, X):
          return self.forward(X)

     @abstractmethod
     def forward(self, X):
          pass

#activation fns
def relu(x):
    return np.where(x>0,x,0)
